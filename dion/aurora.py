import torch
import torch.distributed as dist
from collections import defaultdict
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.distributed.tensor import DTensor
from torch.optim.optimizer import ParamsT
from typing import Callable, Generator, List, Optional, Tuple

from .megabatch_base import (
    DistributedOrthoBase,
    ShardInfo,
    adjust_lr_spectral_norm,
    adjust_lr_rms_norm,
    adjust_lr_keller_muon,
)
from .opt_utils import AsyncTask, to_local
from .muon import muon_update_pre_orthogonalize, muon_update_post_orthogonalize


class Aurora(DistributedOrthoBase):
    """
    Distributed Aurora optimizer for PyTorch FSDP2.

    Aurora applies leverage-uniform polar decomposition via alternating
    row-normalization and orthogonalization, preventing neuron death in
    tall matrices (e.g., SwiGLU up/gate projections) while maintaining
    orthogonality.

    Distributed topology is read per-parameter from each DTensor's own
    ``device_mesh`` and ``placements``; see :class:`DistributedOrthoBase`
    for details.

    Args:
        params: Parameters for the optimizer.
        lr: Base learning rate. For Aurora, this will be scaled based on the
            matrix dimensions.
        mu: Momentum factor for Aurora algorithm.
        pp_iterations: Number of alternating row-normalize + polar iterations
            (default 2, following the Aurora paper).
        pp_beta: Damping exponent for the diagonal preconditioner update
            (default 0.5, following the Aurora paper).
        betas: Tuple of (beta1, beta2) for AdamW and Lion algorithms.
        weight_decay: Weight decay factor.
        cautious_wd: Whether to apply weight decay only where update and
            parameter signs align.
        epsilon: Small value to avoid division by zero.
        nesterov: Whether to use Nesterov momentum (default True, per paper).
        adjust_lr: How to adjust the learning rate for Aurora updates
            ("spectral_norm", "rms_norm", "keller_muon", or None).
        flatten: Whether to flatten 3D+ tensors to 2D for Aurora updates.
        use_gram_newton_schulz: Whether to use Gram Newton-Schulz.
        use_triton: Whether to use Triton kernel for Newton-Schulz.
        use_polar_express: Whether to use Polar Express orthogonalization.
        newton_schulz_func: Custom Newton-Schulz function.

    Aurora optimizer: https://blog.tilderesearch.com/blog/aurora
    Reference implementation: https://github.com/tilde-research/aurora-release
    FSDP2 communication pattern: https://www.essential.ai/blog/infra
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 0.01,
        mu: float = 0.95,
        pp_iterations: int = 2,
        pp_beta: float = 0.5,
        betas: Tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.01,
        cautious_wd: bool = False,
        epsilon: float = 1e-8,
        nesterov: bool = True,
        adjust_lr: Optional[str] = "keller_muon",
        flatten: bool = False,
        use_gram_newton_schulz: bool = False,
        use_triton: bool = False,
        use_polar_express: bool = True,
        newton_schulz_func: Optional[Callable] = None,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if mu < 0.0:
            raise ValueError(f"Invalid momentum factor (mu): {mu}")
        if pp_iterations < 1:
            raise ValueError(f"pp_iterations must be >= 1, got {pp_iterations}")
        if pp_beta <= 0.0:
            raise ValueError(f"pp_beta must be positive, got {pp_beta}")
        if len(betas) != 2 or betas[0] < 0.0 or betas[1] < 0.0:
            raise ValueError(f"Invalid betas: {betas}")
        if adjust_lr not in ("spectral_norm", "rms_norm", "keller_muon", None):
            raise ValueError(
                f"Invalid adjust_lr value: {adjust_lr}. "
                f"Must be 'spectral_norm', 'rms_norm', 'keller_muon', or None."
            )

        defaults = dict(
            lr=lr,
            mu=mu,
            pp_iterations=pp_iterations,
            pp_beta=pp_beta,
            beta1=betas[0],
            beta2=betas[1],
            weight_decay=weight_decay,
            cautious_wd=cautious_wd,
            algorithm="aurora",
            step=0,
            epsilon=epsilon,
            nesterov=nesterov,
            flatten=flatten,
            adjust_lr=adjust_lr,
        )
        super().__init__(
            params, "aurora", defaults,
            use_gram_newton_schulz=use_gram_newton_schulz,
            use_triton=use_triton,
            use_polar_express=use_polar_express,
            newton_schulz_func=newton_schulz_func,
        )

    def _get_shard_info(self, param: Tensor, group: dict) -> ShardInfo:
        info = super()._get_shard_info(param, group)
        if info.is_matrix_sharded and info.sharded_tensor_dim == param.ndim - 1:
            raise NotImplementedError(
                "Aurora currently does not support parameters sharded along the "
                "last dimension. Please avoid shards at dim -1."
            )
        return info

    def _create_ortho_tasks(
        self, param_groups: List[dict]
    ) -> Generator["AsyncTask", None, None]:
        """
        Mega-batched Aurora task creation: groups ALL same-shape parameters
        into a single task to minimize communication rounds and kernel launches.
        """
        for group in param_groups:
            assert group["algorithm"] == self._algo_name
            assert all(
                p.ndim >= 2 for p in group["params"]
            ), "Aurora optimizer only supports matrix parameters."

            group_params = [p for p in group["params"] if p.grad is not None]
            if not group_params:
                continue

            common_args = dict(
                lr=torch.tensor(group["lr"]),
                momentum=torch.tensor(group["mu"]),
                pp_iterations=group["pp_iterations"],
                pp_beta=group["pp_beta"],
                weight_decay=torch.tensor(group["weight_decay"]),
                epsilon=group["epsilon"],
                nesterov=group["nesterov"],
                flatten=group["flatten"],
                adjust_lr=group["adjust_lr"],
                newton_schulz_func=self._newton_schulz_func,
                cautious_wd=group["cautious_wd"],
            )

            shape_groups: dict[tuple, list] = defaultdict(list)
            for p in group_params:
                if isinstance(p, DTensor):
                    key = (p.shape, p.placements, p.device_mesh, p.dtype)
                else:
                    key = (p.shape, None, None, p.dtype)
                shape_groups[key].append(p)

            for params in shape_groups.values():
                gradients = [p.grad for p in params]
                states = [self._get_or_initialize_state(p, self._algo_name) for p in params]
                momentums = [s["momentum"] for s in states]

                shard_info = self._get_shard_info(params[0], group)

                yield AsyncTask(
                    aurora_update_megabatch_async(
                        X=params,
                        G=gradients,
                        M=momentums,
                        shard_dim=shard_info.sharded_tensor_dim,
                        device_rank=shard_info.device_rank,
                        world_size=shard_info.world_size,
                        process_group=shard_info.process_group,
                        **common_args,
                    )
                )


def aurora_update_megabatch_async(
    X: List[Tensor],
    G: List[Tensor],
    M: List[Tensor],
    lr: Tensor,
    momentum: Tensor,
    pp_iterations: int,
    pp_beta: float,
    weight_decay: Tensor,
    epsilon: Tensor,
    nesterov: bool,
    flatten: bool,
    adjust_lr: Optional[str],
    device_rank: int,
    world_size: int,
    shard_dim: Optional[int] = None,
    process_group: Optional[ProcessGroup] = None,
    newton_schulz_func: Optional[Callable] = None,
    cautious_wd: bool = False,
) -> Generator[None, None, None]:
    """
    Mega-batched Aurora update: processes ALL same-shape parameters in one
    communication round instead of world_size-sized batches.
    """
    N = len(X)
    assert N == len(G) == len(M)

    # Pre-orthogonalize: update momentum (same as Muon/NorMuon)
    U = muon_update_pre_orthogonalize(
        G=to_local(G), M=to_local(M), momentum=momentum, nesterov=nesterov,
    )

    # Convert shard_dim to negative for comm_dim
    comm_dim = (shard_dim - X[0].ndim) if shard_dim is not None else None

    # On the sharded path X[0] must still be a DTensor, so .shape[comm_dim]
    # is the unsharded global size. The Aurora megabatch path needs that size
    # both to make all-to-all inputs uniform and to exclude padding rows from
    # its leverage-normalization target.
    if comm_dim is not None:
        if not isinstance(X[0], DTensor):
            raise TypeError(
                "Sharded path requires X[0] to be a DTensor so .shape gives "
                f"the global size; got {type(X[0]).__name__}."
            )
        global_comm_dim_size = X[0].shape[comm_dim]
    else:
        global_comm_dim_size = None

    # Aurora D-iteration polar via megabatch communication
    U = yield from megabatch_aurora_orthogonalize_async(
        U,
        comm_dim=comm_dim,
        device_rank=device_rank,
        world_size=world_size,
        process_group=process_group,
        newton_schulz_func=newton_schulz_func,
        flatten=flatten,
        epsilon=epsilon,
        pp_iterations=pp_iterations,
        pp_beta=pp_beta,
        global_comm_dim_size=global_comm_dim_size,
    )

    # Compute scaled learning rate
    if adjust_lr is None:
        adjusted_lr = lr
    elif adjust_lr == "spectral_norm":
        adjusted_lr = adjust_lr_spectral_norm(lr, X[0].shape, flatten=flatten)
    elif adjust_lr == "rms_norm":
        adjusted_lr = adjust_lr_rms_norm(lr, X[0].shape, flatten=flatten)
    elif adjust_lr == "keller_muon":
        adjusted_lr = adjust_lr_keller_muon(lr, X[0].shape, flatten=flatten)
    else:
        raise ValueError(f"Unknown adjust_lr value: {adjust_lr}")

    # Post-orthogonalize: apply update (same as Muon/NorMuon)
    muon_update_post_orthogonalize(
        X=to_local(X),
        U=U,
        base_lr=lr,
        adjusted_lr=adjusted_lr,
        weight_decay=weight_decay,
        cautious_wd=cautious_wd,
    )


def megabatch_aurora_orthogonalize_async(
    U: List[Tensor],
    comm_dim: Optional[int],
    device_rank: int,
    world_size: int,
    process_group: Optional[ProcessGroup],
    newton_schulz_func: Callable,
    flatten: bool,
    epsilon: Tensor,
    pp_iterations: int,
    pp_beta: float,
    global_comm_dim_size: Optional[int],
) -> Generator[None, None, List[Tensor]]:
    """
    Megabatch communication + Aurora D-iteration polar decomposition.

    Mirrors ``megabatch_orthogonalize_async`` from megabatch_base.py exactly
    in structure, but calls ``aurora_process_matrices`` instead of
    ``muon_update_newton_schulz``.

    This is a generator that yields at async communication points. The
    result is recovered via ``yield from``.
    """
    N = len(U)

    # Pad to divisible by world_size (needed by both distributed paths)
    if process_group is not None and (N > 1 or comm_dim is not None):
        pad_n = (world_size - N % world_size) % world_size
        U_work = U + [torch.zeros_like(U[0])] * pad_n if pad_n > 0 else U
        N_total = len(U_work)
        per_rank = N_total // world_size
    else:
        U_work = U

    if comm_dim is not None and process_group is not None:
        # --- Mega-batched sharded FSDP2 path ---
        if global_comm_dim_size is None:
            raise ValueError(
                "global_comm_dim_size must be passed when comm_dim is not "
                "None; callers should pass the unsharded DTensor's global "
                "size along comm_dim."
            )
        padded_local_size = (global_comm_dim_size + world_size - 1) // world_size
        original_local_size = U_work[0].size(comm_dim)
        if padded_local_size < original_local_size:
            raise RuntimeError(
                f"padded_local_size ({padded_local_size}) < this rank's "
                f"local size ({original_local_size}); FSDP2 contiguous-"
                f"chunking assumption violated (global_comm_dim_size="
                f"{global_comm_dim_size}, world_size={world_size})."
            )

        if padded_local_size != original_local_size:
            pad_spec = [0, 0] * (-comm_dim - 1) + [
                0,
                padded_local_size - original_local_size,
            ]
            U_work = [torch.nn.functional.pad(u, pad_spec) for u in U_work]

        input_chunks = [
            torch.stack(U_work[r * per_rank : (r + 1) * per_rank])
            for r in range(world_size)
        ]

        output_chunks = [torch.empty_like(c) for c in input_chunks]
        work = dist.all_to_all(
            output_chunks, input_chunks, group=process_group, async_op=True
        )
        yield
        work.wait()

        full_matrices = torch.cat(output_chunks, dim=comm_dim)

        # Unlike the standard Muon polar transform, Aurora's target row norm
        # depends explicitly on the matrix height. Exclude synthetic padding
        # rows while processing so uneven sharding is numerically identical to
        # processing the true global matrix, then restore the rows only for the
        # equal-sized return collective.
        real_matrices = full_matrices.narrow(
            comm_dim, 0, global_comm_dim_size
        ).contiguous()
        real_matrices = aurora_process_matrices(
            real_matrices,
            newton_schulz_func=newton_schulz_func,
            flatten=flatten,
            epsilon=epsilon,
            pp_iterations=pp_iterations,
            pp_beta=pp_beta,
        )

        padded_global_size = padded_local_size * world_size
        if padded_global_size != global_comm_dim_size:
            pad_spec = [0, 0] * (-comm_dim - 1) + [
                0,
                padded_global_size - global_comm_dim_size,
            ]
            full_matrices = torch.nn.functional.pad(real_matrices, pad_spec)
        else:
            full_matrices = real_matrices

        split_chunks = [
            s.contiguous()
            for s in torch.tensor_split(full_matrices, world_size, dim=comm_dim)
        ]

        recv_chunks = [torch.empty_like(c) for c in split_chunks]
        work = dist.all_to_all(
            recv_chunks, split_chunks, group=process_group, async_op=True
        )
        yield
        work.wait()

        result = [
            recv_chunks[r][i].narrow(comm_dim, 0, original_local_size).contiguous()
            for r in range(world_size)
            for i in range(per_rank)
        ]
        return result[:N]

    elif N > 1 and process_group is not None:
        # --- Mega-batched non-sharded path ---
        start = device_rank * per_rank
        my_matrices = torch.stack(U_work[start : start + per_rank])
        my_matrices = aurora_process_matrices(
            my_matrices,
            newton_schulz_func=newton_schulz_func,
            flatten=flatten,
            epsilon=epsilon,
            pp_iterations=pp_iterations,
            pp_beta=pp_beta,
        )

        all_chunks = [torch.empty_like(my_matrices) for _ in range(world_size)]
        work = dist.all_gather(
            all_chunks, my_matrices.contiguous(), group=process_group, async_op=True
        )
        yield
        work.wait()

        result = [all_chunks[r][i] for r in range(world_size) for i in range(per_rank)]
        return result[:N]

    elif N == 1:
        return [
            aurora_process_matrices(
                U[0],
                newton_schulz_func=newton_schulz_func,
                flatten=flatten,
                epsilon=epsilon,
                pp_iterations=pp_iterations,
                pp_beta=pp_beta,
            )
        ]

    else:
        # N > 1, no process_group (single GPU or batch-sharded 3D)
        stacked = torch.stack(U)
        stacked = aurora_process_matrices(
            stacked,
            newton_schulz_func=newton_schulz_func,
            flatten=flatten,
            epsilon=epsilon,
            pp_iterations=pp_iterations,
            pp_beta=pp_beta,
        )
        return [stacked[i] for i in range(N)]


def aurora_process_matrices(
    X: Tensor,
    newton_schulz_func: Callable,
    flatten: bool,
    epsilon: Tensor,
    pp_iterations: int,
    pp_beta: float,
) -> Tensor:
    """
    Aurora leverage-uniform polar decomposition.

    For a tall matrix M (rows > cols), iteratively applies:
    1. Diagonal row-preconditioning to target uniform row norms
    2. Polar factor via Newton-Schulz

    This replaces both the standard Newton-Schulz orthogonalization AND
    the NorMuon row-normalization post-processing with a single unified
    algorithm.

    Args:
        X: Input tensor (may be batched/stacked, may need flattening).
        newton_schulz_func: Polar factor function (signature func(Tensor, epsilon) -> Tensor).
        flatten: Whether to flatten 3D+ tensors to 2D.
        epsilon: Small value for numerical stability.
        pp_iterations: Number of D-update iterations.
        pp_beta: Damping exponent for diagonal preconditioner.
    """
    original_shape = X.shape

    # Flatten handling (same as muon_update_newton_schulz in megabatch_base.py)
    if flatten and X.ndim >= 3:
        X = X.flatten(start_dim=1)
    elif X.ndim >= 4:
        X = X.flatten(end_dim=-3)

    m, n = X.shape[-2], X.shape[-1]

    if m == n:
        # Square: standard polar (no leverage freedom to exploit).
        result = newton_schulz_func(X, epsilon=epsilon)
        return result.reshape(original_shape)

    # Ensure we work on tall orientation (m >= n) for proper row-norm targeting.
    transposed = m < n
    if transposed:
        X = X.mT
        m, n = n, m

    # Aurora D-iteration polar (Algorithm 1 from the Aurora paper).
    # Matches the reference implementation at:
    # https://github.com/tilde-research/aurora-release/blob/main/src/aurora.py
    X32 = X.to(torch.float32)
    target_row_sq = n / m

    # Initial diagonal preconditioner from row norms of the momentum buffer.
    row_norm = X32.norm(dim=-1, keepdim=True).clamp_min(epsilon)
    D = 1.0 / row_norm

    for k in range(pp_iterations):
        # Row-precondition then orthogonalize.
        U = newton_schulz_func(D * X32, epsilon=epsilon)

        if k < pp_iterations - 1:
            # Update D based on row norms of the polar output.
            row_sq = U.to(torch.float32).pow(2).sum(dim=-1, keepdim=True)
            row_sq = row_sq.clamp_min(epsilon * epsilon)
            D = D * (target_row_sq / row_sq).pow(pp_beta)

    if transposed:
        U = U.mT

    return U.reshape(original_shape)
