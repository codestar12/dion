import math
from dataclasses import dataclass
import torch
import torch.distributed as dist
from collections import defaultdict
from itertools import chain
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.distributed.tensor import DTensor
from torch.optim.optimizer import Optimizer, ParamsT
from typing import Callable, Generator, List, Optional

from .newton_schulz_triton import (
    TRITON_AVAILABLE,
    newton_schulz_triton,
    zeropower_via_newtonschulz5,
)
from .polar_express import polar_express, polar_express_triton
from .opt_utils import AsyncRuntime, AsyncTask, to_local
from .scalar_opts import adamw_update_foreach_async, lion_update_foreach_async


@dataclass(frozen=True)
class ShardInfo:
    """Per-parameter sharding metadata derived from a DTensor's placements.

    Matrix-dim shards require all-to-all on ``process_group`` to rebuild the
    full matrix; batch-dim shards are local (each rank's slice is complete).
    """

    is_batch_sharded: bool
    is_matrix_sharded: bool
    sharded_tensor_dim: Optional[int]
    process_group: Optional[ProcessGroup]
    device_rank: int
    world_size: int


_LOCAL_SHARD_INFO = ShardInfo(
    is_batch_sharded=False,
    is_matrix_sharded=False,
    sharded_tensor_dim=None,
    process_group=None,
    device_rank=0,
    world_size=1,
)


class DistributedOrthoBase(Optimizer):
    """Shared base class for distributed orthogonalization optimizers (Muon,
    NorMuon, Dion2).

    Distributed topology is read **per parameter** from each DTensor's own
    ``device_mesh`` and ``placements``; no global ``distributed_mesh`` is
    required. One instance can own parameters on different meshes (e.g.
    MoE experts on an EP-region FSDP mesh and non-experts on the global
    FSDP mesh). Plain ``torch.Tensor`` params are treated as single-GPU.

    Subclasses must implement ``_create_ortho_tasks()``.
    """

    def __init__(
        self,
        params: ParamsT,
        algo_name: str,
        defaults: dict,
        use_gram_newton_schulz: bool = False,
        use_triton: bool = False,
        use_polar_express: bool = True,
        newton_schulz_func: Optional[Callable] = None,
    ):
        super().__init__(params, defaults)
        self._algo_name = algo_name

        # Each distinct param shape triggers a Newton-Schulz recompile, so
        # many-layer (MoE) models blow past the default dynamo cache. Raise
        # the floor; user overrides set before this ctor are preserved.
        _DION_COMPILE_CACHE_FLOOR = 128
        if torch._dynamo.config.cache_size_limit < _DION_COMPILE_CACHE_FLOOR:
            torch._dynamo.config.cache_size_limit = _DION_COMPILE_CACHE_FLOOR
        if (
            torch._dynamo.config.accumulated_cache_size_limit
            < _DION_COMPILE_CACHE_FLOOR * 4
        ):
            torch._dynamo.config.accumulated_cache_size_limit = (
                _DION_COMPILE_CACHE_FLOOR * 4
            )

        # Orthogonalization function configuration
        if newton_schulz_func is not None:
            if not callable(newton_schulz_func):
                raise TypeError(
                    f"newton_schulz_func must be a callable function, got {type(newton_schulz_func)}"
                )
            self._newton_schulz_func = newton_schulz_func
        elif use_gram_newton_schulz:
            try:
                from gram_newton_schulz import GramNewtonSchulz
            except ImportError:
                raise ImportError(
                    "use_gram_newton_schulz=True requires the 'gram-newton-schulz' package, "
                    "which is not installed. "
                    "Install it with: pip install gram-newton-schulz"
                )
            use_polar_express = True
            _gns = GramNewtonSchulz(
                ns_use_kernels=use_triton,
                use_gram_newton_schulz=True,
                gram_newton_schulz_reset_iterations=[2],
                # Some compiler crashes were observed with mode="reduce-overhead" when we also compile the entire optimizer step.
                compile_kwargs=dict(fullgraph=True, mode="default"),
            )
            self._newton_schulz_func = lambda X, epsilon=None: _gns(X)
        elif use_polar_express and use_triton:
            self._newton_schulz_func = polar_express_triton
        elif use_polar_express:
            self._newton_schulz_func = polar_express
        elif use_triton:
            if not TRITON_AVAILABLE:
                raise ImportError(
                    "use_triton=True requires the 'triton' package, which is not installed. "
                    "Install it with: pip install dion[triton]  (or: pip install triton)"
                )
            self._newton_schulz_func = newton_schulz_triton
        else:
            self._newton_schulz_func = zeropower_via_newtonschulz5

        # Eagerly materialize optimizer state for every parameter, including
        # those that may never receive a gradient (frozen matrices, or MoE
        # experts that get no tokens on a given rank/step). State is otherwise
        # created lazily only for params with a gradient, so the set of keys in
        # state_dict() can differ across ranks (rank-asymmetric gradients) or
        # between save and resume. Distributed checkpointing (DCP) gathers
        # optimizer state collectively and assumes a rank-symmetric key set, so
        # the lazy path can mismatch or hang. Pre-populating keeps state_dict()
        # complete and identical across ranks. The per-step path reuses the same
        # (now non-empty) state, so this changes nothing numerically.
        # Ported from InternLM/xtuner v1/optim/muon.py (the equivalent __init__
        # loop), generalized here to the whole DistributedOrthoBase family via
        # the (possibly overridden) _get_or_initialize_state.
        self._state_prepopulated = False
        for group in self.param_groups:
            self._prepopulate_group_state(group)
        self._state_prepopulated = True

    def _prepopulate_group_state(self, group: dict) -> None:
        algo = group["algorithm"]
        for p in group["params"]:
            self._get_or_initialize_state(p, algo)

    def add_param_group(self, param_group: dict) -> None:
        super().add_param_group(param_group)
        # Keep the pre-population invariant for groups added after construction
        # so state_dict() stays complete and rank-symmetric. The guard skips the
        # add_param_group calls that Optimizer.__init__ makes before this class
        # finishes setup; the __init__ loop above pre-populates those groups.
        if getattr(self, "_state_prepopulated", False):
            self._prepopulate_group_state(self.param_groups[-1])

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        ortho_groups = []
        lion_groups = []
        adamw_groups = []

        for group in self.param_groups:
            group["step"] += 1
            algo = group["algorithm"]
            if algo == self._algo_name:
                ortho_groups.append(group)
            elif algo == "lion":
                lion_groups.append(group)
            elif algo == "adamw":
                adamw_groups.append(group)
            else:
                raise ValueError(f"Unknown algorithm: {algo}")

        ortho_tasks = self._create_ortho_tasks(ortho_groups)
        lion_tasks = self._create_lion_tasks(lion_groups)
        adamw_tasks = self._create_adamw_tasks(adamw_groups)

        all_tasks = chain(ortho_tasks, lion_tasks, adamw_tasks)
        runtime = AsyncRuntime(all_tasks, max_concurrent_tasks=3)
        runtime.run()

        return loss

    def _get_or_initialize_state(self, param: Tensor, algo: str) -> dict:
        """Get optimizer state, or lazy-initialize if it doesn't exist."""
        state = self.state[param]
        if not state:
            state["momentum"] = torch.zeros_like(param)
            if algo == "adamw":
                state["variance"] = torch.zeros_like(param)
        return state

    def _get_shard_info(self, param: Tensor, group: dict) -> ShardInfo:
        """Derive sharding metadata from ``param``'s DTensor placements.

        Batch-dim shards are local (no comm); a matrix-dim shard triggers
        all-to-all on its mesh-dim's process group.
        """
        if not isinstance(param, DTensor):
            return _LOCAL_SHARD_INFO

        shard_placements = [
            (i, p)
            for i, p in enumerate(param.placements)
            if p.is_shard() and param.device_mesh.size(i) > 1
        ]

        is_batch_sharded = False
        if not group["flatten"]:
            matrix_dims = {param.ndim - 1, param.ndim - 2}
            is_batch_sharded = any(
                p.dim not in matrix_dims for _, p in shard_placements
            )
            shard_placements = [
                (i, p) for i, p in shard_placements if p.dim in matrix_dims
            ]

        if len(shard_placements) == 0:
            return ShardInfo(
                is_batch_sharded=is_batch_sharded,
                is_matrix_sharded=False,
                sharded_tensor_dim=None,
                process_group=None,
                device_rank=0,
                world_size=1,
            )
        if len(shard_placements) > 1:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support parameters with "
                f"multiple matrix-sharded dimensions. Got placements "
                f"{param.placements} on mesh {param.device_mesh}."
            )

        sharded_mesh_dim, shard_placement = shard_placements[0]
        mesh = param.device_mesh
        return ShardInfo(
            is_batch_sharded=is_batch_sharded,
            is_matrix_sharded=True,
            sharded_tensor_dim=shard_placement.dim,
            process_group=mesh.get_group(sharded_mesh_dim),
            device_rank=mesh.get_local_rank(sharded_mesh_dim),
            world_size=mesh.size(sharded_mesh_dim),
        )

    def _create_ortho_tasks(
        self, param_groups: List[dict]
    ) -> Generator["AsyncTask", None, None]:
        """Subclasses implement this to create orthogonalization tasks."""
        raise NotImplementedError

    def _create_lion_tasks(
        self, param_groups: List[dict]
    ) -> Generator["AsyncTask", None, None]:
        for group in param_groups:
            assert group["algorithm"] == "lion"
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue
            gradients = [p.grad for p in params]
            states = [self._get_or_initialize_state(p, "lion") for p in params]
            momentums = [s["momentum"] for s in states]

            yield AsyncTask(
                lion_update_foreach_async(
                    X=to_local(params),
                    G=to_local(gradients),
                    M=to_local(momentums),
                    lr=torch.tensor(group["lr"]),
                    beta1=torch.tensor(group["beta1"]),
                    beta2=torch.tensor(group["beta2"]),
                    weight_decay=torch.tensor(group["weight_decay"]),
                    cautious_wd=group.get("cautious_wd", False),
                )
            )

    def _create_adamw_tasks(
        self, param_groups: List[dict]
    ) -> Generator["AsyncTask", None, None]:
        for group in param_groups:
            assert group["algorithm"] == "adamw"
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue
            gradients = [p.grad for p in params]
            states = [self._get_or_initialize_state(p, "adamw") for p in params]
            momentums = [s["momentum"] for s in states]
            variances = [s["variance"] for s in states]

            yield AsyncTask(
                adamw_update_foreach_async(
                    X=to_local(params),
                    G=to_local(gradients),
                    M=to_local(momentums),
                    V=to_local(variances),
                    lr=torch.tensor(group["lr"]),
                    beta1=torch.tensor(group["beta1"]),
                    beta2=torch.tensor(group["beta2"]),
                    weight_decay=torch.tensor(group["weight_decay"]),
                    step=torch.tensor(group["step"]),
                    epsilon=torch.tensor(group["epsilon"]),
                    cautious_wd=group.get("cautious_wd", False),
                )
            )


def megabatch_orthogonalize_async(
    U: List[Tensor],
    comm_dim: Optional[int],
    device_rank: int,
    world_size: int,
    process_group: Optional[ProcessGroup],
    newton_schulz_func: Callable,
    flatten: bool,
    epsilon: Tensor,
    global_comm_dim_size: Optional[int],
) -> Generator[None, None, List[Tensor]]:
    """
    Shared megabatch communication + Newton-Schulz orthogonalization.

    This is a generator that yields at async communication points and uses
    ``return value`` to pass the result back to the caller. In Python, ``return``
    inside a generator raises ``StopIteration(value)``, and the caller recovers
    the value via ``result = yield from megabatch_orthogonalize_async(...)``.
    The ``yield from`` transparently forwards intermediate yields to AsyncRuntime.

    Args:
        U: List of tensors to orthogonalize (all same shape).
        comm_dim: Dimension for cat/split in all-to-all (negative index).
            None for non-sharded parameters.
        device_rank: This device's rank.
        world_size: Total number of devices.
        process_group: Distributed process group. None for single-GPU.
        newton_schulz_func: Newton-Schulz orthogonalization function.
        flatten: Whether to flatten 3D+ tensors to 2D.
        epsilon: Small value for numerical stability.
        global_comm_dim_size: Required (non-None) when ``comm_dim is not
            None``; pass ``None`` otherwise. The unsharded (global) size
            along ``comm_dim``, taken from the DTensor's global shape
            (``param.shape[comm_dim]``). Used to compute
            ``padded_local_size = ceil(global / world_size)`` so the
            alltoall sees uniform per-pair sizes across ranks.
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

        # Pad each rank's local shard along comm_dim to a rank-consistent
        # ``padded_local_size = ceil(global / world_size)`` so dist.all_to_all
        # sees uniform per-pair sizes. FSDP2 contiguous chunking otherwise
        # leaves some ranks with empty (numel=0) shards when the sharded
        # global dim is smaller than world_size or doesn't divide evenly to
        # fill all ranks (e.g. shape (18, D) over world_size=8: ranks 6 and 7
        # hold (0, D) shards). Without padding the alltoall has mismatched
        # per-pair sizes and hangs. The padding is transport-only: after the
        # first all-to-all, it is removed before Newton-Schulz and restored
        # before the return all-to-all. This avoids feeding synthetic rows to
        # finite-precision orthogonalization kernels.
        #
        # NOTE: this assumes FSDP2-style contiguous chunking, where every rank
        # holds at most ceil(global / world_size) elements along comm_dim. If
        # FSDP2 ever switches to a non-contiguous strategy (e.g. block-cyclic),
        # this derivation would be wrong; the size check below catches that.
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
            # F.pad's pad-spec is built from the LAST dim backwards. comm_dim
            # is negative; pad only the END of comm_dim.
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

        # comm_dim is negative, so it correctly indexes the stacked tensor.
        # Remove synthetic transport rows before orthogonalization. In exact
        # arithmetic zero padding is inert, but Polar Express runs in BF16 and
        # a different GEMM shape can change reduction order/kernel selection.
        # Cropping here keeps the optimizer computation topology-stable while
        # retaining uniform all-to-all payloads for uneven and empty shards.
        full_matrices = torch.cat(output_chunks, dim=comm_dim)
        transport_size = padded_local_size * world_size
        if transport_size != global_comm_dim_size:
            full_matrices = full_matrices.narrow(
                comm_dim, 0, global_comm_dim_size
            ).contiguous()
        full_matrices = muon_update_newton_schulz(
            full_matrices,
            newton_schulz_func=newton_schulz_func,
            flatten=flatten,
            epsilon=epsilon,
        )

        if transport_size != global_comm_dim_size:
            pad_spec = [0, 0] * (-comm_dim - 1) + [
                0,
                transport_size - global_comm_dim_size,
            ]
            full_matrices = torch.nn.functional.pad(full_matrices, pad_spec)

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

        # Narrow each per-rank result back to the rank's original local size.
        # On padding-only ranks original_local_size == 0 and the slice is empty.
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
        my_matrices = muon_update_newton_schulz(
            my_matrices,
            newton_schulz_func=newton_schulz_func,
            flatten=flatten,
            epsilon=epsilon,
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
            muon_update_newton_schulz(
                U[0],
                newton_schulz_func=newton_schulz_func,
                flatten=flatten,
                epsilon=epsilon,
            )
        ]

    else:
        # N > 1, no process_group (single GPU or batch-sharded 3D)
        stacked = torch.stack(U)
        stacked = muon_update_newton_schulz(
            stacked,
            newton_schulz_func=newton_schulz_func,
            flatten=flatten,
            epsilon=epsilon,
        )
        return [stacked[i] for i in range(N)]


def muon_update_newton_schulz(
    X: Tensor,
    newton_schulz_func: Callable,
    flatten: bool,
    epsilon: Tensor,
) -> Tensor:
    """
    Flatten the input tensor if needed and call the Newton-Schulz function.
    """
    original_shape = X.shape
    if flatten and X.ndim >= 3:
        X = X.flatten(start_dim=1)
    elif X.ndim >= 4:
        X = X.flatten(end_dim=-3)

    return newton_schulz_func(X, epsilon=epsilon).reshape(original_shape)


def adjust_lr_rms_norm(lr, param_shape, flatten):
    """Adjust learning rate for constant element-wise RMS norm."""
    if flatten:
        fan_out = param_shape[0]
        fan_in = math.prod(param_shape[1:])
    else:
        fan_out, fan_in = param_shape[-2:]
    adjusted_ratio = 0.2 * math.sqrt(max(fan_out, fan_in))
    return lr * adjusted_ratio


def adjust_lr_spectral_norm(lr, param_shape, flatten):
    """Adjust from spectral norm 1 to RMS operator norm 1."""
    if flatten:
        fan_out = param_shape[0]
        fan_in = math.prod(param_shape[1:])
    else:
        fan_out, fan_in = param_shape[-2:]
    return lr * math.sqrt(fan_out / fan_in)


def adjust_lr_keller_muon(lr, param_shape, flatten):
    """Keller Muon LR scaling used by the Trinity large MuonMS path."""
    if flatten:
        fan_out = param_shape[0]
        fan_in = math.prod(param_shape[1:])
    else:
        fan_out, fan_in = param_shape[-2:]
    return lr * max(1, fan_out / fan_in) ** 0.5
