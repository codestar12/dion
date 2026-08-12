"""End-to-end DCP regression for elastic, plain-FSDP Muon resumes.

The test deliberately changes the data-parallel/FSDP world size instead of
constructing a fixed HSDP mesh:

    1-rank FSDP2 save -> 2-rank FSDP2 load -> real Muon optimizer step

It uses :func:`torch.distributed.checkpoint.state_dict.get_state_dict` and
``set_state_dict`` around a local-filesystem DCP checkpoint, exactly like a
training checkpoint.  The two tiny matrix parameters force both edge cases at
world size two: ``(3, 8)`` produces uneven shards and ``(1, 8)`` leaves rank 1
with an empty shard.  The loaded model, Muon momentum, optimizer step counter,
and post-resume update are compared with a deterministic one-rank oracle.

Run the definitive two-GPU gate with either command::

    pytest -q tests/test_dcp_elastic_muon.py
    python3 tests/test_dcp_elastic_muon.py

The parent process imposes a hard timeout on each ``torchrun`` phase.  Set
``DION_ELASTIC_TEST_TIMEOUT_SECONDS`` to override the default 180 seconds.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
from types import MethodType

_PARAMETER_SHAPES = ((3, 8), (1, 8))
_LR = 0.03125
_MOMENTUM = 0.8
_SEED = 20260811
_SAVE_STEP = 7
_DEFAULT_TIMEOUT_SECONDS = 180


def _torchrun_command(
    nproc: int, phase: str, run_dir: Path, *, pytest_mode: bool
) -> list[str]:
    sibling_torchrun = Path(sys.executable).with_name("torchrun")
    torchrun = (
        str(sibling_torchrun)
        if sibling_torchrun.is_file()
        else shutil.which("torchrun")
    )
    if torchrun is None:
        if pytest_mode:
            import pytest

            pytest.skip("torchrun is required for the elastic DCP integration test")
        raise RuntimeError("torchrun is required for the elastic DCP integration test")
    return [
        torchrun,
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={nproc}",
        "--max-restarts=0",
        str(Path(__file__).resolve()),
        "--worker-phase",
        phase,
        "--run-dir",
        str(run_dir),
    ]


def _terminate_process_group(process: subprocess.Popen) -> None:
    """Terminate torchrun and all of its worker descendants after a timeout."""
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)


def _run_phase(
    nproc: int, phase: str, run_dir: Path, *, pytest_mode: bool = False
) -> None:
    timeout = int(
        os.environ.get(
            "DION_ELASTIC_TEST_TIMEOUT_SECONDS", str(_DEFAULT_TIMEOUT_SECONDS)
        )
    )
    log_path = run_dir / f"{phase}-world-size-{nproc}.log"
    process = subprocess.Popen(
        _torchrun_command(nproc, phase, run_dir, pytest_mode=pytest_mode),
        cwd=Path(__file__).resolve().parents[1],
        env={
            **os.environ,
            "PYTHONUNBUFFERED": "1",
            "NCCL_ASYNC_ERROR_HANDLING": "1",
            "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        bufsize=1,
    )
    output: list[str] = []

    def tee_output() -> None:
        assert process.stdout is not None
        with log_path.open("w", encoding="utf-8") as log:
            for line in process.stdout:
                output.append(line)
                log.write(line)
                log.flush()
                print(line, end="", flush=True)

    reader = threading.Thread(target=tee_output, name=f"tee-{phase}", daemon=True)
    reader.start()
    try:
        returncode = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        _terminate_process_group(process)
        reader.join(timeout=5)
        raise AssertionError(
            f"{phase} phase (world size {nproc}) exceeded {timeout}s; "
            f"partial output is in {log_path}"
        ) from None
    reader.join(timeout=5)
    assert not reader.is_alive(), f"output reader did not finish for {phase} phase"
    assert returncode == 0, (
        f"{phase} phase (world size {nproc}) failed with exit code "
        f"{returncode}; full output is in {log_path}:\n{''.join(output)}"
    )


def test_dcp_reshards_muon_from_one_to_two_ranks(tmp_path: Path) -> None:
    """DCP must reshard model and Muon state before an uneven/empty step."""
    import pytest

    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices for the NCCL/FSDP2 regression")

    _run_phase(1, "save", tmp_path, pytest_mode=True)
    _run_phase(2, "load", tmp_path, pytest_mode=True)


def _worker_imports():
    """Import heavyweight distributed modules only inside torchrun workers."""
    import torch
    import torch.distributed as dist
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard
    from torch.distributed.tensor import DTensor

    from dion import Muon

    return (
        torch,
        dist,
        dcp,
        get_state_dict,
        set_state_dict,
        init_device_mesh,
        fully_shard,
        DTensor,
        Muon,
    )


def _new_model(torch):
    class TinyUnevenModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            generator = torch.Generator(device="cpu").manual_seed(_SEED)
            self.uneven = torch.nn.Parameter(
                torch.randn(_PARAMETER_SHAPES[0], generator=generator)
            )
            self.empty_on_rank_one = torch.nn.Parameter(
                torch.randn(_PARAMETER_SHAPES[1], generator=generator)
            )

    return TinyUnevenModel()


def _new_optimizer(model, Muon):
    return Muon(
        model.parameters(),
        lr=_LR,
        mu=_MOMENTUM,
        weight_decay=0.0,
        nesterov=False,
        adjust_lr=None,
        use_triton=False,
        use_polar_express=True,
    )


def _deterministic_full_gradient(torch, name: str, shape) -> object:
    offset = 101 if name == "uneven" else 211
    values = torch.arange(
        int(torch.tensor(shape).prod().item()), dtype=torch.float32
    ).reshape(shape)
    return ((values + offset) % 17 - 8) / 16


def _assign_gradients(torch, model) -> None:
    for name, param in model.named_parameters():
        full = _deterministic_full_gradient(torch, name, tuple(param.shape)).cuda()
        if hasattr(param, "to_local"):
            # ``distribute_tensor`` shards the global deterministic gradient
            # according to the parameter's current (possibly empty) placement.
            from torch.distributed.tensor import distribute_tensor

            param.grad = distribute_tensor(full, param.device_mesh, param.placements)
        else:
            param.grad = full


def _full_tensor(value, DTensor):
    return value.full_tensor() if isinstance(value, DTensor) else value.detach().clone()


def _full_snapshot(model, optimizer, DTensor) -> dict[str, object]:
    snapshot = {}
    for name, param in model.named_parameters():
        snapshot[f"model.{name}"] = _full_tensor(param, DTensor).cpu()
        snapshot[f"momentum.{name}"] = _full_tensor(
            optimizer.state[param]["momentum"], DTensor
        ).cpu()
    snapshot["step"] = int(optimizer.param_groups[0]["step"])
    return snapshot


def _assert_snapshot(
    torch, actual, expected, label: str, *, rtol: float = 0, atol: float = 0
) -> None:
    assert actual.keys() == expected.keys(), (actual.keys(), expected.keys())
    for key in actual:
        if key == "step":
            assert (
                actual[key] == expected[key]
            ), f"{label}: {key}: {actual[key]} != {expected[key]}"
        else:
            torch.testing.assert_close(
                actual[key],
                expected[key],
                rtol=rtol,
                atol=atol,
                msg=lambda msg: f"{label}: {key}: {msg}",
            )


def _save_oracle(torch, run_dir: Path, before, after) -> None:
    torch.save({"before": before, "after": after}, run_dir / "oracle.pt")


def _load_oracle(torch, run_dir: Path):
    return torch.load(run_dir / "oracle.pt", map_location="cpu", weights_only=True)


def _save_phase(run_dir: Path) -> None:
    (
        torch,
        dist,
        dcp,
        get_state_dict,
        _set_state_dict,
        init_device_mesh,
        fully_shard,
        DTensor,
        Muon,
    ) = _worker_imports()
    assert dist.get_world_size() == 1
    mesh = init_device_mesh("cuda", (1,), mesh_dim_names=("dp",))
    model = _new_model(torch).cuda()
    fully_shard(model, mesh=mesh)
    optimizer = _new_optimizer(model, Muon)

    # Seed non-zero momentum and an exact optimizer step counter.  Saving a
    # populated state is what distinguishes this regression from a model-only
    # elastic load (and catches missing optimizer-key/state materialization).
    _assign_gradients(torch, model)
    optimizer.step()
    optimizer.param_groups[0]["step"] = _SAVE_STEP
    before = _full_snapshot(model, optimizer, DTensor)

    model_state, optimizer_state = get_state_dict(model, optimizer)
    checkpoint = {"model": model_state, "optimizer": optimizer_state}
    dcp.save(checkpoint, checkpoint_id=run_dir / "checkpoint")

    # The one-rank continuation is the numerical oracle for the first post-load
    # step.  It uses identical full gradients to the two-rank resume.
    _assign_gradients(torch, model)
    optimizer.step()
    after = _full_snapshot(model, optimizer, DTensor)
    if dist.get_rank() == 0:
        _save_oracle(torch, run_dir, before, after)


def _load_phase(run_dir: Path) -> None:
    (
        torch,
        dist,
        dcp,
        get_state_dict,
        set_state_dict,
        init_device_mesh,
        fully_shard,
        DTensor,
        Muon,
    ) = _worker_imports()
    assert dist.get_world_size() == 2
    mesh = init_device_mesh("cuda", (2,), mesh_dim_names=("dp",))
    model = _new_model(torch).cuda()
    fully_shard(model, mesh=mesh)
    optimizer = _new_optimizer(model, Muon)

    # This is intentionally a fresh optimizer.  get_state_dict supplies DCP's
    # sharded load template; DCP redistributes the saved one-rank tensors into
    # the current two-rank layout; set_state_dict installs them in Muon.
    # This fresh optimizer must already have a complete template.  PyTorch's
    # generic fallback initializes empty optimizer state by executing a dummy
    # zero-LR optimizer step, which can hide a missing eager-state fix.  Make
    # that fallback a hard failure and prove template construction made zero
    # step calls.
    original_step = optimizer.step
    template_step_calls = 0

    def forbidden_template_step(_optimizer, *args, **kwargs):
        nonlocal template_step_calls
        template_step_calls += 1
        raise AssertionError("get_state_dict must not call Muon.step")

    optimizer.step = MethodType(forbidden_template_step, optimizer)
    try:
        model_state, optimizer_state = get_state_dict(model, optimizer)
    finally:
        optimizer.step = original_step
    assert template_step_calls == 0
    checkpoint = {"model": model_state, "optimizer": optimizer_state}
    dcp.load(checkpoint, checkpoint_id=run_dir / "checkpoint")
    incompatible = set_state_dict(
        model,
        optimizer,
        model_state_dict=checkpoint["model"],
        optim_state_dict=checkpoint["optimizer"],
    )
    assert not incompatible.missing_keys
    assert not incompatible.unexpected_keys

    oracle = _load_oracle(torch, run_dir)
    loaded = _full_snapshot(model, optimizer, DTensor)
    _assert_snapshot(torch, loaded, oracle["before"], "after elastic load")

    local_shapes = [tuple(param.to_local().shape) for param in model.parameters()]
    gathered_shapes = [None] * dist.get_world_size()
    dist.all_gather_object(gathered_shapes, local_shapes)
    if dist.get_rank() == 0:
        # (3, 8) -> (2, 8) + (1, 8); (1, 8) -> (1, 8) + (0, 8).
        assert gathered_shapes == [[(2, 8), (1, 8)], [(1, 8), (0, 8)]]

    _assign_gradients(torch, model)
    optimizer.step()  # real NCCL all-to-all, including the empty local shard
    resumed = _full_snapshot(model, optimizer, DTensor)
    _assert_snapshot(
        torch,
        resumed,
        oracle["after"],
        "after resumed Muon step",
        rtol=2e-3,
        atol=2e-4,
    )


def _worker_main(phase: str, run_dir: Path) -> None:
    torch, dist, *_ = _worker_imports()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    try:
        if phase == "save":
            _save_phase(run_dir)
        elif phase == "load":
            _load_phase(run_dir)
        else:  # pragma: no cover - guarded by argparse choices
            raise ValueError(phase)
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-phase", choices=("save", "load"))
    parser.add_argument("--run-dir", type=Path)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    if args.worker_phase is None:
        # Keep direct execution convenient on a compute node without relying
        # on pytest's test discovery or temporary-directory fixture.
        import torch

        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            raise SystemExit("requires two CUDA devices")
        if args.run_dir is not None:
            path = args.run_dir.resolve()
            path.mkdir(parents=True, exist_ok=True)
            _run_phase(1, "save", path)
            _run_phase(2, "load", path)
        else:
            with tempfile.TemporaryDirectory(prefix="dion-dcp-elastic-") as tmp:
                path = Path(tmp)
                _run_phase(1, "save", path)
                _run_phase(2, "load", path)
    else:
        if args.run_dir is None:
            raise SystemExit("--run-dir is required with --worker-phase")
        _worker_main(args.worker_phase, args.run_dir)
