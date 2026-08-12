"""Aurora regressions for uneven and empty FSDP2 row shards."""

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from dion.aurora import (
    aurora_process_matrices,
    megabatch_aurora_orthogonalize_async,
)
from dion.opt_utils import AsyncRuntime, AsyncTask

MASTER_PORT = "29402"


def _identity_polar(x, epsilon):
    del epsilon
    return x


def _local_chunk(full: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    chunks = torch.chunk(full, world_size, dim=0)
    if rank < len(chunks):
        return chunks[rank].contiguous()
    return full.new_empty((0, *full.shape[1:]))


def _aurora_worker(
    rank: int,
    world_size: int,
    global_rows: int,
    columns: int,
    n_params: int,
    port: int,
) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        full_inputs = [
            torch.arange(1, global_rows * columns + 1, dtype=torch.float32).view(
                global_rows, columns
            )
            + param_index * 100
            for param_index in range(n_params)
        ]
        local_inputs = [_local_chunk(full, rank, world_size) for full in full_inputs]
        state = {}

        def _task_gen():
            result = yield from megabatch_aurora_orthogonalize_async(
                local_inputs,
                comm_dim=-2,
                device_rank=rank,
                world_size=world_size,
                process_group=dist.group.WORLD,
                newton_schulz_func=_identity_polar,
                flatten=False,
                epsilon=torch.tensor(1e-8),
                pp_iterations=2,
                pp_beta=0.5,
                global_comm_dim_size=global_rows,
            )
            state["result"] = result

        runtime = AsyncRuntime(
            (task for task in [AsyncTask(_task_gen())]), max_concurrent_tasks=1
        )
        runtime.run()

        assert len(state["result"]) == n_params
        for full_input, actual in zip(full_inputs, state["result"]):
            expected_full = aurora_process_matrices(
                full_input,
                newton_schulz_func=_identity_polar,
                flatten=False,
                epsilon=torch.tensor(1e-8),
                pp_iterations=2,
                pp_beta=0.5,
            )
            expected = _local_chunk(expected_full, rank, world_size)
            torch.testing.assert_close(actual, expected)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("global_rows", [1, 3])
def test_aurora_megabatch_handles_empty_and_uneven_shards(global_rows):
    world_size = 2
    mp.spawn(
        _aurora_worker,
        args=(world_size, global_rows, 4, 3, int(MASTER_PORT)),
        nprocs=world_size,
        join=True,
    )


def test_aurora_megabatch_requires_global_comm_dim_size():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", MASTER_PORT)
    already_init = dist.is_initialized()
    if not already_init:
        dist.init_process_group("gloo", rank=0, world_size=1)
    try:
        gen = megabatch_aurora_orthogonalize_async(
            [torch.zeros(2, 4)],
            comm_dim=-2,
            device_rank=0,
            world_size=1,
            process_group=dist.group.WORLD,
            newton_schulz_func=_identity_polar,
            flatten=False,
            epsilon=torch.tensor(1e-8),
            pp_iterations=2,
            pp_beta=0.5,
            global_comm_dim_size=None,
        )
        with pytest.raises(ValueError, match="global_comm_dim_size"):
            next(gen)
    finally:
        if not already_init and dist.is_initialized():
            dist.destroy_process_group()
