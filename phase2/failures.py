"""Break a small CPU job and use each rank's last log line to diagnose it."""

import argparse
from datetime import timedelta
import os
import signal
import time

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.elastic.multiprocessing.errors import record
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP


def log(message):
    # Monotonic timestamps are comparable across processes on this Linux host.
    # Flush so the last message survives an exception or a killed worker.
    print(
        f"[{time.monotonic():.3f}s rank={os.environ['RANK']} pid={os.getpid()}] {message}",
        flush=True,
    )


def uneven_batches(rank, world_size):
    # Same model and data as Phase 1; rank 1 takes one extra training step.
    torch.manual_seed(0)
    x = torch.randn(32, 2)
    y = 2 * x[:, :1] - 3 * x[:, 1:] + 1
    assert len(x) % world_size == 0, "Ranks need equal-sized batches."
    local_x = x.chunk(world_size)[rank]
    local_y = y.chunk(world_size)[rank]
    model = DDP(nn.Linear(2, 1))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    batches = 3 if rank == 1 else 2  # Change this to 2 to repair the experiment.
    log(f"training on {batches} batches")

    for step in range(batches):
        optimizer.zero_grad()
        log(f"step {step}: before forward")
        loss = F.mse_loss(model(local_x), local_y)
        log(f"step {step}: before backward (DDP averages gradients here)")
        loss.backward()
        log(f"step {step}: after backward")
        for parameter in model.parameters():
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all(), "Gradients must be finite."
        optimizer.step()

    # A barrier cannot replace a missing DDP step. Other ranks are still training.
    log(f"finished {batches} batches; before final barrier")
    dist.barrier()
    log("after final barrier")

    # These checks become reachable when every rank takes the same batch count.
    for parameter in model.parameters():
        weights = parameter.detach()
        gathered = [torch.empty_like(weights) for _ in range(world_size)]
        dist.all_gather(gathered, weights)
        for other in gathered:
            torch.testing.assert_close(weights, other)
    log("all ranks finished training with matching weights")


@record  # Include uncaught Python tracebacks in torchrun's failure summary.
def main(mode, timeout, delay):
    assert timeout > 0, "The process-group timeout must be positive."
    assert delay >= 0, "A worker's delay cannot be negative."
    torch.set_num_threads(1)
    log(f"initializing Gloo: mode={mode}, timeout={timeout}s")
    dist.init_process_group("gloo", timeout=timedelta(seconds=timeout))
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        assert world_size >= 2, "Use at least two worker processes."
        # All workers finish setup before any rank injects a failure.
        dist.barrier()
        log("setup complete")
        value = torch.tensor([rank + 1])

        if mode == "slow":
            if rank == 1:
                log(f"sleeping for {delay}s")
                time.sleep(delay)
            log("before all_reduce")
            dist.all_reduce(value)
            log("after all_reduce")
            assert value.item() == world_size * (world_size + 1) // 2
            log(f"sum={value.item()}; slow worker check passed")

        elif mode == "mismatch":
            # Same two collectives, reversed on rank 1. DETAIL reveals the mismatch.
            if rank == 1:
                log("before broadcast (wrong order)")
                dist.broadcast(value, src=0)
                log("after broadcast; before all_reduce")
                dist.all_reduce(value)
            else:
                log("before all_reduce")
                dist.all_reduce(value)
                log("after all_reduce; before broadcast")
                dist.broadcast(value, src=0)

        elif mode == "uneven":
            uneven_batches(rank, world_size)

        else:  # exception or kill
            if rank == 1:
                if mode == "exception":
                    log("raising an injected Python exception")
                    raise RuntimeError("Injected failure on rank 1")
                log("sending SIGKILL to my own PID")
                os.kill(os.getpid(), signal.SIGKILL)
            log("before all_reduce (rank 1 will never join)")
            dist.all_reduce(value)
            log("after all_reduce")
    finally:
        # Do not add a barrier here: a failed rank may never reach it.
        # SIGKILL skips Python cleanup entirely on the killed rank.
        log("destroying process group")
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["slow", "mismatch", "uneven", "exception", "kill"])
    parser.add_argument("--timeout", type=int, default=5, help="Gloo timeout in seconds")
    parser.add_argument("--delay", type=float, default=2, help="Rank 1's sleep in slow mode")
    args = parser.parse_args()
    main(args.mode, args.timeout, args.delay)
