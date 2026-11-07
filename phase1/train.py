"""Compare manual gradient averaging and DDP with ordinary training."""

import argparse
from copy import deepcopy
from datetime import timedelta

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP


def main(mode):
    torch.set_num_threads(1)
    # torchrun supplies the rank, world size, and connection settings.
    dist.init_process_group("gloo", timeout=timedelta(seconds=30))
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        assert world_size >= 2, "Use at least two worker processes."

        # Same seed means every rank creates the same data and starting weights.
        torch.manual_seed(0)
        x = torch.randn(32, 2)
        y = 2 * x[:, :1] - 3 * x[:, 1:] + 1
        assert len(x) % world_size == 0, "Ranks need equal-sized batches."
        local_x = x.chunk(world_size)[rank]
        local_y = y.chunk(world_size)[rank]

        local_model = nn.Linear(2, 1)
        reference = deepcopy(local_model)
        # CPU DDP wraps the model and averages gradients during backward().
        model = DDP(local_model) if mode == "ddp" else local_model
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)

        with torch.no_grad():
            initial_loss = F.mse_loss(reference(x), y).item()

        for step in range(5):
            optimizer.zero_grad()
            reference_optimizer.zero_grad()

            # Each worker trains on a different slice of the global batch.
            loss = F.mse_loss(model(local_x), local_y)
            loss.backward()
            if mode == "manual":
                for parameter in local_model.parameters():
                    assert parameter.grad is not None
                    dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
                    # Equal batch sizes make this the global mean gradient.
                    parameter.grad /= world_size

            # The reference sees the entire batch, without distributed calls.
            reference_loss = F.mse_loss(reference(x), y)
            reference_loss.backward()
            for parameter, expected in zip(local_model.parameters(), reference.parameters()):
                assert parameter.grad is not None and expected.grad is not None
                torch.testing.assert_close(parameter.grad, expected.grad)

            optimizer.step()
            reference_optimizer.step()
            for parameter, expected in zip(local_model.parameters(), reference.parameters()):
                torch.testing.assert_close(parameter, expected)

            if rank == 0:
                print(f"step {step}: global loss = {reference_loss.item():.4f}")

        with torch.no_grad():
            final_loss = F.mse_loss(reference(x), y).item()
        assert final_loss < initial_loss, "Training should reduce the loss."
        dist.barrier()
        if rank == 0:
            print(f"{mode}: gradients and weights match the reference on all {world_size} ranks.")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["manual", "ddp"])
    main(parser.parse_args().mode)
