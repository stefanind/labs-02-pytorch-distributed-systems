"""Each process runs the same collective operations, in the same order."""

from datetime import timedelta

import torch
import torch.distributed as dist


def main():
    # Use one thread for tensor operations in each worker.
    torch.set_num_threads(1)
    # torchrun supplies the rank, world size, and connection settings.
    dist.init_process_group("gloo", timeout=timedelta(seconds=30))
    try:
        rank = dist.get_rank()  # This process's ID: 0, 1, ...
        world_size = dist.get_world_size()  # Number of processes.
        assert world_size >= 2, "Use at least two worker processes."

        # Collectives write their results into the tensors we pass in.
        # Broadcast: rank 0 shares its value with every rank.
        value = torch.tensor([42 if rank == 0 else -1])
        dist.broadcast(value, src=0)
        assert value.item() == 42

        # All-reduce: add everyone's values; everyone receives the sum.
        value = torch.tensor([rank + 1])
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        expected_sum = world_size * (world_size + 1) // 2
        assert value.item() == expected_sum

        # All-gather: everyone receives each rank's value, in rank order.
        value = torch.tensor([rank])
        gathered = [torch.empty_like(value) for _ in range(world_size)]
        dist.all_gather(gathered, value)
        ranks = [item.item() for item in gathered]
        assert ranks == list(range(world_size))

        # Barrier: nobody continues until every rank reaches this line.
        # It also ensures every rank passed the assertions before we print.
        dist.barrier()
        if rank == 0:
            print(f"broadcast: 42 on all {world_size} ranks")
            print(f"all_reduce: {expected_sum} on all ranks")
            print(f"all_gather: {ranks} on all ranks")
            print("All collective checks passed.")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
