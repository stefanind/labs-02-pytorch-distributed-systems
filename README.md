# Distributed training learning lab

Small experiments for learning PyTorch distributed training on Linux. Phases 1
and 2 run on a CPU. Read the scripts, run them, and change one thing at a time.

## Setup

Use Python 3.10 to 3.13.

From the repository root, create a fresh environment and install CPU PyTorch:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu
```

The commands use this environment directly, so no activation is needed.

The examples use **Gloo**, PyTorch's CPU communication backend. No GPU or dataset
download is needed.

## Phase 1

Run these from the repository root, in order:

```bash
.venv/bin/torchrun --standalone --nproc-per-node=2 phase1/collectives.py
.venv/bin/torchrun --standalone --nproc-per-node=2 phase1/train.py manual
.venv/bin/torchrun --standalone --nproc-per-node=2 phase1/train.py ddp
```

`torchrun` is PyTorch's standard launcher. `--standalone` sets up a job on one
machine. `--nproc-per-node=2` starts two worker processes. Each process runs the
same lesson, and `torchrun` supplies its rank and connection settings:

- **Rank:** a process's ID, starting at 0.
- **World size:** the total number of processes.
- **Process group:** the processes that communicate with one another.
- **Collective:** a communication operation that every process in the group joins.

Every rank must call collectives in the same order. One rank calling an operation
alone can leave it waiting for the others. The scripts use a 30-second process
group timeout and release the group when they finish.

### 1. Collective communication

Read [phase1/collectives.py](phase1/collectives.py). Its assertions check:

| Operation | What happens | Expected with 4 ranks |
| --- | --- | --- |
| Broadcast | Rank 0 shares one value. | Everyone gets `42`. |
| All-reduce | Add the values `1, 2, 3, 4`. | Everyone gets `10`. |
| All-gather | Collect the rank IDs in rank order. | Everyone gets `[0, 1, 2, 3]`. |
| Barrier | Wait until all ranks arrive. | The success message follows all checks. |

Only rank 0 prints, but every rank runs the assertions.

### 2. Manual distributed training

Read [phase1/train.py](phase1/train.py), starting with `manual` mode.
The model learns `y = 2*x0 - 3*x1 + 1` with one linear layer and five SGD steps.

The global batch always has 32 examples. With 4 ranks, each rank trains on 8
different examples. Each rank starts with identical model weights, computes its
local mean loss, and runs backward to get local gradients.

`all_reduce` adds those gradients. Dividing by the number of ranks gives their
average. Because each rank has the same number of examples, this equals the
gradient of the mean loss over all 32 examples. Each rank then applies the same
optimizer update to its own model.

A separate reference model trains on all 32 examples. **Every step checks both
gradients and updated weights against that reference, on every rank.**
`torch.testing.assert_close` allows small rounding differences between floating
point calculations. A final assertion checks that the loss decreased.

For this tiny lab, every rank creates the full dataset and its own reference
model so the checks are easy to follow. The distributed model uses only its slice.

### 3. DDP

Run the same file with `ddp`. DDP means **Distributed Data Parallel**: each rank
keeps a full copy of the model and trains on different examples.

The wrapper averages gradients during `backward()`, so we skip the manual
all-reduce and division. DDP does not split the data for us; the `chunk` calls
still do that. The same reference assertions must pass.

### Try it yourself

1. Repeat all three commands with `--nproc-per-node=4`, then `8`. Predict the
   collective outputs and local batch size before running them.
2. Remove the division by `world_size` in manual mode. The gradient check should
   fail. Explain why, then restore it.
3. Give every rank the first data chunk. The reference check should fail because
   the workers repeat the same examples. Restore the rank-based slicing.
4. Change the broadcast value and update its assertion. Check that every rank
   still receives the value from rank 0.

Phase 1 is complete when all three commands pass with 2, 4, and 8 processes,
and you can explain why manual averaging and DDP match the reference.

All nine runs passed on Ubuntu with Python 3.12.3 and CPU PyTorch 2.11.0.

## Phase 2: Failure experiments

Read [phase2/failures.py](phase2/failures.py). Each experiment changes what rank 1
does. Logs include a monotonic timestamp, rank, and PID, and flush immediately so
you can see where each worker stopped.

Run these from the repository root. [phase2/run.sh](phase2/run.sh) starts two
workers, saves the logs, and prints the run directory and exit code:

```bash
./phase2/run.sh slow            # Expect exit 0.
./phase2/run.sh slow --delay 8  # Expect nonzero, as with the cases below.
./phase2/run.sh mismatch
./phase2/run.sh uneven
./phase2/run.sh exception
./phase2/run.sh kill
```

The launcher enables `TORCH_DISTRIBUTED_DEBUG=DETAIL` only for `mismatch` to check
collective ordering; other experiments use `OFF`.
In the tested CPU PyTorch 2.11.0 environment, `DETAIL` caused a segmentation fault
during DDP construction. Keeping it off for `uneven` lets that experiment reach
the intended extra-backward failure.

The process group has a **5-second timeout**; change it with `--timeout`.
The default slow-worker delay is 2 seconds. The outer Linux `timeout` limits the
whole launcher to 30 seconds, then forces termination after another 5 seconds if
needed. `--max-restarts=0` keeps failed workers from restarting. An exit status of
124 or 137 means the outer guard fired; investigate why the inner timeout or
launcher did not stop the job first.

| Experiment | What to observe |
| --- | --- |
| `slow` | Other ranks wait in all-reduce until rank 1 arrives. The sum assertion passes. |
| `slow --delay 8` | Rank 1 sleeps longer than the group timeout. Waiting ranks report a communication timeout. |
| `mismatch` | Rank 1 calls collectives in a different order. `DETAIL` reports inconsistent collective fingerprints. |
| `uneven` | Rank 1 attempts 3 DDP batches; other ranks finish 2 and enter the final barrier. The extra backward and barrier cannot match. |
| `exception` | Rank 1 raises a Python exception. Find its traceback; peers may report communication errors before launcher shutdown. |
| `kill` | Rank 1 sends `SIGKILL` to its own PID. It cannot run Python cleanup. Peers or the launcher detect its disappearance. |

DDP communicates inside `backward()`. In `uneven`, a rank that finishes its data
still cannot proceed to a different collective while another rank trains. The
final barrier makes that disagreement visible; it does not fix it. Assertions
check that completed steps have finite gradients. Giving every rank the same
batch count also lets the final assertions check that all ranks' weights match.

Each invocation creates a unique `.tmp/phase2/<mode>.<suffix>/` directory:

- `command.txt` records the debug setting and exact launch command.
- `console.log` combines the launcher and workers' terminal output. Start here.
- `exit-code.txt` records the job's actual exit code. The script returns it too.

Individual ranks' logs remain under `<run_id>/attempt_0/<rank>/` inside that
directory. The `@record` decorator on `main` also saves uncaught Python exceptions
in `error.json` and includes their tracebacks in the launcher summary. A rank
killed by `SIGKILL` cannot write that file.

For an exception run, use the directory printed by the launcher:

```bash
run_dir=.tmp/phase2/exception.ABC123  # Replace with your run directory.
cat "$run_dir/console.log"
cat "$run_dir/exit-code.txt"
cat "$run_dir"/*/attempt_0/1/stderr.log
```

Start with the last logged action on each rank, then find the originating traceback.
Exact error text and reporting order can vary: a peer may notice a closed
connection before the launcher reports the original failure.

### Try it yourself

1. Run all cases with 2 workers, then repeat with 4 and 8 using
   `NPROC_PER_NODE=4 ./phase2/run.sh slow`, and so on. The default `slow` run should
   succeed; every injected failure should terminate before the outer guard.
2. Change `--delay` across the group timeout. Explain which ranks wait and why
   a slower worker can be healthy until the timeout is exceeded.
3. Give every rank 2 batches in `uneven`. The gradient and weight assertions and
   final barrier should pass. Restore the uneven count afterward.
4. Explain why an exception has a Python traceback while `SIGKILL` has none, and
   why a final barrier cannot repair mismatched collective counts.

Phase 2 is complete when you can locate each failure in the per-rank logs,
explain its cause, and show that every job terminates within the configured bounds.

All six experiments and the equal-batch repair were verified with 2, 4, and 8
workers. Injected failures exited nonzero before the outer guard.

## References

- [Writing distributed applications with PyTorch](https://docs.pytorch.org/tutorials/intermediate/dist_tuto.html)
- [Getting started with DDP](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [Distributed debug checks](https://docs.pytorch.org/docs/2.11/distributed.html#torch-distributed-debug)
- [torchrun](https://docs.pytorch.org/docs/2.11/elastic/run.html)
