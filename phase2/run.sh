#!/usr/bin/env bash
# Run one experiment and save enough context to diagnose it afterward.
set -euo pipefail
cd "$(dirname "$0")/.."

case "${1:-}" in
    slow|mismatch|uneven|exception|kill) ;;
    *)
        echo "Usage: $0 {slow|mismatch|uneven|exception|kill} [script options]" >&2
        exit 2
        ;;
esac

mkdir -p .tmp/phase2
run_dir=$(mktemp -d ".tmp/phase2/$1.XXXXXX")
export TORCH_DISTRIBUTED_DEBUG=OFF
if [[ $1 == mismatch ]]; then
    export TORCH_DISTRIBUTED_DEBUG=DETAIL
fi

launch=(
    timeout --kill-after=5s 30s .venv/bin/torchrun
    --standalone --nproc-per-node="${NPROC_PER_NODE:-2}"
    --max-restarts=0 --monitor-interval=1
    --log-dir="$run_dir" --tee=3 phase2/failures.py "$@"
)
printf 'TORCH_DISTRIBUTED_DEBUG=%q ' "$TORCH_DISTRIBUTED_DEBUG" > "$run_dir/command.txt"
printf '%q ' "${launch[@]}" >> "$run_dir/command.txt"
printf '\n' >> "$run_dir/command.txt"

# Expected worker failures must still reach the status file. tee has its own status;
# PIPESTATUS[0] preserves the launcher's exit code instead of reporting tee's success.
status=0
"${launch[@]}" 2>&1 | tee "$run_dir/console.log" || status=${PIPESTATUS[0]}
printf '%s\n' "$status" > "$run_dir/exit-code.txt"
printf 'Job exit code: %s; logs: %s\n' "$status" "$run_dir"
exit "$status"
