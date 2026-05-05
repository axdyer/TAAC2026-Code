#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

# ---- Active config: RankMixer NS tokenizer (no ns_groups.json required) ----
"${PYTHON_BIN}" -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --split_mode timestamp \
    --ns_groups_json "" \
    --emb_skip_threshold 1000000 \
    --amp_dtype bf16 \
    --compile_model \
    --compile_mode default \
    --num_workers 8 \
    --prefetch_factor 4 \
    "$@"

# ---- Time feature ablation switches (append to the command above via "$@") ----
#   --use_hour_encoding        # Beijing-time hour-of-day sin/cos → user_dense
#   --use_user_time_stats      # Per-sequence recency/time_span/frequency → user_dense
#   --use_fine_time_buckets    # 88 finer-grained time-delta buckets (vs baseline 64)
#   --use_time_decay_attn      # Learnable multiplicative time-decay gating on seq tokens
#
# Example: combine all four:
#   bash run.sh --use_hour_encoding --use_user_time_stats \
#               --use_fine_time_buckets --use_time_decay_attn

# ---- Time-range filter mode (split_mode=rowgroupinterval) ----
# Filters ALL row groups to a [start, end) timestamp window.
#   --split_mode rowgroupinterval --time_range START END
#
# Example: train only on March 21-23 data:
#   bash run.sh --split_mode rowgroupinterval --time_range 1774080000 1774252800

# ---- Alternative config: GroupNSTokenizer driven by ns_groups.json ----
# Uses feature grouping from ns_groups.json (7 user groups + 4 item groups).
# With d_model=64 and num_ns=12 (7 user_int + 1 user_dense + 4 item_int),
# only num_queries=1 satisfies d_model % T == 0 (T = num_queries*4 + num_ns).
# To switch, comment out the block above and uncomment the block below.
#
# python3 -u "${SCRIPT_DIR}/train.py" \
#     --ns_tokenizer_type group \
#     --ns_groups_json "${SCRIPT_DIR}/ns_groups.json" \
#     --num_queries 1 \
#     --emb_skip_threshold 1000000 \
#     --num_workers 8 \
#     "$@"
