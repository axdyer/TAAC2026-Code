"""PCVRHyFormer training entry point (self-contained baseline).

Usage:
    python train.py [--num_epochs 10] [--batch_size 256] ...

Environment variables (take precedence over CLI flags):
    TRAIN_DATA_PATH  Training data directory (*.parquet + schema.json)
    TRAIN_CKPT_PATH  Checkpoint output directory
    TRAIN_LOG_PATH   Log directory
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import torch

from utils import set_seed, EarlyStopping, create_logger
from dataset import FeatureSchema, get_pcvr_data, NUM_TIME_BUCKETS, NUM_FINE_TIME_BUCKETS
from model import PCVRHyFormer
from trainer import PCVRHyFormerRankingTrainer


def build_feature_specs(
    schema: FeatureSchema,
    per_position_vocab_sizes: List[int],
) -> List[Tuple[int, int, int]]:
    """Build feature_specs of the form ``[(vocab_size, offset, length), ...]``
    ordered by the positions recorded in ``schema.entries``.
    """
    specs: List[Tuple[int, int, int]] = []
    for fid, offset, length in schema.entries:
        vs = max(per_position_vocab_sizes[offset:offset + length])
        specs.append((vs, offset, length))
    return specs


def parse_user_feat_pair(value: str) -> List[int]:
    """Parse --user_feat_pair.

    Accepts either ``62,63,64`` or a JSON-style list ``[62,63,64]``. The
    empty string / ``[]`` disables the feature.
    """
    s = value.strip()
    if not s or s == '[]':
        return []
    if s.startswith('['):
        parsed = json.loads(s)
        if not isinstance(parsed, list):
            raise ValueError("--user_feat_pair JSON value must be a list")
        fids = parsed
    else:
        fids = [p.strip() for p in s.split(',') if p.strip()]
    result: List[int] = []
    seen = set()
    for fid in fids:
        if not isinstance(fid, int):
            if not isinstance(fid, str) or not fid.isdigit():
                raise ValueError(f"--user_feat_pair contains non-integer fid: {fid!r}")
            fid = int(fid)
        if fid in seen:
            raise ValueError(f"--user_feat_pair contains duplicate fid: {fid}")
        seen.add(fid)
        result.append(fid)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PCVRHyFormer Training")

    # Paths (environment variables take precedence).
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Training data directory (env: TRAIN_DATA_PATH)')
    parser.add_argument('--schema_path', type=str, default=None,
                        help='Schema JSON path (defaults to <data_dir>/schema.json)')
    parser.add_argument('--ckpt_dir', type=str, default=None,
                        help='Checkpoint output directory (env: TRAIN_CKPT_PATH)')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='Log directory (env: TRAIN_LOG_PATH)')

    # Training hyperparameters.
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for both training and validation')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for dense parameters (AdamW)')
    parser.add_argument('--num_epochs', type=int, default=999,
                        help='Maximum number of training epochs '
                             '(typically terminated earlier by early stopping)')
    parser.add_argument('--patience', type=int, default=5,
                        help='Early-stopping patience '
                             '(number of validations without improvement)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Training device, e.g. cuda or cpu')

    # Speed controls.
    parser.add_argument('--amp_dtype', type=str, default='none',
                        choices=['none', 'bf16'],
                        help='Automatic mixed precision mode. bf16 requires CUDA '
                             'bf16 support and fails immediately when unsupported.')
    parser.add_argument('--compile_model', action='store_true', default=False,
                        help='Compile the training forward path with torch.compile. '
                             'Compilation failures are not swallowed.')
    parser.add_argument('--no_compile_model', dest='compile_model',
                        action='store_false',
                        help='Explicitly disable torch.compile, useful for overriding run.sh')
    parser.add_argument('--compile_mode', type=str, default='reduce-overhead',
                        choices=['default', 'reduce-overhead', 'max-autotune'],
                        help='torch.compile mode used when --compile_model is set')
    parser.add_argument('--allow_tf32', action='store_true', default=False,
                        help='Enable CUDA TF32 matmul/cuDNN kernels for additional speed. '
                             'This is explicit because it changes fp32 numerics.')
    parser.add_argument('--show_progress_bar', action='store_true', default=False,
                        help='Show tqdm progress bars for train/eval loops. '
                             'Disabled by default to keep platform logs compact.')

    # Data pipeline.
    parser.add_argument('--num_workers', type=int, default=16,
                        help='Number of DataLoader workers')
    parser.add_argument('--prefetch_factor', type=int, default=2,
                        help='DataLoader prefetch_factor when num_workers > 0')
    parser.add_argument('--buffer_batches', type=int, default=20,
                        help='Shuffle buffer size, in units of batches. '
                             'Lower values reduce memory usage.')
    parser.add_argument('--split_mode', type=str, default='rowgroup',
                        choices=['timestamp', 'rowgroup'],
                        help='Train/valid split mode. timestamp = row-level time '
                             'split by timestamp, using the latest valid_ratio rows '
                             'as validation; rowgroup = baseline tail Row Group split.')
    parser.add_argument('--train_ratio', type=float, default=1.0,
                        help='Fraction of training Row Groups to use in rowgroup split '
                             '(timestamp split requires this to stay at 1.0)')
    parser.add_argument('--valid_ratio', type=float, default=0.1,
                        help='Fraction of data used for validation. timestamp split uses '
                             'the latest rows by timestamp; rowgroup split uses tail Row Groups.')
    parser.add_argument('--interval', action='store_true', default=False,
                        help='After the train/valid split, further filter both sets '
                             'to only keep rows whose timestamp falls in --time_range. '
                             'Requires --time_range START END.')
    parser.add_argument('--time_range', type=int, nargs=2, default=None,
                        metavar=('START', 'END'),
                        help='Time range filter (Unix timestamps), used when --interval '
                             'is set. Rows with START <= timestamp < END are kept.')
    parser.add_argument('--eval_every_n_steps', type=int, default=0,
                        help='Run validation every N steps '
                             '(0 = only at the end of each epoch)')
    parser.add_argument('--seq_max_lens', type=str,
                        default='seq_a:256,seq_b:256,seq_c:512,seq_d:512',
                        help='Per-domain sequence truncation, format: seq_d:256,seq_c:128')

    # Model hyperparameters.
    parser.add_argument('--d_model', type=int, default=64,
                        help='Backbone hidden dimension (output size of each block)')
    parser.add_argument('--emb_dim', type=int, default=64,
                        help='Per-Embedding-table dimension (before projection)')
    parser.add_argument('--num_queries', type=int, default=1,
                        help='Number of Query tokens generated independently per sequence domain')
    parser.add_argument('--num_hyformer_blocks', type=int, default=2,
                        help='Number of stacked MultiSeqHyFormerBlock layers')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='Number of attention heads (must satisfy d_model %% num_heads == 0)')
    parser.add_argument('--seq_encoder_type', type=str, default='transformer',
                        choices=['swiglu', 'transformer', 'longer'],
                        help='Sequence encoder variant: '
                             'swiglu = SwiGLU without attention, '
                             'transformer = standard self-attention, '
                             'longer = Top-K compressed encoder '
                             '(only this variant consumes --seq_top_k / --seq_causal)')
    parser.add_argument('--hidden_mult', type=int, default=4,
                        help='FFN inner-dim multiplier relative to d_model')
    parser.add_argument('--dropout_rate', type=float, default=0.01,
                        help='Dropout rate for the backbone '
                             '(seq id-embedding dropout is twice this value)')
    parser.add_argument('--seq_top_k', type=int, default=50,
                        help='Number of most-recent tokens kept by LongerEncoder '
                             '(only effective when --seq_encoder_type=longer)')
    parser.add_argument('--seq_causal', action='store_true', default=False,
                        help='Whether the LongerEncoder self-attention uses a causal mask '
                             '(only effective when --seq_encoder_type=longer)')
    parser.add_argument('--action_num', type=int, default=1,
                        help='Classifier output dimension '
                             '(1 = single binary-classification logit; >1 = multi-label)')
    parser.add_argument('--use_time_buckets', action='store_true', default=True,
                        help='Enable the time-bucket embedding (default on). '
                             'The actual bucket count is uniquely determined by '
                             'dataset.BUCKET_BOUNDARIES; this flag is a pure on/off switch.')
    parser.add_argument('--no_time_buckets', dest='use_time_buckets', action='store_false',
                        help='Disable the time-bucket embedding')

    # Time feature ablation switches (all default off).
    parser.add_argument('--use_hour_encoding', action='store_true', default=False,
                        help='Add Beijing-time hour-of-day sin/cos to user_dense')
    parser.add_argument('--use_user_time_stats', action='store_true', default=False,
                        help='Add per-sequence recency/time_span/frequency to user_dense')
    parser.add_argument('--use_fine_time_buckets', action='store_true', default=False,
                        help='Use finer-grained time-delta bucket boundaries '
                             '(88 boundaries vs baseline 64)')
    parser.add_argument('--use_time_decay_attn', action='store_true', default=False,
                        help='Apply learnable multiplicative time-decay gating '
                             'to sequence token embeddings')
    parser.add_argument('--rank_mixer_mode', type=str, default='full',
                        choices=['full', 'ffn_only', 'none'],
                        help='RankMixerBlock mode: '
                             'full = token mixing + per-token FFN (requires d_model divisible by T), '
                             'ffn_only = per-token FFN only, '
                             'none = identity passthrough')
    parser.add_argument('--use_rope', action='store_true', default=False,
                        help='Enable RoPE positional encoding in sequence attention')
    parser.add_argument('--rope_base', type=float, default=10000.0,
                        help='RoPE base frequency (default 10000)')

    # Loss function.
    parser.add_argument('--loss_type', type=str, default='bce', choices=['bce', 'focal'],
                        help='Loss type: bce = BCEWithLogits, focal = Focal Loss')
    parser.add_argument('--focal_alpha', type=float, default=0.1,
                        help='Focal Loss positive-class weight alpha '
                             '(effective only when --loss_type=focal)')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focal Loss focusing parameter gamma '
                             '(effective only when --loss_type=focal)')
    parser.add_argument('--use_time_sample_weight', action='store_true', default=False,
                        help='Enable fixed per-sample loss weights based on timestamp '
                             'recency. Disabled by default.')
    parser.add_argument('--time_weight_ref_timestamp', type=int, default=0,
                        help='Reference/cutoff Unix timestamp for recency weighting. '
                             'Required when --use_time_sample_weight is set. Samples '
                             'closer to this timestamp receive larger loss weights.')
    parser.add_argument('--time_weight_half_life_hours', type=float, default=3.0,
                        help='Half-life in hours for exponential time-sample weighting. '
                             'Every half-life farther from --time_weight_ref_timestamp '
                             'halves the distance from max weight to min weight.')
    parser.add_argument('--time_weight_min', type=float, default=0.05,
                        help='Minimum fixed loss weight for old samples when '
                             '--use_time_sample_weight is set.')
    parser.add_argument('--time_weight_max', type=float, default=3.0,
                        help='Maximum fixed loss weight for samples at '
                             '--time_weight_ref_timestamp when --use_time_sample_weight '
                             'is set.')

    # Sparse optimizer.
    parser.add_argument('--sparse_lr', type=float, default=0.05,
                        help='Learning rate for sparse parameters (Adagrad over Embeddings)')
    parser.add_argument('--sparse_weight_decay', type=float, default=0.0,
                        help='Weight decay for sparse parameters (Adagrad over Embeddings)')
    parser.add_argument('--reinit_sparse_after_epoch', type=int, default=1,
                        help='Starting from the N-th epoch, at the end of every epoch '
                             're-initialize Embeddings with vocab_size > '
                             '--reinit_cardinality_threshold and rebuild the Adagrad '
                             'optimizer state (cold-restart trick for high-cardinality '
                             'features to reduce overfitting)')
    parser.add_argument('--reinit_cardinality_threshold', type=int, default=0,
                        help='Cardinality threshold used by the re-init strategy: '
                             'Embeddings whose vocab_size exceeds this value are reset '
                             'at each epoch end (0 = never reset any Embedding)')

    # Embedding construction control.
    parser.add_argument('--emb_skip_threshold', type=int, default=0,
                        help='At model construction time, features whose vocab_size '
                             'exceeds this value get no Embedding and are represented '
                             'by a zero vector at forward time (0 = no skipping; '
                             'all features get an Embedding). Useful for saving GPU '
                             'memory on ultra-high-cardinality features.')
    parser.add_argument('--seq_id_threshold', type=int, default=10000,
                        help='Within the sequence tokenizer, features with vocab_size '
                             'exceeding this value are treated as id features and receive '
                             'extra dropout(rate*2) during training to reduce overfitting. '
                             'Features at or below this threshold are treated as side-info '
                             'and receive no extra dropout.')

    _default_ns_groups = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'ns_groups.json')
    parser.add_argument('--ns_groups_json', type=str, default=_default_ns_groups,
                        help='Path to the NS-groups JSON file. If it does not exist, '
                             'each feature is placed in its own singleton group.')

    # NS tokenizer variant.
    parser.add_argument('--ns_tokenizer_type', type=str, default='rankmixer',
                        choices=['group', 'rankmixer'],
                        help='NS tokenizer variant: '
                             'group = project each group to one token, '
                             'rankmixer = concatenate all embeddings then split into '
                             'equal-size chunks (token count is tunable)')
    parser.add_argument('--user_ns_tokens', type=int, default=0,
                        help='Number of user NS tokens in rankmixer mode '
                             '(0 = automatically use the number of user groups)')
    parser.add_argument('--item_ns_tokens', type=int, default=0,
                        help='Number of item NS tokens in rankmixer mode '
                             '(0 = automatically use the number of item groups)')
    parser.add_argument('--user_feat_pair', type=str, default='',
                        help='Comma-separated or JSON list of user fids whose '
                             'aligned user_int/user_dense values should be paired '
                             'and fused into the user_dense token, e.g. '
                             "'62,63,64,65,66' or '[65,66]'. Empty disables it.")

    args = parser.parse_args()
    args.user_feat_pair = parse_user_feat_pair(args.user_feat_pair)

    # Environment variables take precedence.
    args.data_dir = os.environ.get('TRAIN_DATA_PATH', args.data_dir)
    args.ckpt_dir = os.environ.get('TRAIN_CKPT_PATH', args.ckpt_dir)
    args.log_dir = os.environ.get('TRAIN_LOG_PATH', args.log_dir)
    args.tf_events_dir = os.environ.get('TRAIN_TF_EVENTS_PATH')

    return args


def main() -> None:
    args = parse_args()
    if args.num_workers > 0 and args.prefetch_factor < 1:
        raise ValueError("--prefetch_factor must be >= 1 when --num_workers > 0")
    if args.interval:
        if args.time_range is None or len(args.time_range) != 2:
            raise ValueError(
                "--interval requires --time_range START END "
                "(two Unix timestamps)")
        if args.time_range[0] >= args.time_range[1]:
            raise ValueError(
                f"--time_range start must be < end, got "
                f"{args.time_range[0]} >= {args.time_range[1]}")
    if args.use_time_sample_weight:
        if args.time_weight_ref_timestamp <= 0:
            raise ValueError(
                "--time_weight_ref_timestamp must be explicitly set to a positive "
                "Unix timestamp when --use_time_sample_weight is enabled"
            )
        if args.time_weight_half_life_hours <= 0:
            raise ValueError("--time_weight_half_life_hours must be > 0")
        if args.time_weight_min <= 0:
            raise ValueError("--time_weight_min must be > 0")
        if args.time_weight_max < args.time_weight_min:
            raise ValueError("--time_weight_max must be >= --time_weight_min")

    # Create output directories.
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.tf_events_dir).mkdir(parents=True, exist_ok=True)

    # Initialize logger and RNG.
    set_seed(args.seed)
    create_logger(os.path.join(args.log_dir, 'train.log'))
    logging.info(f"Args: {vars(args)}")

    if args.allow_tf32:
        if not args.device.startswith('cuda') or not torch.cuda.is_available():
            raise RuntimeError("--allow_tf32 requires a CUDA device")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')
        logging.info("CUDA TF32 enabled for matmul and cuDNN")

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(args.tf_events_dir)

    # ---- Data loading ----
    if args.schema_path:
        schema_path = args.schema_path
    else:
        schema_path = os.path.join(args.data_dir, 'schema.json')

    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"schema file not found at {schema_path}")

    # Parse per-domain sequence-length overrides.
    seq_max_lens = {}
    if args.seq_max_lens:
        for pair in args.seq_max_lens.split(','):
            k, v = pair.split(':')
            seq_max_lens[k.strip()] = int(v.strip())
        logging.info(f"Seq max_lens override: {seq_max_lens}")

    logging.info("Using Parquet data format (IterableDataset)")
    train_loader, valid_loader, pcvr_dataset = get_pcvr_data(
        data_dir=args.data_dir,
        schema_path=schema_path,
        batch_size=args.batch_size,
        valid_ratio=args.valid_ratio,
        train_ratio=args.train_ratio,
        split_mode=args.split_mode,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        buffer_batches=args.buffer_batches,
        seed=args.seed,
        seq_max_lens=seq_max_lens,
        use_hour_encoding=args.use_hour_encoding,
        use_user_time_stats=args.use_user_time_stats,
        use_fine_time_buckets=args.use_fine_time_buckets,
        use_time_decay_attn=args.use_time_decay_attn,
        time_range=args.time_range,
        interval=args.interval,
    )

    # ---- NS groups ----
    if args.ns_groups_json and os.path.exists(args.ns_groups_json):
        logging.info(f"Loading NS groups from {args.ns_groups_json}")
        with open(args.ns_groups_json, 'r') as f:
            ns_groups_cfg = json.load(f)
        user_fid_to_idx = {fid: i for i, (fid, _, _) in enumerate(pcvr_dataset.user_int_schema.entries)}
        item_fid_to_idx = {fid: i for i, (fid, _, _) in enumerate(pcvr_dataset.item_int_schema.entries)}
        user_ns_groups = [[user_fid_to_idx[f] for f in fids] for fids in ns_groups_cfg['user_ns_groups'].values()]
        item_ns_groups = [[item_fid_to_idx[f] for f in fids] for fids in ns_groups_cfg['item_ns_groups'].values()]
        logging.info(f"User NS groups ({len(user_ns_groups)}): {list(ns_groups_cfg['user_ns_groups'].keys())}")
        logging.info(f"Item NS groups ({len(item_ns_groups)}): {list(ns_groups_cfg['item_ns_groups'].keys())}")
    else:
        logging.info("No NS groups JSON found, using default: each feature as one group")
        user_ns_groups = [[i] for i in range(len(pcvr_dataset.user_int_schema.entries))]
        item_ns_groups = [[i] for i in range(len(pcvr_dataset.item_int_schema.entries))]

    # ---- Build model ----
    user_int_feature_specs = build_feature_specs(
        pcvr_dataset.user_int_schema, pcvr_dataset.user_int_vocab_sizes)
    item_int_feature_specs = build_feature_specs(
        pcvr_dataset.item_int_schema, pcvr_dataset.item_int_vocab_sizes)

    model_args = {
        "user_int_feature_specs": user_int_feature_specs,
        "item_int_feature_specs": item_int_feature_specs,
        "user_int_feature_ids": pcvr_dataset.user_int_schema.feature_ids,
        "user_dense_feature_specs": pcvr_dataset.user_dense_schema.entries,
        "user_dense_dim": pcvr_dataset.user_dense_schema.total_dim,
        "item_dense_dim": pcvr_dataset.item_dense_schema.total_dim,
        "seq_vocab_sizes": pcvr_dataset.seq_domain_vocab_sizes,
        "user_ns_groups": user_ns_groups,
        "item_ns_groups": item_ns_groups,
        "d_model": args.d_model,
        "emb_dim": args.emb_dim,
        "num_queries": args.num_queries,
        "num_hyformer_blocks": args.num_hyformer_blocks,
        "num_heads": args.num_heads,
        "seq_encoder_type": args.seq_encoder_type,
        "hidden_mult": args.hidden_mult,
        "dropout_rate": args.dropout_rate,
        "seq_top_k": args.seq_top_k,
        "seq_causal": args.seq_causal,
        "action_num": args.action_num,
        "num_time_buckets": (
            NUM_FINE_TIME_BUCKETS if (args.use_time_buckets and args.use_fine_time_buckets)
            else NUM_TIME_BUCKETS if args.use_time_buckets
            else 0
        ),
        "rank_mixer_mode": args.rank_mixer_mode,
        "use_rope": args.use_rope,
        "rope_base": args.rope_base,
        "emb_skip_threshold": args.emb_skip_threshold,
        "seq_id_threshold": args.seq_id_threshold,
        "ns_tokenizer_type": args.ns_tokenizer_type,
        "user_ns_tokens": args.user_ns_tokens,
        "item_ns_tokens": args.item_ns_tokens,
        "user_feat_pair": args.user_feat_pair,
        "use_time_decay_attn": args.use_time_decay_attn,
    }

    model = PCVRHyFormer(**model_args).to(args.device)

    # Log model sizing info.
    num_sequences = len(pcvr_dataset.seq_domains)
    num_ns = model.num_ns
    T = args.num_queries * num_sequences + num_ns
    logging.info(f"PCVRHyFormer model created: num_ns={num_ns}, T={T}, d_model={args.d_model}, rank_mixer_mode={args.rank_mixer_mode}")
    logging.info(f"User NS groups: {user_ns_groups}")
    logging.info(f"Item NS groups: {item_ns_groups}")
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Total parameters: {total_params:,}")

    # ---- Training ----
    early_stopping = EarlyStopping(
        checkpoint_path=os.path.join(args.ckpt_dir, "placeholder", "model.pt"),
        patience=args.patience,
        label='model',
    )

    ckpt_params = {
        "layer": args.num_hyformer_blocks,
        "head": args.num_heads,
        "hidden": args.d_model,
    }

    trainer = PCVRHyFormerRankingTrainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        lr=args.lr,
        num_epochs=args.num_epochs,
        device=args.device,
        save_dir=args.ckpt_dir,
        early_stopping=early_stopping,
        loss_type=args.loss_type,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        use_time_sample_weight=args.use_time_sample_weight,
        time_weight_ref_timestamp=args.time_weight_ref_timestamp,
        time_weight_half_life_hours=args.time_weight_half_life_hours,
        time_weight_min=args.time_weight_min,
        time_weight_max=args.time_weight_max,
        sparse_lr=args.sparse_lr,
        sparse_weight_decay=args.sparse_weight_decay,
        reinit_sparse_after_epoch=args.reinit_sparse_after_epoch,
        reinit_cardinality_threshold=args.reinit_cardinality_threshold,
        ckpt_params=ckpt_params,
        writer=writer,
        schema_path=schema_path,
        ns_groups_path=args.ns_groups_json if args.ns_groups_json and os.path.exists(args.ns_groups_json) else None,
        eval_every_n_steps=args.eval_every_n_steps,
        train_config=vars(args),
        amp_dtype=args.amp_dtype,
        compile_model=args.compile_model,
        compile_mode=args.compile_mode,
        show_progress_bar=args.show_progress_bar,
    )

    trainer.train()
    writer.close()

    logging.info("Training complete!")


if __name__ == "__main__":
    main()
