"""PCVR Parquet dataset module (performance-tuned).

Reads raw multi-column Parquet directly and obtains feature metadata from
``schema.json``.

Optimizations:
- Pre-allocated numpy buffers to eliminate ``np.zeros`` + ``np.stack`` overhead.
- Fused padding loop over sequence domains that writes directly into a 3D buffer.
- Pre-computed column-index lookup to avoid per-row string lookups.
- ``file_system`` tensor-sharing strategy to work around ``/dev/shm`` exhaustion
  when using many DataLoader workers.
"""

import os
import logging
import random
import json
import gc

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
import torch.multiprocessing
from torch.utils.data import IterableDataset, DataLoader
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

# numpy.typing is available since numpy >= 1.20; on older numpy fall back to a
# no-op shim so that forward-referenced annotations like ``npt.NDArray[np.int64]``
# keep working as plain strings without raising at import time.
try:
    import numpy.typing as npt  # noqa: F401
except ImportError:  # pragma: no cover
    class _NptFallback:  # type: ignore[no-redef]
        NDArray = Any

    npt = _NptFallback()  # type: ignore[assignment]


TimestampRange = Tuple[int, int]
SampleWeightRange = Tuple[int, int, float]


def _coerce_timestamp_range_pairs(
    value: Optional[Any],
    label: str,
) -> List[TimestampRange]:
    """Parse either ``[s, e, ...]`` or ``[[s, e], ...]`` into pairs."""
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        value = value.tolist()

    values = list(value)
    if not values:
        return []

    nested = [isinstance(v, (list, tuple, np.ndarray)) for v in values]
    pairs: List[TimestampRange] = []
    if all(nested):
        for pair in values:
            pair_values = pair.tolist() if isinstance(pair, np.ndarray) else list(pair)
            if len(pair_values) != 2:
                raise ValueError(
                    f"{label} must contain timestamp pairs, got {pair!r}")
            pairs.append((int(pair_values[0]), int(pair_values[1])))
    elif any(nested):
        raise ValueError(
            f"{label} must be either flat START END pairs or nested pairs, "
            f"got {value!r}")
    else:
        if len(values) % 2 != 0:
            raise ValueError(
                f"{label} must contain an even number of Unix timestamps, "
                f"got {len(values)} values")
        for i in range(0, len(values), 2):
            pairs.append((int(values[i]), int(values[i + 1])))
    return pairs


def _merge_closed_timestamp_ranges(
    ranges: List[TimestampRange],
) -> List[TimestampRange]:
    if not ranges:
        return []

    merged: List[TimestampRange] = []
    for start, end in sorted(ranges):
        if not merged:
            merged.append((start, end))
            continue
        prev_start, prev_end = merged[-1]
        if start <= prev_end + 1:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _merge_exclusive_timestamp_ranges(
    ranges: List[TimestampRange],
) -> List[TimestampRange]:
    if not ranges:
        return []

    merged: List[TimestampRange] = []
    for start, end in sorted(ranges):
        if not merged:
            merged.append((start, end))
            continue
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def normalize_closed_timestamp_ranges(
    time_range: Optional[Any] = None,
    time_ranges: Optional[Any] = None,
) -> Optional[List[TimestampRange]]:
    """Normalize user-facing closed timestamp ranges.

    ``time_range`` is the legacy single interval ``START END`` argument.
    ``time_ranges`` accepts multiple closed intervals, either as a flat list
    ``START1 END1 START2 END2`` or as nested pairs. The two arguments are
    intentionally mutually exclusive to avoid silently widening a training
    window because of an accidental leftover flag.
    """
    if time_range is not None and time_ranges is not None:
        raise ValueError("--time_range and --time_ranges are mutually exclusive")

    label = "--time_ranges" if time_ranges is not None else "--time_range"
    ranges = _coerce_timestamp_range_pairs(
        time_ranges if time_ranges is not None else time_range,
        label,
    )
    if not ranges:
        return None

    for start, end in ranges:
        if start > end:
            raise ValueError(
                f"{label} ranges must satisfy START <= END, got "
                f"{start} > {end}")
    return _merge_closed_timestamp_ranges(ranges)


def normalize_closed_timestamp_windows(
    time_ranges: Optional[Any],
    label: str = "--multi_valid_time_ranges",
) -> Optional[List[TimestampRange]]:
    """Normalize closed timestamp windows while preserving user order.

    Unlike ``normalize_closed_timestamp_ranges``, this helper intentionally
    does not merge overlapping or adjacent ranges. It is used for independent
    validation windows, where ``[1, 5]`` and ``[3, 6]`` should remain two
    separate validation sets.
    """
    ranges = _coerce_timestamp_range_pairs(time_ranges, label)
    if not ranges:
        return None

    for start, end in ranges:
        if start > end:
            raise ValueError(
                f"{label} windows must satisfy START <= END, got "
                f"{start} > {end}")
    return ranges


def _closed_to_exclusive_timestamp_ranges(
    ranges: Optional[List[TimestampRange]],
) -> Optional[List[TimestampRange]]:
    if not ranges:
        return None
    return [(start, end + 1) for start, end in ranges]


def _normalize_exclusive_timestamp_ranges(
    timestamp_ranges: Optional[Any],
) -> Optional[List[TimestampRange]]:
    ranges = _coerce_timestamp_range_pairs(timestamp_ranges, "timestamp_ranges")
    if not ranges:
        return None
    for start, end in ranges:
        if start >= end:
            raise ValueError(
                f"timestamp_ranges must be half-open ranges with START < END, "
                f"got {start} >= {end}")
    return _merge_exclusive_timestamp_ranges(ranges)


def _format_closed_timestamp_ranges(
    ranges: Optional[List[TimestampRange]],
) -> str:
    if not ranges:
        return "[]"
    return "[" + ", ".join(f"[{start}, {end}]" for start, end in ranges) + "]"


def normalize_sample_weight_ranges(
    value: Optional[Any],
) -> List[SampleWeightRange]:
    """Parse user-facing sample weight ranges.

    Expected string format:
        ``START,END,WEIGHT;START,END,WEIGHT``

    Ranges are closed intervals and must not overlap. Adjacent ranges such as
    ``[1, 2]`` and ``[3, 4]`` are allowed.
    """
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        entries = [part.strip() for part in text.split(';') if part.strip()]
    else:
        entries = list(value)
        if not entries:
            return []

    ranges: List[SampleWeightRange] = []
    for idx, entry in enumerate(entries, start=1):
        if isinstance(entry, str):
            parts = [p.strip() for p in entry.split(',')]
        else:
            parts = list(entry)
        if len(parts) != 3:
            raise ValueError(
                "--sample_weight_ranges entries must be START,END,WEIGHT; "
                f"entry #{idx} is {entry!r}")
        start = int(parts[0])
        end = int(parts[1])
        weight = float(parts[2])
        if start > end:
            raise ValueError(
                "--sample_weight_ranges uses closed intervals and requires "
                f"START <= END, got {start} > {end}")
        if not np.isfinite(weight) or weight <= 0.0:
            raise ValueError(
                "--sample_weight_ranges weights must be finite positive "
                f"numbers, got {weight!r} for [{start}, {end}]")
        ranges.append((start, end, weight))

    sorted_ranges = sorted(ranges, key=lambda x: (x[0], x[1]))
    for prev, curr in zip(sorted_ranges, sorted_ranges[1:]):
        prev_start, prev_end, _ = prev
        curr_start, curr_end, _ = curr
        if curr_start <= prev_end:
            raise ValueError(
                "--sample_weight_ranges must not overlap; got closed "
                f"ranges [{prev_start}, {prev_end}] and "
                f"[{curr_start}, {curr_end}]")
    return sorted_ranges


def _apply_timestamp_filter_np(
    arr: "npt.NDArray[np.int64]",
    timestamp_min: Optional[int] = None,
    timestamp_max: Optional[int] = None,
    timestamp_ranges: Optional[List[TimestampRange]] = None,
) -> "npt.NDArray[np.int64]":
    if timestamp_min is not None:
        arr = arr[arr >= timestamp_min]
    if timestamp_max is not None:
        arr = arr[arr < timestamp_max]
    if timestamp_ranges is not None:
        mask = np.zeros(arr.shape, dtype=bool)
        for start, end in timestamp_ranges:
            mask |= (arr >= start) & (arr < end)
        arr = arr[mask]
    return arr


# ─────────────────────────── Feature Schema ──────────────────────────────────


class FeatureSchema:
    """Records ``(feature_id, offset, length)`` for each feature so downstream
    code can locate the segment of the flattened tensor that belongs to a
    specific feature id.

    For int features:
      - int_value: length = 1
      - int_array: length = array length
      - int_array_and_float_array: int part length
    For dense features:
      - float_value: length = 1
      - float_array: length = array length
      - int_array_and_float_array: float part length
    """

    def __init__(self) -> None:
        # Ordered list of (feature_id, offset, length).
        self.entries: List[Tuple[int, int, int]] = []
        self.total_dim: int = 0
        # Quick lookup from fid to its (offset, length).
        self._fid_to_entry: Dict[int, Tuple[int, int]] = {}

    def add(self, feature_id: int, length: int) -> None:
        """Append a feature to the schema."""
        offset = self.total_dim
        self.entries.append((feature_id, offset, length))
        self._fid_to_entry[feature_id] = (offset, length)
        self.total_dim += length

    def get_offset_length(self, feature_id: int) -> Tuple[int, int]:
        """Get ``(offset, length)`` for a feature_id."""
        return self._fid_to_entry[feature_id]

    @property
    def feature_ids(self) -> List[int]:
        """Return all feature_ids in their insertion order."""
        return [fid for fid, _, _ in self.entries]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict (for JSON dumping)."""
        return {
            'entries': self.entries,
            'total_dim': self.total_dim,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'FeatureSchema':
        """Reconstruct a :class:`FeatureSchema` from its dict form."""
        schema = cls()
        for fid, offset, length in d['entries']:
            schema.entries.append((fid, offset, length))
            schema._fid_to_entry[fid] = (offset, length)
        schema.total_dim = d['total_dim']
        return schema

    def __repr__(self) -> str:
        lines = [f"FeatureSchema(total_dim={self.total_dim}, features=["]
        for fid, offset, length in self.entries:
            lines.append(f"  fid={fid}: offset={offset}, length={length}")
        lines.append("])")
        return "\n".join(lines)

# Use filesystem-based tensor sharing (instead of /dev/shm) to avoid running
# out of shared memory when many DataLoader workers are active.
torch.multiprocessing.set_sharing_strategy('file_system')

# Time-delta bucket boundaries (64 edges -> 65 buckets: 0=padding, 1..64).
BUCKET_BOUNDARIES = np.array([
    5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60,
    120, 180, 240, 300, 360, 420, 480, 540, 600,
    900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600,
    5400, 7200, 9000, 10800, 12600, 14400, 16200, 18000, 19800, 21600,
    32400, 43200, 54000, 64800, 75600, 86400,
    172800, 259200, 345600, 432000, 518400, 604800,
    1123200, 1641600, 2160000, 2592000,
    4320000, 6048000, 7776000,
    11664000, 15552000,
    31536000,
], dtype=np.int64)

# Total number of time-bucket embedding slots (= number of boundaries + 1, with
# padding=0 included).
#
# This constant is uniquely determined by the length of BUCKET_BOUNDARIES; on
# the model side, ``nn.Embedding(num_embeddings=NUM_TIME_BUCKETS)`` must match
# this value exactly, otherwise an IndexError may be raised at runtime.
#
# That is why ``train.py`` / ``infer.py`` only expose the boolean flag
# ``--use_time_buckets`` and derive the concrete bucket count from here.
NUM_TIME_BUCKETS = len(BUCKET_BOUNDARIES) + 1
DEFAULT_DOMAIN_RECENCY_WINDOWS = (
    300,       # 5 minutes
    900,       # 15 minutes
    3600,      # 1 hour
    21600,     # 6 hours
    86400,     # 1 day
    259200,    # 3 days
    604800,    # 7 days
    2592000,   # 30 days
)

HARDCODED_DOMAIN_RECENCY_WINDOWS: Dict[str, Tuple[int, ...]] = {
    # Probe-derived windows. Keep these intentionally asymmetric: seq_c carries
    # much older history, while seq_d is dominated by near-term dense behavior.
    "seq_a": (
        3600,      # 1 hour
        21600,     # 6 hours
        86400,     # 1 day
        259200,    # 3 days
        604800,    # 7 days
        2592000,   # 30 days
        7776000,   # 90 days
        15552000,  # 180 days
    ),
    "seq_b": (
        3600,
        21600,
        86400,
        259200,
        604800,
        2592000,
        7776000,
        15552000,
    ),
    "seq_c": (
        3600,
        21600,
        86400,
        259200,
        604800,
        2592000,
        7776000,
        15552000,
        31104000,  # 360 days
        51840000,  # 600 days
    ),
    "seq_d": (
        300,       # 5 minutes
        900,       # 15 minutes
        3600,
        21600,
        86400,
        259200,
        604800,
        2592000,
    ),
}


def _validate_recency_window_list(values: Any, name: str) -> List[int]:
    if isinstance(values, tuple):
        values = list(values)
    if not isinstance(values, list):
        raise ValueError(
            f"{name} must be a list of positive integers, got {type(values).__name__}")
    result: List[int] = []
    prev = 0
    for idx, raw in enumerate(values):
        if isinstance(raw, bool):
            raise ValueError(f"{name}[{idx}] must be an integer, got {raw!r}")
        window = int(raw)
        if window <= 0:
            raise ValueError(f"{name}[{idx}] must be positive, got {window}")
        if window <= prev:
            raise ValueError(
                f"{name} must be strictly increasing; {window} <= {prev}")
        result.append(window)
        prev = window
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result


def parse_recency_windows(value: Any) -> Any:
    if value is None:
        return list(DEFAULT_DOMAIN_RECENCY_WINDOWS)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise ValueError("recency windows string must not be empty")
        if stripped in {"domain_specific", "probe", "hardcoded"}:
            return {
                domain: list(windows)
                for domain, windows in HARDCODED_DOMAIN_RECENCY_WINDOWS.items()
            }
        if stripped.startswith("{") or stripped.startswith("["):
            return parse_recency_windows(json.loads(stripped))
        if os.path.exists(stripped):
            with open(stripped, 'r', encoding='utf-8') as f:
                return parse_recency_windows(json.load(f))
        if stripped.endswith(".json"):
            raise FileNotFoundError(f"recency windows JSON not found: {stripped}")
        values = [part.strip() for part in stripped.split(',') if part.strip()]
        return _validate_recency_window_list(values, "recency windows")
    if isinstance(value, dict):
        if not value:
            raise ValueError("domain-specific recency windows must not be empty")
        return {
            str(domain): _validate_recency_window_list(
                windows, f"domain_recency_windows[{domain!r}]")
            for domain, windows in value.items()
        }
    if isinstance(value, (list, tuple)):
        return _validate_recency_window_list(list(value), "recency windows")
    raise ValueError(
        f"recency windows must be a comma-separated string, JSON list, JSON object, "
        f"or file path, got {type(value).__name__}")


def normalize_domain_recency_windows(
    seq_domains: List[str],
    windows: Any,
) -> Tuple[Dict[str, npt.NDArray[np.int64]], Dict[str, int]]:
    parsed = parse_recency_windows(windows)
    if isinstance(parsed, list):
        by_domain = {domain: parsed for domain in seq_domains}
    elif isinstance(parsed, dict):
        expected_domains = set(seq_domains)
        provided_domains = set(parsed.keys())
        missing = sorted(expected_domains - provided_domains)
        extra = sorted(provided_domains - expected_domains)
        if missing or extra:
            raise ValueError(
                "domain recency windows mismatch: "
                f"missing={missing}, extra={extra}, expected={sorted(expected_domains)}")
        by_domain = parsed
    else:
        raise TypeError(
            f"parse_recency_windows returned unsupported type {type(parsed).__name__}")

    normalized: Dict[str, npt.NDArray[np.int64]] = {}
    dims: Dict[str, int] = {}
    for domain in seq_domains:
        values = _validate_recency_window_list(
            list(by_domain[domain]),
            f"domain_recency_windows[{domain!r}]",
        )
        normalized[domain] = np.array(values, dtype=np.int64)
        dims[domain] = 4 + len(values)
    return normalized, dims

# Domain-specific time-delta bucket boundaries used when
# ``--time_bucket_boundaries_json ""`` is passed. Keep every domain list the
# same length because the model has one shared ``num_time_buckets`` value.
HARDCODED_DOMAIN_TIME_BUCKET_BOUNDARIES: Dict[str, List[int]] = {
    "seq_a": [
        60, 300, 900, 3600, 21600, 86400, 259200, 604800, 1209600,
        2592000, 3888000, 5184000, 7776000, 10368000, 12096000,
        15552000, 21600000, 31536000, 51840000, 63072000,
    ],
    "seq_b": [
        60, 300, 900, 3600, 21600, 86400, 172800, 432000, 604800,
        1209600, 2592000, 5184000, 7776000, 10368000, 12096000,
        15552000, 21600000, 31536000, 51840000, 63072000,
    ],
    "seq_c": [
        60, 300, 900, 3600, 21600, 86400, 259200, 604800, 1209600,
        2592000, 5184000, 7776000, 10368000, 15552000, 21600000,
        31536000, 38880000, 46656000, 54432000, 63072000,
    ],
    "seq_d": [
        60, 300, 900, 1800, 3600, 7200, 14400, 21600, 43200,
        86400, 172800, 259200, 432000, 604800, 864000, 1209600,
        1814400, 2592000, 3888000, 7776000,
    ],
}


def _validate_bucket_boundary_list(
    values: Any,
    name: str,
) -> List[int]:
    if not isinstance(values, list):
        raise ValueError(f"{name} must be a list of positive integer boundaries")
    if not values:
        raise ValueError(f"{name} must not be empty")
    result: List[int] = []
    prev = 0
    for idx, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                f"{name}[{idx}] must be an integer boundary, got {value!r}")
        if value <= 0:
            raise ValueError(
                f"{name}[{idx}] must be positive, got {value}")
        if value <= prev:
            raise ValueError(
                f"{name} must be strictly increasing; "
                f"boundary {value} at index {idx} is <= previous {prev}")
        result.append(int(value))
        prev = int(value)
    return result


def load_time_bucket_boundaries_json(path: str) -> Dict[str, List[int]]:
    """Load domain-specific time-delta bucket boundaries from JSON.

    The JSON must be a mapping from every sequence domain name (for example
    ``seq_a``) to a strictly increasing list of positive integer second values.
    Exact domain coverage is validated after ``schema.json`` is loaded, because
    the schema is the source of truth for available domains.
    """
    with open(path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(
            f"time bucket boundaries JSON must be an object, got {type(raw).__name__}")
    return {
        str(domain): _validate_bucket_boundary_list(values, str(domain))
        for domain, values in raw.items()
    }


def normalize_time_bucket_boundaries(
    seq_domains: List[str],
    boundaries_by_domain: Optional[Dict[str, List[int]]] = None,
) -> Tuple[Dict[str, npt.NDArray[np.int64]], int]:
    """Validate and materialize time bucket boundaries for every sequence domain.

    When ``boundaries_by_domain`` is ``None``, every domain uses the legacy
    shared ``BUCKET_BOUNDARIES`` and the resulting behavior is exactly the
    original code path.
    """
    if boundaries_by_domain is None:
        return {
            domain: BUCKET_BOUNDARIES
            for domain in seq_domains
        }, NUM_TIME_BUCKETS

    expected_domains = set(seq_domains)
    provided_domains = set(boundaries_by_domain.keys())
    missing = sorted(expected_domains - provided_domains)
    extra = sorted(provided_domains - expected_domains)
    if missing or extra:
        raise ValueError(
            "time bucket boundaries JSON domain mismatch: "
            f"missing={missing}, extra={extra}, expected={sorted(expected_domains)}")

    normalized: Dict[str, npt.NDArray[np.int64]] = {}
    boundary_count: Optional[int] = None
    for domain in seq_domains:
        values = _validate_bucket_boundary_list(
            boundaries_by_domain[domain],
            f"time_bucket_boundaries[{domain!r}]",
        )
        if boundary_count is None:
            boundary_count = len(values)
        elif len(values) != boundary_count:
            raise ValueError(
                "all domain-specific time bucket boundary lists must have the "
                f"same length; {domain} has {len(values)}, expected {boundary_count}")
        normalized[domain] = np.array(values, dtype=np.int64)

    assert boundary_count is not None
    return normalized, boundary_count + 1


class PCVRParquetDataset(IterableDataset):
    """PCVR dataset that reads raw multi-column Parquet directly.

    - int features: scalar or list (multi-hot); values <= 0 are mapped to 0 (padding).
    - dense features: ``list<float>``, variable-length padded up to ``max_dim``.
    - sequence features: ``list<int64>``, grouped by domain; includes side-info
      columns and an optional timestamp column (used for time-bucketing).
    - label: mapped from ``label_type == 2``.
    """

    def __init__(
        self,
        parquet_path: str,
        schema_path: str,
        batch_size: int = 256,
        seq_max_lens: Optional[Dict[str, int]] = None,
        shuffle: bool = True,
        buffer_batches: int = 20,
        row_group_range: Optional[Tuple[int, int]] = None,
        row_groups: Optional[List[Tuple[str, int, int]]] = None,
        timestamp_min: Optional[int] = None,
        timestamp_max: Optional[int] = None,
        timestamp_ranges: Optional[List[TimestampRange]] = None,
        known_num_rows: Optional[int] = None,
        clip_vocab: bool = True,
        is_training: bool = True,
        time_bucket_boundaries: Optional[Dict[str, List[int]]] = None,
        use_domain_recency_stats: bool = False,
        domain_recency_windows: Optional[Any] = None,
        domain_recency_stats_source: str = 'truncated',
        use_sample_weight: bool = False,
        sample_weight_ranges: Optional[List[SampleWeightRange]] = None,
        sample_weight_default: float = 1.0,
        sample_weight_normalize: str = 'mean',
    ) -> None:
        """
        Args:
            parquet_path: either a directory containing ``*.parquet`` files or
                a single parquet file path.
            schema_path: path of the schema JSON describing feature layouts.
            batch_size: fixed batch size used for the pre-allocated buffers.
            seq_max_lens: optional per-domain override of sequence truncation,
                e.g. ``{'seq_d': 256}``. Domains not listed fall back to the
                schema default of 256.
            shuffle: whether to shuffle within a ``buffer_batches``-sized window.
            buffer_batches: shuffle buffer size in units of batches.
            row_group_range: ``(start, end)`` slice of Row Groups; ``None`` to
                use all Row Groups.
            row_groups: explicit Row Group list. Used when a prior row-level
                time filter has defined a sub-dataset before Row Group splitting.
            timestamp_min: optional inclusive lower bound for row-level
                ``timestamp`` filtering.
            timestamp_max: optional exclusive upper bound for row-level
                ``timestamp`` filtering.
            timestamp_ranges: optional half-open timestamp ranges
                ``[(start, end_exclusive), ...]``. When set, rows are kept
                only if their ``timestamp`` falls in at least one range.
            known_num_rows: exact row count after filtering. Used for logging
                and progress estimates when timestamp filtering is enabled.
            clip_vocab: if True, clip out-of-bound ids to 0; if False, raise.
            is_training: if True, derive ``label`` from ``label_type == 2``;
                if False, return an all-zeros label column.
            time_bucket_boundaries: optional domain-specific bucket boundaries.
                When omitted, the legacy shared ``BUCKET_BOUNDARIES`` are used.
            use_domain_recency_stats: whether to emit per-domain explicit
                recency statistics derived from sequence timestamps.
            domain_recency_windows: shared increasing second windows, or a
                mapping from every sequence domain to that domain's increasing
                second windows. Used for count statistics when
                ``use_domain_recency_stats`` is enabled.
            domain_recency_stats_source: ``truncated`` computes recency stats
                from the same timestamp prefix used by sequence tokens.
                ``full`` keeps sequence tokens truncated but computes recency
                stats from the full raw timestamp list for each domain.
            use_sample_weight: whether to emit a per-sample training weight.
            sample_weight_ranges: closed timestamp intervals with explicit
                weights. Only used when ``use_sample_weight`` is true.
            sample_weight_default: finite positive default weight for rows that
                do not fall into any configured interval.
            sample_weight_normalize: ``mean`` divides emitted weights by the
                training split mean raw weight. ``none`` leaves raw weights.
        """
        super().__init__()

        if sample_weight_normalize not in ('mean', 'none'):
            raise ValueError(
                "sample_weight_normalize must be 'mean' or 'none', got "
                f"{sample_weight_normalize!r}")
        if not np.isfinite(sample_weight_default) or sample_weight_default <= 0.0:
            raise ValueError(
                "sample_weight_default must be a finite positive number, got "
                f"{sample_weight_default!r}")
        normalized_sample_weight_ranges = normalize_sample_weight_ranges(
            sample_weight_ranges)
        if use_sample_weight and not is_training:
            raise ValueError("use_sample_weight is only valid for training datasets")
        if use_sample_weight and not normalized_sample_weight_ranges:
            raise ValueError(
                "use_sample_weight=True requires at least one "
                "sample_weight_ranges entry")
        if domain_recency_stats_source not in ('truncated', 'full'):
            raise ValueError(
                "domain_recency_stats_source must be 'truncated' or 'full', "
                f"got {domain_recency_stats_source!r}")
        if domain_recency_stats_source != 'truncated' and not use_domain_recency_stats:
            raise ValueError(
                "domain_recency_stats_source is only meaningful when "
                "use_domain_recency_stats=True")

        if timestamp_min is not None and timestamp_max is not None:
            if timestamp_min >= timestamp_max:
                raise ValueError(
                    f"timestamp_min must be < timestamp_max, got "
                    f"{timestamp_min} >= {timestamp_max}")
        normalized_timestamp_ranges = _normalize_exclusive_timestamp_ranges(
            timestamp_ranges)

        # Accept either a directory or a single file path.
        if os.path.isdir(parquet_path):
            import glob
            files = sorted(glob.glob(os.path.join(parquet_path, '*.parquet')))
            if not files:
                raise FileNotFoundError(f"No .parquet files in {parquet_path}")
            self._parquet_files = files
        else:
            self._parquet_files = [parquet_path]

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.buffer_batches = buffer_batches
        self.clip_vocab = clip_vocab
        self.is_training = is_training
        self.use_domain_recency_stats = bool(use_domain_recency_stats)
        self.domain_recency_stats_source = domain_recency_stats_source
        self.use_sample_weight = bool(use_sample_weight)
        self.sample_weight_ranges = normalized_sample_weight_ranges
        self.sample_weight_default = float(sample_weight_default)
        self.sample_weight_normalize = sample_weight_normalize
        self.sample_weight_divisor: Optional[float] = (
            1.0 if sample_weight_normalize == 'none' else None
        )
        self.timestamp_min = timestamp_min
        self.timestamp_max = timestamp_max
        self.timestamp_ranges = normalized_timestamp_ranges
        self._timestamp_filter_enabled = (
            timestamp_min is not None or timestamp_max is not None
            or normalized_timestamp_ranges is not None
        )
        self._known_num_rows = known_num_rows is not None
        # Out-of-bound statistics:
        #   {(group, col_idx): {'count': N, 'max': M, 'min_oob': M, 'vocab': V}}
        self._oob_stats: Dict[Tuple[str, int], Dict[str, int]] = {}

        # Build the list of Row Groups.
        if row_groups is not None:
            if row_group_range is not None:
                raise ValueError("row_groups and row_group_range are mutually exclusive")
            self._rg_list = list(row_groups)
        else:
            self._rg_list = []
            for f in self._parquet_files:
                pf = pq.ParquetFile(f)
                for i in range(pf.metadata.num_row_groups):
                    self._rg_list.append((f, i, pf.metadata.row_group(i).num_rows))

            if row_group_range is not None:
                start, end = row_group_range
                self._rg_list = self._rg_list[start:end]

        self.num_rows = (
            int(known_num_rows)
            if known_num_rows is not None
            else sum(r[2] for r in self._rg_list)
        )

        # Load schema.json.
        self._load_schema(schema_path, seq_max_lens or {})
        self.time_bucket_boundaries, self.num_time_buckets = (
            normalize_time_bucket_boundaries(
                self.seq_domains,
                time_bucket_boundaries,
            )
        )
        self.domain_recency_windows, self.domain_recency_stats_dims = (
            normalize_domain_recency_windows(
                self.seq_domains,
                domain_recency_windows,
            )
        )
        self.domain_recency_stats_dim = (
            next(iter(set(self.domain_recency_stats_dims.values())))
            if len(set(self.domain_recency_stats_dims.values())) == 1
            else 0
        )
        if self.use_domain_recency_stats:
            logging.info(
                "Domain recency windows: %s; stats dims: %s; source=%s",
                {
                    domain: self.domain_recency_windows[domain].tolist()
                    for domain in self.seq_domains
                },
                self.domain_recency_stats_dims,
                self.domain_recency_stats_source,
            )

        # ---- Pre-compute column index lookup ----
        pf = pq.ParquetFile(self._parquet_files[0])
        schema_names = pf.schema_arrow.names
        self._col_idx = {name: i for i, name in enumerate(schema_names)}
        if self._timestamp_filter_enabled and 'timestamp' not in self._col_idx:
            raise KeyError(
                "timestamp split/filter was requested, but the parquet schema "
                "does not contain a 'timestamp' column")

        # ---- Pre-allocate numpy buffers ----
        B = batch_size
        self._buf_user_int = np.zeros((B, self.user_int_schema.total_dim), dtype=np.int64)
        self._buf_item_int = np.zeros((B, self.item_int_schema.total_dim), dtype=np.int64)
        self._buf_user_dense = np.zeros((B, self.user_dense_schema.total_dim), dtype=np.float32)
        self._buf_seq = {}
        self._buf_seq_tb = {}
        self._buf_seq_lens = {}
        for domain in self.seq_domains:
            max_len = self._seq_maxlen[domain]
            n_feats = len(self.sideinfo_fids[domain])
            self._buf_seq[domain] = np.zeros((B, n_feats, max_len), dtype=np.int64)
            self._buf_seq_tb[domain] = np.zeros((B, max_len), dtype=np.int64)
            self._buf_seq_lens[domain] = np.zeros(B, dtype=np.int64)

        # ---- Pre-compute (col_idx, offset, vocab_size) plans for int columns ----
        self._user_int_plan = []  # [(col_idx, dim, offset, vocab_size), ...]
        offset = 0
        for fid, vs, dim in self._user_int_cols:
            ci = self._col_idx.get(f'user_int_feats_{fid}')
            self._user_int_plan.append((ci, dim, offset, vs))
            offset += dim

        self._item_int_plan = []
        offset = 0
        for fid, vs, dim in self._item_int_cols:
            ci = self._col_idx.get(f'item_int_feats_{fid}')
            self._item_int_plan.append((ci, dim, offset, vs))
            offset += dim

        self._user_dense_plan = []
        offset = 0
        for fid, dim in self._user_dense_cols:
            ci = self._col_idx.get(f'user_dense_feats_{fid}')
            self._user_dense_plan.append((ci, dim, offset))
            offset += dim

        # Sequence column plan: {domain: ([(col_idx, feat_slot, vocab_size), ...], ts_col_idx)}
        self._seq_plan = {}
        for domain in self.seq_domains:
            prefix = self._seq_prefix[domain]
            sideinfo_fids = self.sideinfo_fids[domain]
            ts_fid = self.ts_fids[domain]
            side_plan = []
            for slot, fid in enumerate(sideinfo_fids):
                ci = self._col_idx.get(f'{prefix}_{fid}')
                vs = self.seq_vocab_sizes[domain][fid]
                side_plan.append((ci, slot, vs))
            ts_ci = self._col_idx.get(f'{prefix}_{ts_fid}') if ts_fid is not None else None
            self._seq_plan[domain] = (side_plan, ts_ci)
            if self.use_domain_recency_stats and ts_ci is None:
                raise KeyError(
                    f"use_domain_recency_stats=True requires timestamp column "
                    f"for {domain}, but {prefix}_{ts_fid} is missing")

        logging.info(
            f"PCVRParquetDataset: {self.num_rows} rows from "
            f"{len(self._parquet_files)} file(s), batch_size={batch_size}, "
            f"buffer_batches={buffer_batches}, shuffle={shuffle}, "
            f"timestamp_min={timestamp_min}, timestamp_max={timestamp_max}, "
            f"timestamp_ranges={normalized_timestamp_ranges}")

    def _load_schema(self, schema_path: str, seq_max_lens: Dict[str, int]) -> None:
        """Populate per-group schema information from ``schema_path``."""
        with open(schema_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)

        # ---- user_int: [[fid, vocab_size, dim], ...] ----
        self._user_int_cols: List[List[int]] = raw['user_int']
        self.user_int_schema: FeatureSchema = FeatureSchema()
        self.user_int_vocab_sizes: List[int] = []
        for fid, vs, dim in self._user_int_cols:
            self.user_int_schema.add(fid, dim)
            self.user_int_vocab_sizes.extend([vs] * dim)

        # ---- item_int ----
        self._item_int_cols: List[List[int]] = raw['item_int']
        self.item_int_schema: FeatureSchema = FeatureSchema()
        self.item_int_vocab_sizes: List[int] = []
        for fid, vs, dim in self._item_int_cols:
            self.item_int_schema.add(fid, dim)
            self.item_int_vocab_sizes.extend([vs] * dim)

        # ---- user_dense: [[fid, dim], ...] ----
        self._user_dense_cols: List[List[int]] = raw['user_dense']
        self.user_dense_schema: FeatureSchema = FeatureSchema()
        for fid, dim in self._user_dense_cols:
            self.user_dense_schema.add(fid, dim)

        # ---- item_dense (empty) ----
        self.item_dense_schema: FeatureSchema = FeatureSchema()

        # ---- sequence domains ----
        self._seq_cfg: Dict[str, Dict[str, Any]] = raw['seq']
        self.seq_domains: List[str] = sorted(self._seq_cfg.keys())
        self.seq_feature_ids: Dict[str, List[int]] = {}
        self.seq_vocab_sizes: Dict[str, Dict[int, int]] = {}
        self.seq_domain_vocab_sizes: Dict[str, List[int]] = {}
        self.ts_fids: Dict[str, Optional[int]] = {}
        self.sideinfo_fids: Dict[str, List[int]] = {}
        self._seq_prefix: Dict[str, str] = {}
        self._seq_maxlen: Dict[str, int] = {}

        for domain in self.seq_domains:
            cfg = self._seq_cfg[domain]
            self._seq_prefix[domain] = cfg['prefix']
            ts_fid = cfg['ts_fid']
            self.ts_fids[domain] = ts_fid

            all_fids = [fid for fid, vs in cfg['features']]
            self.seq_feature_ids[domain] = all_fids
            self.seq_vocab_sizes[domain] = {fid: vs for fid, vs in cfg['features']}

            sideinfo = [fid for fid in all_fids if fid != ts_fid]
            self.sideinfo_fids[domain] = sideinfo
            self.seq_domain_vocab_sizes[domain] = [
                self.seq_vocab_sizes[domain][fid] for fid in sideinfo
            ]

            # max_len: from seq_max_lens arg; unspecified domains fall back to 256.
            self._seq_maxlen[domain] = seq_max_lens.get(domain, 256)

    def __len__(self) -> int:
        if self._known_num_rows:
            return (self.num_rows + self.batch_size - 1) // self.batch_size
        # Ceiling per Row Group; this is an upper bound on the true batch count.
        return sum((n + self.batch_size - 1) // self.batch_size
                   for _, _, n in self._rg_list)

    def max_timestamp(self, scan_batch_size: int = 65536) -> int:
        """Return the maximum ``timestamp`` in this dataset after filters.

        This scans only the timestamp column over the dataset's Row Group slice
        and applies the same row-level timestamp_min/timestamp_max filters used
        during iteration. It is intended for resolving sample-level time feature
        reference points from the actual training split.
        """
        if 'timestamp' not in self._col_idx:
            raise KeyError(
                "Cannot resolve max timestamp because the parquet schema does "
                "not contain a 'timestamp' column")

        stats = self.timestamp_stats(scan_batch_size=scan_batch_size)
        return stats['timestamp_max']

    def timestamp_stats(self, scan_batch_size: int = 65536) -> Dict[str, int]:
        """Scan this dataset partition and return real timestamp statistics.

        The scan follows the dataset's actual Row Group list and applies the
        same row-level timestamp_min/timestamp_max filters used by iteration.
        This is used to audit train/valid splits from the resulting datasets,
        not from split parameters.
        """
        if 'timestamp' not in self._col_idx:
            raise KeyError(
                "Cannot compute timestamp stats because the parquet schema "
                "does not contain a 'timestamp' column")

        count = 0
        min_ts: Optional[int] = None
        max_ts: Optional[int] = None
        for file_path, rg_idx, _ in self._rg_list:
            pf = pq.ParquetFile(file_path)
            for batch in pf.iter_batches(
                batch_size=scan_batch_size,
                row_groups=[rg_idx],
                columns=['timestamp'],
            ):
                col = batch.column(0)
                if col.null_count:
                    raise ValueError(
                        f"timestamp contains null values in {file_path}, "
                        f"row_group={rg_idx}")
                arr = col.to_numpy(zero_copy_only=False).astype(np.int64)
                arr = _apply_timestamp_filter_np(
                    arr,
                    timestamp_min=self.timestamp_min,
                    timestamp_max=self.timestamp_max,
                    timestamp_ranges=self.timestamp_ranges,
                )
                if arr.size == 0:
                    continue
                count += int(arr.shape[0])
                batch_min = int(arr.min())
                batch_max = int(arr.max())
                min_ts = batch_min if min_ts is None else min(min_ts, batch_min)
                max_ts = batch_max if max_ts is None else max(max_ts, batch_max)

        if max_ts is None:
            raise ValueError(
                "No rows found while scanning timestamp stats. Check data_dir, "
                "split settings, and timestamp filters.")
        return {
            'rows': count,
            'timestamp_min': int(min_ts),
            'timestamp_max': int(max_ts),
        }

    def _timestamp_filter_mask_np(
        self,
        timestamps: "npt.NDArray[np.int64]",
    ) -> "npt.NDArray[np.bool_]":
        mask = np.ones(timestamps.shape, dtype=bool)
        if self.timestamp_min is not None:
            mask &= timestamps >= self.timestamp_min
        if self.timestamp_max is not None:
            mask &= timestamps < self.timestamp_max
        if self.timestamp_ranges is not None:
            range_mask = np.zeros(timestamps.shape, dtype=bool)
            for start, end in self.timestamp_ranges:
                range_mask |= (timestamps >= start) & (timestamps < end)
            mask &= range_mask
        return mask

    def _sample_weights_np(
        self,
        timestamps: "npt.NDArray[np.int64]",
        normalize: bool = True,
    ) -> "npt.NDArray[np.float32]":
        weights = np.full(
            timestamps.shape,
            self.sample_weight_default,
            dtype=np.float32,
        )
        for start, end, weight in self.sample_weight_ranges:
            mask = (timestamps >= start) & (timestamps <= end)
            weights[mask] = np.float32(weight)
        if normalize and self.sample_weight_normalize == 'mean':
            if self.sample_weight_divisor is None:
                raise RuntimeError(
                    "sample_weight_normalize='mean' requires "
                    "sample_weight_audit(set_normalizer=True) before iteration")
            weights = weights / np.float32(self.sample_weight_divisor)
        return weights

    def sample_weight_audit(
        self,
        scan_batch_size: int = 65536,
        set_normalizer: bool = False,
    ) -> Dict[str, Any]:
        """Scan the actual training split and summarize sample weights."""
        if not self.use_sample_weight:
            raise RuntimeError("sample_weight_audit called when sample weighting is disabled")
        if 'timestamp' not in self._col_idx:
            raise KeyError(
                "Cannot compute sample weights because the parquet schema "
                "does not contain a 'timestamp' column")
        if 'label_type' not in self._col_idx:
            raise KeyError(
                "Cannot audit sample weights because the parquet schema does "
                "not contain a 'label_type' column")

        total_rows = 0
        pos_rows = 0
        weight_sum = 0.0
        weighted_pos_sum = 0.0
        weight_min = float('inf')
        weight_max = float('-inf')
        default_rows = 0
        default_pos = 0
        range_rows = [0 for _ in self.sample_weight_ranges]
        range_pos = [0 for _ in self.sample_weight_ranges]

        for file_path, rg_idx, _ in self._rg_list:
            pf = pq.ParquetFile(file_path)
            for batch in pf.iter_batches(
                batch_size=scan_batch_size,
                row_groups=[rg_idx],
                columns=['timestamp', 'label_type'],
            ):
                ts_col = batch.column(0)
                label_col = batch.column(1)
                if ts_col.null_count:
                    raise ValueError(
                        f"timestamp contains null values in {file_path}, "
                        f"row_group={rg_idx}")
                timestamps = ts_col.to_numpy(zero_copy_only=False).astype(np.int64)
                labels = (label_col.fill_null(0)
                          .to_numpy(zero_copy_only=False).astype(np.int64) == 2)
                keep_mask = self._timestamp_filter_mask_np(timestamps)
                if not keep_mask.any():
                    continue
                timestamps = timestamps[keep_mask]
                labels = labels[keep_mask]
                raw_weights = self._sample_weights_np(timestamps, normalize=False)

                total_rows += int(timestamps.shape[0])
                pos_rows += int(labels.sum())
                weight_sum += float(raw_weights.sum(dtype=np.float64))
                weighted_pos_sum += float(raw_weights[labels].sum(dtype=np.float64))
                weight_min = min(weight_min, float(raw_weights.min()))
                weight_max = max(weight_max, float(raw_weights.max()))

                matched_any = np.zeros(timestamps.shape, dtype=bool)
                for idx, (start, end, _) in enumerate(self.sample_weight_ranges):
                    range_mask = (timestamps >= start) & (timestamps <= end)
                    if range_mask.any():
                        range_rows[idx] += int(range_mask.sum())
                        range_pos[idx] += int(labels[range_mask].sum())
                        matched_any |= range_mask
                default_mask = ~matched_any
                if default_mask.any():
                    default_rows += int(default_mask.sum())
                    default_pos += int(labels[default_mask].sum())

        if total_rows <= 0:
            raise ValueError(
                "No rows found while auditing sample weights. Check split "
                "settings and timestamp filters.")
        for idx, count in enumerate(range_rows):
            if count <= 0:
                start, end, weight = self.sample_weight_ranges[idx]
                raise ValueError(
                    "sample_weight_ranges entry matched zero training rows: "
                    f"[{start}, {end}], weight={weight}")

        raw_mean = weight_sum / float(total_rows)
        if not np.isfinite(raw_mean) or raw_mean <= 0.0:
            raise RuntimeError(f"Invalid sample weight mean: {raw_mean}")
        if set_normalizer and self.sample_weight_normalize == 'mean':
            self.sample_weight_divisor = float(raw_mean)

        range_summaries = []
        for idx, (start, end, weight) in enumerate(self.sample_weight_ranges):
            rows = range_rows[idx]
            pos = range_pos[idx]
            range_summaries.append({
                'index': idx + 1,
                'start': start,
                'end': end,
                'weight': float(weight),
                'rows': rows,
                'pos': pos,
                'label_rate': float(pos / rows) if rows > 0 else float('nan'),
            })

        return {
            'rows': total_rows,
            'pos': pos_rows,
            'neg': total_rows - pos_rows,
            'label_rate': float(pos_rows / total_rows),
            'weighted_label_rate': float(weighted_pos_sum / weight_sum),
            'weight_min': weight_min,
            'weight_max': weight_max,
            'weight_mean_raw': float(raw_mean),
            'weight_divisor': float(self.sample_weight_divisor or 1.0),
            'weight_mean_effective': (
                float(raw_mean / self.sample_weight_divisor)
                if self.sample_weight_divisor else float(raw_mean)
            ),
            'default': {
                'weight': float(self.sample_weight_default),
                'rows': default_rows,
                'pos': default_pos,
                'label_rate': (
                    float(default_pos / default_rows)
                    if default_rows > 0 else float('nan')
                ),
            },
            'ranges': range_summaries,
        }

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        worker_info = torch.utils.data.get_worker_info()
        rg_list = self._rg_list
        if worker_info is not None and worker_info.num_workers > 1:
            rg_list = [rg for i, rg in enumerate(rg_list)
                       if i % worker_info.num_workers == worker_info.id]

        buffer: List[Dict[str, Any]] = []
        for file_path, rg_idx, _ in rg_list:
            pf = pq.ParquetFile(file_path)
            for batch in pf.iter_batches(batch_size=self.batch_size, row_groups=[rg_idx]):
                batch = self._filter_batch_by_timestamp(batch)
                if batch is None:
                    continue
                batch_dict = self._convert_batch(batch)
                if self.shuffle and self.buffer_batches > 1:
                    buffer.append(batch_dict)
                    if len(buffer) >= self.buffer_batches:
                        yield from self._flush_buffer(buffer)
                        buffer = []
                else:
                    yield batch_dict

        if buffer:
            yield from self._flush_buffer(buffer)

        del buffer
        gc.collect()

    def _flush_buffer(
        self, buffer: List[Dict[str, Any]]
    ) -> Iterator[Dict[str, Any]]:
        """Concatenate the buffered batches, shuffle at the row level, then
        re-slice and yield batch-sized chunks.
        """
        merged: Dict[str, torch.Tensor] = {}
        non_tensor_keys: Dict[str, Any] = {}
        for k in buffer[0].keys():
            if isinstance(buffer[0][k], torch.Tensor):
                merged[k] = torch.cat([b[k] for b in buffer], dim=0)
            else:
                non_tensor_keys[k] = buffer[0][k]
        total_rows = merged['label'].shape[0]
        rand_idx = torch.randperm(total_rows) if self.shuffle else torch.arange(total_rows)
        for i in range(0, total_rows, self.batch_size):
            end = min(i + self.batch_size, total_rows)
            batch: Dict[str, Any] = {k: v[rand_idx[i:end]] for k, v in merged.items()}
            batch.update(non_tensor_keys)
            yield batch
        del merged
        buffer.clear()

    # ---- Helpers ----

    def _filter_batch_by_timestamp(
        self,
        batch: "pa.RecordBatch",
    ) -> Optional["pa.RecordBatch"]:
        """Apply the row-level timestamp filter for time-based train/valid split."""
        if not self._timestamp_filter_enabled:
            return batch

        ts_col = batch.column(self._col_idx['timestamp'])
        if ts_col.null_count:
            raise ValueError(
                "timestamp contains null values; timestamp split requires "
                "non-null timestamps")

        mask = None
        if self.timestamp_min is not None:
            cond = pc.greater_equal(
                ts_col,
                pa.scalar(self.timestamp_min, type=ts_col.type),
            )
            mask = cond if mask is None else pc.and_(mask, cond)
        if self.timestamp_max is not None:
            cond = pc.less(
                ts_col,
                pa.scalar(self.timestamp_max, type=ts_col.type),
            )
            mask = cond if mask is None else pc.and_(mask, cond)
        if self.timestamp_ranges is not None:
            range_mask = None
            for start, end in self.timestamp_ranges:
                lower = pc.greater_equal(
                    ts_col,
                    pa.scalar(start, type=ts_col.type),
                )
                upper = pc.less(
                    ts_col,
                    pa.scalar(end, type=ts_col.type),
                )
                cond = pc.and_(lower, upper)
                range_mask = cond if range_mask is None else pc.or_(range_mask, cond)
            mask = range_mask if mask is None else pc.and_(mask, range_mask)

        if mask is None:
            return batch

        mask = pc.fill_null(mask, False)
        filtered = batch.filter(mask)
        if filtered.num_rows == 0:
            return None
        return filtered

    def _record_oob(
        self,
        group: str,
        col_idx: int,
        arr: "npt.NDArray[np.int64]",
        vocab_size: int,
    ) -> None:
        """Record out-of-bound indices and (optionally) clip them to 0,
        without printing to the console.
        """
        oob_mask = arr >= vocab_size
        if not oob_mask.any():
            return
        key = (group, col_idx)
        oob_vals = arr[oob_mask]
        n = int(oob_mask.sum())
        mx = int(oob_vals.max())
        mn = int(oob_vals.min())
        if key in self._oob_stats:
            s = self._oob_stats[key]
            s['count'] += n
            s['max'] = max(s['max'], mx)
            s['min_oob'] = min(s['min_oob'], mn)
        else:
            self._oob_stats[key] = {
                'count': n, 'max': mx, 'min_oob': mn, 'vocab': vocab_size,
            }
        if self.clip_vocab:
            arr[oob_mask] = 0
        else:
            raise ValueError(
                f"{group} col_idx={col_idx}: {n} values out of range "
                f"[0, {vocab_size}), actual=[{mn}, {mx}]. "
                f"Use clip_vocab=True to clip or fix schema.json")

    def dump_oob_stats(self, path: Optional[str] = None) -> None:
        """Dump out-of-bound statistics to a file if ``path`` is provided,
        otherwise to ``logging.info``.
        """
        if not self._oob_stats:
            logging.info("No out-of-bound values detected.")
            return
        lines = ["=== Out-of-Bound Stats ==="]
        for (group, ci), s in sorted(self._oob_stats.items()):
            direction = "TOO_HIGH" if s['min_oob'] >= s['vocab'] else "TOO_LOW"
            lines.append(
                f"  {group} col_idx={ci}: vocab={s['vocab']}, "
                f"oob_count={s['count']}, range=[{s['min_oob']}, {s['max']}], "
                f"{direction}")
        msg = "\n".join(lines)
        if path:
            with open(path, 'w') as f:
                f.write(msg + "\n")
            logging.info(f"OOB stats written to {path}")
        else:
            logging.info(msg)

    def _compute_domain_recency_stats(
        self,
        domain: str,
        time_diff: "npt.NDArray[np.int64]",
        valid_mask: "npt.NDArray[np.bool_]",
    ) -> "npt.NDArray[np.float32]":
        """Build explicit per-domain sequence recency statistics.

        Features are log1p-scaled to keep magnitudes bounded:
        ``[valid_count, min_age, mean_age, max_age, count<=window_1, ...]``.
        """
        B = time_diff.shape[0]
        windows = self.domain_recency_windows[domain]
        stats = np.zeros((B, self.domain_recency_stats_dims[domain]), dtype=np.float32)
        valid_count = valid_mask.sum(axis=1).astype(np.float32)
        has_valid = valid_count > 0
        stats[:, 0] = np.log1p(valid_count)
        if not has_valid.any():
            return stats

        ages = np.where(valid_mask, time_diff, 0).astype(np.float32)
        min_source = np.where(valid_mask, time_diff, np.iinfo(np.int64).max)
        min_age = np.where(has_valid, min_source.min(axis=1), 0)
        max_age = np.where(
            has_valid,
            np.where(valid_mask, time_diff, 0).max(axis=1),
            0,
        )
        sum_age = ages.sum(axis=1)
        mean_age = np.zeros(B, dtype=np.float32)
        mean_age[has_valid] = sum_age[has_valid] / valid_count[has_valid]

        stats[:, 1] = np.log1p(min_age.astype(np.float32))
        stats[:, 2] = np.log1p(mean_age)
        stats[:, 3] = np.log1p(max_age.astype(np.float32))

        for idx, window in enumerate(windows):
            count = ((time_diff <= int(window)) & valid_mask).sum(axis=1)
            stats[:, 4 + idx] = np.log1p(count.astype(np.float32))

        return stats

    def _compute_domain_recency_stats_from_full_ts(
        self,
        domain: str,
        sample_timestamps: "npt.NDArray[np.int64]",
        ts_offsets: "npt.NDArray[np.int64]",
        ts_values: "npt.NDArray[np.int64]",
        B: int,
    ) -> "npt.NDArray[np.float32]":
        """Build per-domain recency stats from the full raw timestamp list.

        This intentionally does not change the sequence tokens themselves:
        tokenization may still use ``seq_max_lens`` while these statistics
        keep long-history activity/count information visible to the model.
        """
        windows = self.domain_recency_windows[domain]
        stats = np.zeros((B, self.domain_recency_stats_dims[domain]), dtype=np.float32)

        for i in range(B):
            start = int(ts_offsets[i])
            end = int(ts_offsets[i + 1])
            if end <= start:
                continue

            raw_ts = ts_values[start:end]
            valid_ts = raw_ts[raw_ts > 0]
            valid_count = int(valid_ts.shape[0])
            if valid_count == 0:
                continue

            ages = np.maximum(int(sample_timestamps[i]) - valid_ts.astype(np.int64), 0)
            ages_f = ages.astype(np.float32)
            stats[i, 0] = np.log1p(np.float32(valid_count))
            stats[i, 1] = np.log1p(np.float32(ages.min()))
            stats[i, 2] = np.log1p(ages_f.mean())
            stats[i, 3] = np.log1p(np.float32(ages.max()))

            for idx, window in enumerate(windows):
                stats[i, 4 + idx] = np.log1p(
                    np.float32((ages <= int(window)).sum()))

        return stats

    def _pad_varlen_int_column(
        self,
        arrow_col: "pa.ListArray",
        max_len: int,
        B: int,
    ) -> Tuple["npt.NDArray[np.int64]", "npt.NDArray[np.int64]"]:
        """Pad an Arrow ``ListArray`` of ints to shape ``[B, max_len]``.

        Values <= 0 are mapped to 0 (padding). Note: the raw data contains -1
        (missing); currently treated the same way as 0 (padding).

        Returns:
            A tuple ``(padded, lengths)`` where ``padded`` has shape
            ``[B, max_len]`` and ``lengths`` has shape ``[B]``.
        """
        offsets = arrow_col.offsets.to_numpy()
        values = arrow_col.values.to_numpy()

        padded = np.zeros((B, max_len), dtype=np.int64)
        lengths = np.zeros(B, dtype=np.int64)

        for i in range(B):
            start, end = int(offsets[i]), int(offsets[i + 1])
            raw_len = end - start
            if raw_len <= 0:
                continue
            use_len = min(raw_len, max_len)
            padded[i, :use_len] = values[start:start + use_len]
            lengths[i] = use_len

        padded[padded <= 0] = 0
        return padded, lengths

    # Backwards-compatible alias kept for bench_raw_dataset.py and other
    # external callers that pre-date the rename. New code should call
    # `_pad_varlen_int_column` directly.
    _pad_varlen_column = _pad_varlen_int_column

    def _pad_varlen_float_column(
        self,
        arrow_col: "pa.ListArray",
        max_dim: int,
        B: int,
    ) -> "npt.NDArray[np.float32]":
        """Pad an Arrow ``ListArray<float>`` to shape ``[B, max_dim]``."""
        offsets = arrow_col.offsets.to_numpy()
        values = arrow_col.values.to_numpy()

        padded = np.zeros((B, max_dim), dtype=np.float32)

        for i in range(B):
            start, end = int(offsets[i]), int(offsets[i + 1])
            raw_len = end - start
            if raw_len <= 0:
                continue
            use_len = min(raw_len, max_dim)
            padded[i, :use_len] = values[start:start + use_len]

        return padded

    def _convert_batch(self, batch: "pa.RecordBatch") -> Dict[str, Any]:
        """Convert an Arrow RecordBatch into a training-ready dict of tensors."""
        B = batch.num_rows

        # ---- meta ----
        timestamps = batch.column(self._col_idx['timestamp']).to_numpy().astype(np.int64)
        if self.is_training:
            labels = (batch.column(self._col_idx['label_type']).fill_null(0)
                      .to_numpy(zero_copy_only=False).astype(np.int64) == 2).astype(np.int64)
        else:
            labels = np.zeros(B, dtype=np.int64)
        sample_weights = None
        if self.use_sample_weight:
            sample_weights = self._sample_weights_np(timestamps, normalize=True)
        user_ids = batch.column(self._col_idx['user_id']).to_pylist()

        # ---- user_int: write into pre-allocated buffer ----
        # Note: null -> 0 (via fill_null), -1 -> 0 (via arr<=0); missing values
        # are treated the same as padding. Features with vs==0 have no vocab
        # information and are forced to 0 on the dataset side so that the
        # model's 1-slot Embedding (created for vs=0) is never indexed out of
        # range.
        user_int = self._buf_user_int[:B]
        user_int[:] = 0
        for ci, dim, offset, vs in self._user_int_plan:
            col = batch.column(ci)
            if dim == 1:
                arr = col.fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64)
                arr[arr <= 0] = 0
                if vs > 0:
                    self._record_oob('user_int', ci, arr, vs)
                else:
                    arr[:] = 0
                user_int[:, offset] = arr
            else:
                padded, _ = self._pad_varlen_int_column(col, dim, B)
                if vs > 0:
                    self._record_oob('user_int', ci, padded, vs)
                else:
                    padded[:] = 0
                user_int[:, offset:offset + dim] = padded

        # ---- item_int ----
        item_int = self._buf_item_int[:B]
        item_int[:] = 0
        for ci, dim, offset, vs in self._item_int_plan:
            col = batch.column(ci)
            if dim == 1:
                arr = col.fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64)
                arr[arr <= 0] = 0
                if vs > 0:
                    self._record_oob('item_int', ci, arr, vs)
                else:
                    arr[:] = 0
                item_int[:, offset] = arr
            else:
                padded, _ = self._pad_varlen_int_column(col, dim, B)
                if vs > 0:
                    self._record_oob('item_int', ci, padded, vs)
                else:
                    padded[:] = 0
                item_int[:, offset:offset + dim] = padded

        # ---- user_dense ----
        user_dense = self._buf_user_dense[:B]
        user_dense[:] = 0
        for ci, dim, offset in self._user_dense_plan:
            col = batch.column(ci)
            padded = self._pad_varlen_float_column(col, dim, B)
            user_dense[:, offset:offset + dim] = padded

        # Sequence data accumulators (filled during the loop below).
        seq_data_dict: Dict[str, torch.Tensor] = {}
        seq_lens_dict: Dict[str, torch.Tensor] = {}
        seq_tb_dict: Dict[str, torch.Tensor] = {}
        seq_recency_dict: Dict[str, torch.Tensor] = {}

        # ---- Sequence features: fused padding directly into the 3D buffer ----
        for domain in self.seq_domains:
            max_len = self._seq_maxlen[domain]
            side_plan, ts_ci = self._seq_plan[domain]

            # Write directly into the pre-allocated 3D buffer.
            out = self._buf_seq[domain][:B]
            out[:] = 0
            lengths = self._buf_seq_lens[domain][:B]
            lengths[:] = 0

            # Fused path: first collect (offsets, values, vocab_size, col_idx)
            # for every side-info column, then fill the buffer in a single pass.
            col_data = []
            for ci, slot, vs in side_plan:
                col = batch.column(ci)
                col_data.append((col.offsets.to_numpy(), col.values.to_numpy(), vs, ci))

            for c, (offs, vals, vs, ci) in enumerate(col_data):
                for i in range(B):
                    s = int(offs[i])
                    e = int(offs[i + 1])
                    rl = e - s
                    if rl <= 0:
                        continue
                    ul = min(rl, max_len)
                    out[i, c, :ul] = vals[s:s + ul]
                    if ul > lengths[i]:
                        lengths[i] = ul

            # Values <= 0 -> 0.
            out[out <= 0] = 0

            # Check out-of-bound values per feature's vocab_size.
            # vs==0 means no vocab info; force the whole slice to 0 so that
            # the model's 1-slot Embedding is never indexed out of range.
            for c, (_, _, vs, ci) in enumerate(col_data):
                slice_c = out[:, c, :]
                if vs > 0:
                    self._record_oob(f'seq_{domain}', ci, slice_c, vs)
                else:
                    slice_c[:] = 0

            seq_data_dict[domain] = torch.from_numpy(out.copy())
            seq_lens_dict[f'{domain}_len'] = torch.from_numpy(lengths.copy())

            # Time bucketing.
            time_bucket = self._buf_seq_tb[domain][:B]
            time_bucket[:] = 0
            if ts_ci is not None:
                ts_col = batch.column(ts_ci)
                ts_offs = ts_col.offsets.to_numpy()
                ts_vals = ts_col.values.to_numpy()
                # Pad timestamps into shape (B, max_len).
                ts_padded = np.zeros((B, max_len), dtype=np.int64)
                for i in range(B):
                    s = int(ts_offs[i])
                    e = int(ts_offs[i + 1])
                    rl = e - s
                    if rl <= 0:
                        continue
                    ul = min(rl, max_len)
                    ts_padded[i, :ul] = ts_vals[s:s + ul]

                ts_expanded = timestamps.reshape(-1, 1)
                time_diff = np.maximum(ts_expanded - ts_padded, 0)
                valid_ts_mask = ts_padded > 0
                # np.searchsorted returns values in [0, len(BUCKET_BOUNDARIES)].
                # After +1 the nominal range is [1, len(BUCKET_BOUNDARIES)+1];
                # the upper bound only appears when time_diff exceeds the
                # largest boundary (~1 year) and would index past
                # nn.Embedding(NUM_TIME_BUCKETS=len(BUCKET_BOUNDARIES)+1).
                # Clip raw result to [0, len(BUCKET_BOUNDARIES)-1] so the final
                # bucket id (after +1) stays within [1, len(BUCKET_BOUNDARIES)]
                # and is always a valid Embedding index. Time-diffs beyond the
                # largest boundary collapse into the last bucket.
                boundaries = self.time_bucket_boundaries[domain]
                raw_buckets = np.clip(
                    np.searchsorted(boundaries, time_diff.ravel()),
                    0, len(boundaries) - 1,
                )
                buckets = raw_buckets.reshape(B, max_len) + 1
                buckets[ts_padded == 0] = 0
                time_bucket[:] = buckets
                if self.use_domain_recency_stats:
                    if self.domain_recency_stats_source == 'truncated':
                        recency_stats = self._compute_domain_recency_stats(
                            domain,
                            time_diff,
                            valid_ts_mask,
                        )
                    elif self.domain_recency_stats_source == 'full':
                        recency_stats = self._compute_domain_recency_stats_from_full_ts(
                            domain,
                            timestamps,
                            ts_offs,
                            ts_vals,
                            B,
                        )
                    else:
                        raise RuntimeError(
                            "unreachable domain_recency_stats_source: "
                            f"{self.domain_recency_stats_source!r}")
                    seq_recency_dict[f'{domain}_recency_stats'] = torch.from_numpy(
                        recency_stats)

            seq_tb_dict[f'{domain}_time_bucket'] = torch.from_numpy(time_bucket.copy())

        # ---- Assemble result dict (after time stats have been written to user_dense) ----
        result = {
            'user_int_feats': torch.from_numpy(user_int.copy()),
            'user_dense_feats': torch.from_numpy(user_dense.copy()),
            'item_int_feats': torch.from_numpy(item_int.copy()),
            'item_dense_feats': torch.zeros(B, 0, dtype=torch.float32),
            'label': torch.from_numpy(labels),
            'timestamp': torch.from_numpy(timestamps),
            'user_id': user_ids,
            '_seq_domains': self.seq_domains,
        }
        if sample_weights is not None:
            result['sample_weight'] = torch.from_numpy(sample_weights)
        # Merge sequence data, lengths, time buckets.
        result.update(seq_data_dict)
        result.update(seq_lens_dict)
        result.update(seq_tb_dict)
        result.update(seq_recency_dict)

        return result


def get_pcvr_data(
    data_dir: str,
    schema_path: str,
    batch_size: int = 256,
    valid_ratio: float = 0.1,
    train_ratio: float = 1.0,
    split_mode: str = 'rowgroup',
    num_workers: int = 16,
    prefetch_factor: int = 2,
    buffer_batches: int = 20,
    shuffle_train: bool = True,
    seed: int = 42,
    clip_vocab: bool = True,
    seq_max_lens: Optional[Dict[str, int]] = None,
    **kwargs: Any,
) -> Tuple[DataLoader, Union[DataLoader, List[Tuple[str, DataLoader]]], PCVRParquetDataset]:
    """Create train / valid DataLoaders from raw multi-column Parquet files.

    Split modes:
      - ``timestamp``: compute a row-level cutoff from the ``timestamp`` column.
        Rows with ``timestamp < cutoff`` are train; rows with
        ``timestamp >= cutoff`` are validation.
      - ``rowgroup``: reproduce the baseline behavior, using the tail
        ``valid_ratio`` fraction of Row Groups as validation.
      - ``manual_time``: use two explicit closed timestamp ranges passed via
        ``train_val_range=(train_min, train_max, valid_min, valid_max)``.

    When ``interval`` is True, rows are first filtered to either the legacy
    single closed ``time_range=[START, END]`` or the multi-interval closed
    ``time_ranges=[[START1, END1], ...]`` union, and the requested split mode
    is then applied to that sub-dataset.

    When ``multi_valid_time_ranges`` is provided, the training split is still
    built from ``split_mode`` as usual, but validation is replaced by a list of
    independent full-dataset time-window loaders named ``valid1``,
    ``valid2``, ... . Overlapping windows are allowed and are not merged.

    Returns:
        A tuple ``(train_loader, valid_loader, train_dataset)``. With
        ``multi_valid_time_ranges``, the second element is a list of
        ``(name, loader)`` pairs; otherwise it is the legacy single DataLoader.
        The third
        element is returned so the caller can access the feature schema
        (``user_int_schema``, ``item_int_schema``, ...) needed to construct
        the model.
    """
    random.seed(seed)
    interval: bool = kwargs.get('interval', False)
    time_range = kwargs.get('time_range', None)
    time_ranges = kwargs.get('time_ranges', None)
    train_val_range = kwargs.get('train_val_range', None)
    time_bucket_boundaries = kwargs.get('time_bucket_boundaries', None)
    use_domain_recency_stats = kwargs.get('use_domain_recency_stats', False)
    domain_recency_windows = kwargs.get('domain_recency_windows', None)
    domain_recency_stats_source = kwargs.get(
        'domain_recency_stats_source', 'truncated')
    multi_valid_time_ranges = kwargs.get('multi_valid_time_ranges', None)
    rowgroup_valid_sub_time_start = kwargs.get(
        'rowgroup_valid_sub_time_start', None)
    rowgroup_valid_sub_time_end = kwargs.get(
        'rowgroup_valid_sub_time_end', None)
    use_sample_weight = bool(kwargs.get('use_sample_weight', False))
    sample_weight_ranges = normalize_sample_weight_ranges(
        kwargs.get('sample_weight_ranges', None))
    sample_weight_default = float(kwargs.get('sample_weight_default', 1.0))
    sample_weight_normalize = kwargs.get('sample_weight_normalize', 'mean')
    closed_interval_ranges: Optional[List[TimestampRange]] = None
    interval_ranges: Optional[List[TimestampRange]] = None
    multi_valid_closed_ranges: Optional[List[TimestampRange]] = (
        normalize_closed_timestamp_windows(multi_valid_time_ranges)
        if multi_valid_time_ranges is not None else None
    )
    has_rowgroup_valid_sub = (
        rowgroup_valid_sub_time_start is not None
        or rowgroup_valid_sub_time_end is not None
    )
    rowgroup_valid_sub_range: Optional[TimestampRange] = None
    if has_rowgroup_valid_sub:
        if rowgroup_valid_sub_time_start is None or rowgroup_valid_sub_time_end is None:
            raise ValueError(
                "rowgroup valid sub range requires both "
                "rowgroup_valid_sub_time_start and rowgroup_valid_sub_time_end")
        if split_mode != 'rowgroup':
            raise ValueError(
                "rowgroup valid sub range is only supported with "
                "--split_mode rowgroup")
        if multi_valid_closed_ranges is not None:
            raise ValueError(
                "rowgroup valid sub range cannot be combined with "
                "--multi_valid_time_ranges")
        start = int(rowgroup_valid_sub_time_start)
        end = int(rowgroup_valid_sub_time_end)
        if start > end:
            raise ValueError(
                f"rowgroup valid sub range must satisfy START <= END, got "
                f"{start} > {end}")
        rowgroup_valid_sub_range = (start, end)
    if interval:
        closed_interval_ranges = normalize_closed_timestamp_ranges(
            time_range=time_range,
            time_ranges=time_ranges,
        )
        if not closed_interval_ranges:
            raise ValueError(
                "--interval requires --time_range START END or "
                "--time_ranges START1 END1 [START2 END2 ...]")
        interval_ranges = _closed_to_exclusive_timestamp_ranges(
            closed_interval_ranges)
    if split_mode not in ('timestamp', 'rowgroup', 'manual_time'):
        raise ValueError(
            f"split_mode must be one of 'timestamp', 'rowgroup', 'manual_time', "
            f"got {split_mode!r}")
    if split_mode != 'manual_time' and not (0.0 < valid_ratio < 1.0):
        raise ValueError(f"valid_ratio must be in (0, 1), got {valid_ratio}")
    manual_train_min: Optional[int] = None
    manual_train_max_exclusive: Optional[int] = None
    manual_valid_min: Optional[int] = None
    manual_valid_max_exclusive: Optional[int] = None
    if split_mode == 'manual_time':
        if interval:
            raise ValueError(
                "--split_mode manual_time cannot be combined with --interval; "
                "use --train_val_range to define both partitions")
        if train_val_range is None or len(train_val_range) != 4:
            raise ValueError(
                "--split_mode manual_time requires --train_val_range "
                "TRAIN_MIN TRAIN_MAX VALID_MIN VALID_MAX")
        train_min, train_max, valid_min, valid_max = [int(v) for v in train_val_range]
        if train_min <= 0 or train_max <= 0 or valid_min <= 0 or valid_max <= 0:
            raise ValueError(
                "--train_val_range values must be positive Unix timestamps, "
                f"got {train_val_range}")
        if train_min > train_max:
            raise ValueError(
                f"manual train range must satisfy TRAIN_MIN <= TRAIN_MAX, got "
                f"{train_min} > {train_max}")
        if valid_min > valid_max:
            raise ValueError(
                f"manual valid range must satisfy VALID_MIN <= VALID_MAX, got "
                f"{valid_min} > {valid_max}")
        if not (train_max < valid_min or valid_max < train_min):
            raise ValueError(
                "manual train/valid time ranges overlap; refusing to create a "
                f"leaky validation split: train=[{train_min}, {train_max}], "
                f"valid=[{valid_min}, {valid_max}]")
        manual_train_min = train_min
        manual_train_max_exclusive = train_max + 1
        manual_valid_min = valid_min
        manual_valid_max_exclusive = valid_max + 1
    elif train_val_range is not None:
        raise ValueError("--train_val_range is only valid with --split_mode manual_time")
    if split_mode == 'timestamp' and train_ratio < 1.0:
        raise ValueError(
            "--train_ratio < 1.0 is only supported with --split_mode rowgroup. "
            "Timestamp mode uses all rows before the timestamp cutoff.")
    if split_mode == 'manual_time' and train_ratio < 1.0:
        raise ValueError(
            "--train_ratio < 1.0 is not supported with --split_mode manual_time. "
            "Set the train interval explicitly through --train_val_range.")

    import glob as _glob
    pq_files = sorted(_glob.glob(os.path.join(data_dir, '*.parquet')))
    if not pq_files:
        raise FileNotFoundError(f"No .parquet files in {data_dir}")

    rg_info = []
    for f in pq_files:
        pf = pq.ParquetFile(f)
        for i in range(pf.metadata.num_row_groups):
            rg_info.append((f, i, pf.metadata.row_group(i).num_rows))

    timestamp_cutoff: Optional[int] = None
    train_row_group_range: Optional[Tuple[int, int]] = None
    valid_row_group_range: Optional[Tuple[int, int]] = None
    train_row_groups: Optional[List[Tuple[str, int, int]]] = None
    valid_row_groups: Optional[List[Tuple[str, int, int]]] = None
    train_timestamp_min: Optional[int] = None
    train_timestamp_max: Optional[int] = None
    valid_timestamp_min: Optional[int] = None
    valid_timestamp_max: Optional[int] = None
    rowgroup_valid_sub_rows: Optional[int] = None

    interval_min = interval_ranges[0][0] if interval_ranges else None
    interval_max = interval_ranges[-1][1] if interval_ranges else None
    if split_mode == 'manual_time':
        train_row_groups = _filter_rg_info_by_timestamp(
            rg_info=rg_info,
            timestamp_min=manual_train_min,
            timestamp_max=manual_train_max_exclusive,
        )
        valid_row_groups = _filter_rg_info_by_timestamp(
            rg_info=rg_info,
            timestamp_min=manual_valid_min,
            timestamp_max=manual_valid_max_exclusive,
        )
        if not train_row_groups:
            raise ValueError(
                f"No training rows found in manual closed time range "
                f"[{manual_train_min}, {manual_train_max_exclusive - 1}]")
        if not valid_row_groups:
            raise ValueError(
                f"No validation rows found in manual closed time range "
                f"[{manual_valid_min}, {manual_valid_max_exclusive - 1}]")
        split_rg_info = rg_info
    else:
        split_rg_info = (
            _filter_rg_info_by_timestamp(
                rg_info=rg_info,
                timestamp_min=interval_min,
                timestamp_max=interval_max,
                timestamp_ranges=interval_ranges,
            )
            if interval else rg_info
        )
        if not split_rg_info:
            raise ValueError(
                f"No rows found after applying interval filter "
                f"{_format_closed_timestamp_ranges(closed_interval_ranges)}")
    total_rgs = len(split_rg_info)

    if split_mode == 'manual_time':
        train_rows = sum(r[2] for r in train_row_groups)
        valid_rows = sum(r[2] for r in valid_row_groups)
        train_timestamp_min = manual_train_min
        train_timestamp_max = manual_train_max_exclusive
        valid_timestamp_min = manual_valid_min
        valid_timestamp_max = manual_valid_max_exclusive

        logging.info(
            "Manual time split: train uses closed range [%s, %s] "
            "(%s rows from %s Row Groups), valid uses closed range [%s, %s] "
            "(%s rows from %s Row Groups)",
            manual_train_min,
            manual_train_max_exclusive - 1,
            train_rows,
            len(train_row_groups),
            manual_valid_min,
            manual_valid_max_exclusive - 1,
            valid_rows,
            len(valid_row_groups),
        )
    elif split_mode == 'timestamp':
        timestamp_cutoff, train_rows, valid_rows = _compute_timestamp_split(
            rg_info=split_rg_info,
            valid_ratio=valid_ratio,
            timestamp_min=interval_min,
            timestamp_max=interval_max,
            timestamp_ranges=interval_ranges,
        )
        train_timestamp_min = interval_min
        train_timestamp_max = timestamp_cutoff
        valid_timestamp_min = timestamp_cutoff
        valid_timestamp_max = interval_max
    else:
        n_valid_rgs = max(1, int(total_rgs * valid_ratio))
        n_train_rgs = total_rgs - n_valid_rgs

        # train_ratio: use only the first N% of the training Row Groups.
        if train_ratio < 1.0:
            n_train_rgs = max(1, int(n_train_rgs * train_ratio))
            logging.info(
                f"train_ratio={train_ratio}: using {n_train_rgs} train Row Groups")

        train_rows = sum(r[2] for r in split_rg_info[:n_train_rgs])
        valid_rows = sum(r[2] for r in split_rg_info[n_train_rgs:])
        rowgroup_valid_base_rg_info = split_rg_info[n_train_rgs:]
        if interval:
            train_row_groups = split_rg_info[:n_train_rgs]
            valid_row_groups = rowgroup_valid_base_rg_info
            train_timestamp_min = interval_min
            train_timestamp_max = interval_max
            valid_timestamp_min = interval_min
            valid_timestamp_max = interval_max
        else:
            train_row_group_range = (0, n_train_rgs)
            valid_row_group_range = (n_train_rgs, total_rgs)

        logging.info(
            f"Row Group split: {n_train_rgs} train ({train_rows} rows), "
            f"{n_valid_rgs} valid ({valid_rows} rows)")

    train_dataset = PCVRParquetDataset(
        parquet_path=data_dir,
        schema_path=schema_path,
        batch_size=batch_size,
        seq_max_lens=seq_max_lens,
        shuffle=shuffle_train,
        buffer_batches=buffer_batches,
        row_group_range=train_row_group_range,
        row_groups=train_row_groups,
        timestamp_min=train_timestamp_min,
        timestamp_max=train_timestamp_max,
        timestamp_ranges=interval_ranges,
        known_num_rows=train_rows if split_mode in ('timestamp', 'manual_time') else None,
        clip_vocab=clip_vocab,
        time_bucket_boundaries=time_bucket_boundaries,
        use_domain_recency_stats=use_domain_recency_stats,
        domain_recency_windows=domain_recency_windows,
        domain_recency_stats_source=domain_recency_stats_source,
        use_sample_weight=use_sample_weight,
        sample_weight_ranges=sample_weight_ranges,
        sample_weight_default=sample_weight_default,
        sample_weight_normalize=sample_weight_normalize,
    )

    if use_sample_weight:
        audit = train_dataset.sample_weight_audit(set_normalizer=True)
        logging.info(
            "SAMPLE_WEIGHT_AUDIT rows=%s pos=%s neg=%s label_rate=%.6f "
            "weighted_label_rate=%.6f weight_min=%.6f weight_max=%.6f "
            "weight_mean_raw=%.6f weight_divisor=%.6f "
            "weight_mean_effective=%.6f normalize=%s default_weight=%.6f "
            "default_rows=%s default_label_rate=%.6f",
            audit['rows'],
            audit['pos'],
            audit['neg'],
            audit['label_rate'],
            audit['weighted_label_rate'],
            audit['weight_min'],
            audit['weight_max'],
            audit['weight_mean_raw'],
            audit['weight_divisor'],
            audit['weight_mean_effective'],
            sample_weight_normalize,
            audit['default']['weight'],
            audit['default']['rows'],
            audit['default']['label_rate'],
        )
        for range_info in audit['ranges']:
            logging.info(
                "SAMPLE_WEIGHT_RANGE index=%s start=%s end=%s weight=%.6f "
                "rows=%s pos=%s label_rate=%.6f",
                range_info['index'],
                range_info['start'],
                range_info['end'],
                range_info['weight'],
                range_info['rows'],
                range_info['pos'],
                range_info['label_rate'],
            )

    use_cuda = torch.cuda.is_available()
    _train_kw = {}
    if num_workers > 0:
        _train_kw['persistent_workers'] = True
        _train_kw['prefetch_factor'] = prefetch_factor

    train_loader = DataLoader(
        train_dataset, batch_size=None,
        num_workers=num_workers, pin_memory=use_cuda, **_train_kw,
    )

    valid_dataset = None
    valid_loader: Any
    audit_datasets: List[Tuple[str, PCVRParquetDataset]] = [('train', train_dataset)]

    if multi_valid_closed_ranges is not None:
        valid_entries = []
        logging.info(
            "Multi validation enabled: %s independent full-dataset time windows; "
            "valid1 is used as the primary AUC for checkpoint naming, best_model, "
            "and early stopping.",
            len(multi_valid_closed_ranges),
        )
        for idx, (start, end) in enumerate(multi_valid_closed_ranges, start=1):
            name = f"valid{idx}"
            end_exclusive = end + 1
            window_row_groups = _filter_rg_info_by_timestamp(
                rg_info=rg_info,
                timestamp_min=start,
                timestamp_max=end_exclusive,
            )
            window_rows = sum(r[2] for r in window_row_groups)
            if not window_row_groups or window_rows <= 0:
                raise ValueError(
                    f"No validation rows found for {name} closed time range "
                    f"[{start}, {end}]")
            window_dataset = PCVRParquetDataset(
                parquet_path=data_dir,
                schema_path=schema_path,
                batch_size=batch_size,
                seq_max_lens=seq_max_lens,
                shuffle=False,
                buffer_batches=0,
                row_groups=window_row_groups,
                timestamp_min=start,
                timestamp_max=end_exclusive,
                known_num_rows=window_rows,
                clip_vocab=clip_vocab,
                time_bucket_boundaries=time_bucket_boundaries,
                use_domain_recency_stats=use_domain_recency_stats,
                domain_recency_windows=domain_recency_windows,
                domain_recency_stats_source=domain_recency_stats_source,
            )
            window_loader = DataLoader(
                window_dataset, batch_size=None,
                num_workers=0, pin_memory=use_cuda,
            )
            valid_entries.append((name, window_loader))
            audit_datasets.append((name, window_dataset))
            logging.info(
                "Multi validation window %s: closed_range=[%s, %s], rows=%s, "
                "row_groups=%s",
                name,
                start,
                end,
                window_rows,
                len(window_row_groups),
            )
        valid_loader = valid_entries
    else:
        valid_dataset = PCVRParquetDataset(
            parquet_path=data_dir,
            schema_path=schema_path,
            batch_size=batch_size,
            seq_max_lens=seq_max_lens,
            shuffle=False,
            buffer_batches=0,
            row_group_range=valid_row_group_range,
            row_groups=valid_row_groups,
            timestamp_min=valid_timestamp_min,
            timestamp_max=valid_timestamp_max,
            timestamp_ranges=interval_ranges,
            known_num_rows=valid_rows if split_mode in ('timestamp', 'manual_time') else None,
            clip_vocab=clip_vocab,
            time_bucket_boundaries=time_bucket_boundaries,
            use_domain_recency_stats=use_domain_recency_stats,
            domain_recency_windows=domain_recency_windows,
            domain_recency_stats_source=domain_recency_stats_source,
        )
        valid_loader = DataLoader(
            valid_dataset, batch_size=None,
            num_workers=0, pin_memory=use_cuda,
        )
        audit_datasets.append(('valid', valid_dataset))

        if rowgroup_valid_sub_range is not None:
            sub_start, sub_end = rowgroup_valid_sub_range
            sub_end_exclusive = sub_end + 1
            sub_rg_source = (
                valid_row_groups
                if valid_row_groups is not None
                else split_rg_info[valid_row_group_range[0]:valid_row_group_range[1]]
            )
            sub_row_groups = _filter_rg_info_by_timestamp(
                rg_info=sub_rg_source,
                timestamp_min=sub_start,
                timestamp_max=sub_end_exclusive,
                timestamp_ranges=interval_ranges,
            )
            rowgroup_valid_sub_rows = sum(r[2] for r in sub_row_groups)
            if not sub_row_groups or rowgroup_valid_sub_rows <= 0:
                raise ValueError(
                    "No rows found for rowgroup validation sub window "
                    f"[{sub_start}, {sub_end}] inside the rowgroup validation "
                    "split. Check the timestamp range or valid_ratio.")

            sub_dataset = PCVRParquetDataset(
                parquet_path=data_dir,
                schema_path=schema_path,
                batch_size=batch_size,
                seq_max_lens=seq_max_lens,
                shuffle=False,
                buffer_batches=0,
                row_groups=sub_row_groups,
                timestamp_min=sub_start,
                timestamp_max=sub_end_exclusive,
                timestamp_ranges=interval_ranges,
                known_num_rows=rowgroup_valid_sub_rows,
                clip_vocab=clip_vocab,
                time_bucket_boundaries=time_bucket_boundaries,
                use_domain_recency_stats=use_domain_recency_stats,
                domain_recency_windows=domain_recency_windows,
                domain_recency_stats_source=domain_recency_stats_source,
            )
            sub_loader = DataLoader(
                sub_dataset, batch_size=None,
                num_workers=0, pin_memory=use_cuda,
            )
            valid_loader = [('valid', valid_loader), ('valid_sub', sub_loader)]
            audit_datasets.append(('valid_sub', sub_dataset))
            logging.info(
                "Rowgroup validation sub window: closed_range=[%s, %s], "
                "rows=%s, row_groups=%s. This is an observation-only metric; "
                "training and primary valid are unchanged.",
                sub_start,
                sub_end,
                rowgroup_valid_sub_rows,
                len(sub_row_groups),
            )

    for split_name, dataset in audit_datasets:
        stats = dataset.timestamp_stats()
        if stats['rows'] != dataset.num_rows:
            raise RuntimeError(
                f"{split_name} split timestamp audit row mismatch: "
                f"dataset.num_rows={dataset.num_rows}, scanned_rows={stats['rows']}. "
                "This indicates the split metadata and actual timestamp filters "
                "are inconsistent.")
        logging.info(
            "SPLIT_AUDIT split=%s rows=%s timestamp_min=%s timestamp_max=%s",
            split_name,
            stats['rows'],
            stats['timestamp_min'],
            stats['timestamp_max'],
        )

    if multi_valid_closed_ranges is not None:
        valid_log_desc = f"multi_valid: {len(multi_valid_closed_ranges)} windows"
    elif rowgroup_valid_sub_rows is not None:
        valid_log_desc = (
            f"valid: {valid_rows} rows, valid_sub: {rowgroup_valid_sub_rows} rows"
        )
    else:
        valid_log_desc = f"valid: {valid_rows} rows"

    if split_mode == 'manual_time':
        logging.info(
            f"Parquet split_mode=manual_time, "
            f"train_time_range=[{manual_train_min}, {manual_train_max_exclusive - 1}], "
            f"valid_time_range=[{manual_valid_min}, {manual_valid_max_exclusive - 1}], "
            f"train: {train_rows} rows, {valid_log_desc}, "
            f"batch_size={batch_size}, buffer_batches={buffer_batches}, "
            f"prefetch_factor={prefetch_factor if num_workers > 0 else 0}")
    elif interval:
        logging.info(
            f"Parquet split_mode={split_mode}, interval=True, "
            f"time_ranges={_format_closed_timestamp_ranges(closed_interval_ranges)}, "
            f"sub_row_groups={total_rgs}, sub_rows={sum(r[2] for r in split_rg_info)}, "
            f"train: {train_rows} rows, {valid_log_desc}, "
            f"timestamp_cutoff={timestamp_cutoff}, "
            f"batch_size={batch_size}, buffer_batches={buffer_batches}, "
            f"prefetch_factor={prefetch_factor if num_workers > 0 else 0}")
    else:
        logging.info(
            f"Parquet split_mode={split_mode}, train: {train_rows} rows, "
            f"{valid_log_desc}, timestamp_cutoff={timestamp_cutoff}, "
            f"batch_size={batch_size}, buffer_batches={buffer_batches}, "
            f"prefetch_factor={prefetch_factor if num_workers > 0 else 0}")

    return train_loader, valid_loader, train_dataset


def _filter_rg_info_by_timestamp(
    rg_info: List[Tuple[str, int, int]],
    timestamp_min: Optional[int] = None,
    timestamp_max: Optional[int] = None,
    timestamp_ranges: Optional[List[TimestampRange]] = None,
    scan_batch_size: int = 65536,
) -> List[Tuple[str, int, int]]:
    """Return Row Groups with their row counts after a timestamp filter."""
    timestamp_ranges = _normalize_exclusive_timestamp_ranges(timestamp_ranges)
    if timestamp_min is None and timestamp_max is None and timestamp_ranges is None:
        return rg_info

    filtered_rg_info: List[Tuple[str, int, int]] = []
    for file_path, rg_idx, _ in rg_info:
        pf = pq.ParquetFile(file_path)
        if 'timestamp' not in pf.schema_arrow.names:
            raise KeyError(f"{file_path} does not contain required column 'timestamp'")

        count = 0
        for batch in pf.iter_batches(
            batch_size=scan_batch_size,
            row_groups=[rg_idx],
            columns=['timestamp'],
        ):
            col = batch.column(0)
            if col.null_count:
                raise ValueError(
                    f"timestamp contains null values in {file_path}, "
                    f"row_group={rg_idx}")
            arr = col.to_numpy(zero_copy_only=False).astype(np.int64)
            arr = _apply_timestamp_filter_np(
                arr,
                timestamp_min=timestamp_min,
                timestamp_max=timestamp_max,
                timestamp_ranges=timestamp_ranges,
            )
            count += int(arr.shape[0])

        if count > 0:
            filtered_rg_info.append((file_path, rg_idx, count))

    return filtered_rg_info


def _compute_timestamp_split(
    rg_info: List[Tuple[str, int, int]],
    valid_ratio: float,
    timestamp_min: Optional[int] = None,
    timestamp_max: Optional[int] = None,
    timestamp_ranges: Optional[List[TimestampRange]] = None,
    scan_batch_size: int = 65536,
) -> Tuple[int, int, int]:
    """Compute the timestamp cutoff for a strict row-level time split."""
    timestamp_ranges = _normalize_exclusive_timestamp_ranges(timestamp_ranges)
    total_rows = sum(n for _, _, n in rg_info)
    if total_rows <= 1:
        raise ValueError(f"Need at least 2 rows for timestamp split, got {total_rows}")

    chunks: List[np.ndarray] = []
    for file_path, rg_idx, _ in rg_info:
        pf = pq.ParquetFile(file_path)
        if 'timestamp' not in pf.schema_arrow.names:
            raise KeyError(f"{file_path} does not contain required column 'timestamp'")
        for batch in pf.iter_batches(
            batch_size=scan_batch_size,
            row_groups=[rg_idx],
            columns=['timestamp'],
        ):
            col = batch.column(0)
            if col.null_count:
                raise ValueError(
                    f"timestamp contains null values in {file_path}, "
                    f"row_group={rg_idx}")
            arr = col.to_numpy(zero_copy_only=False).astype(np.int64)
            arr = _apply_timestamp_filter_np(
                arr,
                timestamp_min=timestamp_min,
                timestamp_max=timestamp_max,
                timestamp_ranges=timestamp_ranges,
            )
            if arr.size > 0:
                chunks.append(arr)

    if not chunks:
        raise ValueError("No timestamp values found while computing timestamp split")

    timestamps = np.concatenate(chunks)
    if timestamps.shape[0] != total_rows:
        raise ValueError(
            f"Timestamp scan row count mismatch: scanned {timestamps.shape[0]} "
            f"but parquet metadata reports {total_rows}")

    n_valid_target = max(1, int(total_rows * valid_ratio))
    cutoff_index = total_rows - n_valid_target
    if cutoff_index <= 0 or cutoff_index >= total_rows:
        raise ValueError(
            f"Invalid timestamp cutoff index {cutoff_index} for total_rows={total_rows}, "
            f"valid_ratio={valid_ratio}")

    cutoff = int(np.partition(timestamps, cutoff_index)[cutoff_index])
    train_rows = int((timestamps < cutoff).sum())
    valid_rows = int((timestamps >= cutoff).sum())
    if train_rows <= 0 or valid_rows <= 0:
        raise ValueError(
            "Timestamp split produced an empty partition: "
            f"cutoff={cutoff}, train_rows={train_rows}, valid_rows={valid_rows}. "
            "This usually means the timestamp distribution has too many identical values.")

    equal_cutoff_rows = int((timestamps == cutoff).sum())
    logging.info(
        "Timestamp split: train uses timestamp < %s, valid uses timestamp >= %s; "
        "train_rows=%s, valid_rows=%s, target_valid_ratio=%.6f, "
        "actual_valid_ratio=%.6f, min_timestamp=%s, max_timestamp=%s, "
        "rows_equal_cutoff=%s",
        cutoff,
        cutoff,
        train_rows,
        valid_rows,
        valid_ratio,
        valid_rows / total_rows,
        int(timestamps.min()),
        int(timestamps.max()),
        equal_cutoff_rows,
    )
    return cutoff, train_rows, valid_rows
