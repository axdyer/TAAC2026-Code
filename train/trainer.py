"""PCVRHyFormer pointwise trainer (binary-classification, AUC-monitored).

Despite the historical "Ranking" suffix in the class name, the training loop
uses pointwise BCE / Focal loss and evaluates Binary AUC + binary logloss.
"""

import os
import shutil
import logging
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from utils import sigmoid_focal_loss, EarlyStopping
from model import ModelInput


class PCVRHyFormerRankingTrainer:
    """PCVRHyFormer trainer for pointwise binary classification.

    Uses PCVR data layout:
    - user_int_feats, user_dense_feats
    - item_int_feats, item_dense_feats
    - seq_a, seq_b, seq_c, seq_d (each with *_len companion)
    - label (binary)

    Loss: BCEWithLogitsLoss or Focal Loss.
    Metrics: BinaryAUROC + binary logloss.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        valid_loader: Any,
        lr: float,
        num_epochs: int,
        device: str,
        save_dir: str,
        early_stopping: EarlyStopping,
        loss_type: str = 'bce',
        focal_alpha: float = 0.1,
        focal_gamma: float = 2.0,
        sparse_lr: float = 0.05,
        sparse_weight_decay: float = 0.0,
        reinit_sparse_after_epoch: int = 1,
        reinit_cardinality_threshold: int = 0,
        ckpt_params: Optional[Dict[str, Any]] = None,
        writer: Optional[Any] = None,
        schema_path: Optional[str] = None,
        ns_groups_path: Optional[str] = None,
        eval_every_n_steps: int = 0,
        train_config: Optional[Dict[str, Any]] = None,
        amp_dtype: str = 'none',
        compile_model: bool = False,
        compile_mode: str = 'reduce-overhead',
        show_progress_bar: bool = False,
    ) -> None:
        self.raw_model: nn.Module = model
        self.model: nn.Module = model
        self.train_loader: DataLoader = train_loader
        self.valid_loaders, self.multi_valid_enabled = self._normalize_valid_loaders(
            valid_loader)
        self.valid_loader: DataLoader = self.valid_loaders[0][1]
        self.writer = writer
        # schema_path is copied alongside every checkpoint so that infer.py can
        # rebuild the exact same feature schema the model was trained with.
        self.schema_path: Optional[str] = schema_path
        # ns_groups_path is optional; copied next to schema.json when provided
        # and points at an existing file. Keeping the JSON inside the ckpt dir
        # makes the checkpoint self-contained for evaluation environments that
        # do not ship ns_groups.json separately.
        self.ns_groups_path: Optional[str] = ns_groups_path

        # Dual optimizer: Adagrad for sparse Embeddings, AdamW for dense params.
        self.sparse_optimizer: Optional[torch.optim.Optimizer]
        if hasattr(self.raw_model, 'get_sparse_params'):
            sparse_params = self.raw_model.get_sparse_params()
            dense_params = self.raw_model.get_dense_params()
            sparse_param_count = sum(p.numel() for p in sparse_params)
            dense_param_count = sum(p.numel() for p in dense_params)
            logging.info(f"Sparse params: {len(sparse_params)} tensors, {sparse_param_count:,} parameters (Adagrad lr={sparse_lr})")
            logging.info(f"Dense params: {len(dense_params)} tensors, {dense_param_count:,} parameters (AdamW lr={lr})")
            self.sparse_optimizer = torch.optim.Adagrad(
                sparse_params, lr=sparse_lr, weight_decay=sparse_weight_decay
            )
            self.dense_optimizer: torch.optim.Optimizer = torch.optim.AdamW(
                dense_params, lr=lr, betas=(0.9, 0.98)
            )
        else:
            self.sparse_optimizer = None
            self.dense_optimizer = torch.optim.AdamW(
                self.raw_model.parameters(), lr=lr, betas=(0.9, 0.98)
            )

        self.amp_dtype_name: str = amp_dtype
        self.amp_torch_dtype: Optional[torch.dtype] = self._resolve_amp_dtype(
            amp_dtype, device)
        self.use_amp: bool = self.amp_torch_dtype is not None

        if compile_model:
            if not hasattr(torch, 'compile'):
                raise RuntimeError("compile_model=True requires torch.compile, but this torch build does not provide it")
            logging.info(f"Compiling model with torch.compile(mode={compile_mode})")
            self.model = torch.compile(self.raw_model, mode=compile_mode)

        self.num_epochs: int = num_epochs
        self.device: str = device
        self.save_dir: str = save_dir
        self.early_stopping: EarlyStopping = early_stopping
        self.loss_type: str = loss_type
        self.focal_alpha: float = focal_alpha
        self.focal_gamma: float = focal_gamma
        self.reinit_sparse_after_epoch: int = reinit_sparse_after_epoch
        self.reinit_cardinality_threshold: int = reinit_cardinality_threshold
        self.sparse_lr: float = sparse_lr
        self.sparse_weight_decay: float = sparse_weight_decay
        self.ckpt_params: Dict[str, Any] = ckpt_params or {}
        self.eval_every_n_steps: int = eval_every_n_steps
        self.train_config: Optional[Dict[str, Any]] = train_config
        self.eval_checkpoint_index: int = 0
        self.show_progress_bar: bool = show_progress_bar
        self._last_eval_diagnostics_log: Optional[str] = None

        logging.info(f"PCVRHyFormerRankingTrainer loss_type={loss_type}, "
                     f"focal_alpha={focal_alpha}, focal_gamma={focal_gamma}, "
                     f"reinit_sparse_after_epoch={reinit_sparse_after_epoch}, "
                     f"amp_dtype={amp_dtype}, compile_model={compile_model}, "
                     f"show_progress_bar={show_progress_bar}")
        logging.info(
            "Validation loaders: %s",
            ", ".join(name for name, _ in self.valid_loaders),
        )
        if self.multi_valid_enabled:
            logging.info(
                "Multi validation mode is enabled; %s is the primary metric "
                "for checkpoint naming, best_model, and early stopping.",
                self.valid_loaders[0][0],
            )

    @staticmethod
    def _normalize_valid_loaders(
        valid_loader: Any,
    ) -> Tuple[List[Tuple[str, DataLoader]], bool]:
        """Normalize a legacy loader or a list of named loaders."""
        if isinstance(valid_loader, DataLoader):
            return [('valid', valid_loader)], False
        if isinstance(valid_loader, list):
            if not valid_loader:
                raise ValueError("valid_loader list must contain at least one loader")
            result: List[Tuple[str, DataLoader]] = []
            seen = set()
            for idx, entry in enumerate(valid_loader, start=1):
                if not (
                    isinstance(entry, tuple)
                    and len(entry) == 2
                    and isinstance(entry[0], str)
                    and isinstance(entry[1], DataLoader)
                ):
                    raise TypeError(
                        "valid_loader list entries must be (name, DataLoader) "
                        f"tuples, got {entry!r}")
                name, loader = entry
                if not name:
                    raise ValueError(f"validation loader #{idx} has an empty name")
                if name in seen:
                    raise ValueError(f"duplicate validation loader name: {name}")
                seen.add(name)
                result.append((name, loader))
            return result, True
        raise TypeError(
            "valid_loader must be a DataLoader or a list of (name, DataLoader) "
            f"tuples, got {type(valid_loader).__name__}")

    @staticmethod
    def _resolve_amp_dtype(amp_dtype: str, device: str) -> Optional[torch.dtype]:
        """Resolve explicit AMP mode.

        There is intentionally no silent fallback: when bf16 is requested on
        unsupported hardware, the run stops before training starts.
        """
        if amp_dtype == 'none':
            return None
        if amp_dtype != 'bf16':
            raise ValueError(f"Unsupported amp_dtype={amp_dtype!r}; expected 'none' or 'bf16'")
        if not device.startswith('cuda'):
            raise RuntimeError("amp_dtype=bf16 requires a CUDA device")
        if not torch.cuda.is_available():
            raise RuntimeError("amp_dtype=bf16 requested but CUDA is not available")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("amp_dtype=bf16 requested but this GPU/torch build does not support bf16")
        return torch.bfloat16

    def _autocast_context(self):
        if self.amp_torch_dtype is None:
            return nullcontext()
        return torch.autocast(device_type='cuda', dtype=self.amp_torch_dtype)

    @staticmethod
    def _format_metric(value: float) -> str:
        """Format a metric value for safe checkpoint directory names."""
        return f"{value:.6f}".replace("-", "neg")

    @staticmethod
    def _format_diag_value(value: float) -> str:
        """Format a diagnostic scalar for compact single-line logs."""
        if np.isnan(value):
            return "nan"
        if np.isposinf(value):
            return "inf"
        if np.isneginf(value):
            return "-inf"
        return f"{value:.6f}"

    @staticmethod
    def _safe_mean(arr: np.ndarray) -> float:
        if arr.size == 0:
            return float('nan')
        return float(np.mean(arr))

    @staticmethod
    def _safe_std(arr: np.ndarray) -> float:
        if arr.size == 0:
            return float('nan')
        return float(np.std(arr))

    @staticmethod
    def _safe_quantiles(arr: np.ndarray) -> Tuple[float, float, float]:
        if arr.size == 0:
            return float('nan'), float('nan'), float('nan')
        q01, q50, q99 = np.quantile(arr, [0.01, 0.5, 0.99])
        return float(q01), float(q50), float(q99)

    def _build_eval_diagnostics_log(
        self,
        epoch: int,
        labels_np: np.ndarray,
        logits_np: np.ndarray,
        probs_np: np.ndarray,
        split_name: str = 'valid',
    ) -> str:
        """Build a compact validation diagnostics log line."""
        n = int(probs_np.size)
        if n == 0:
            return f"VALID_DIAGNOSTICS split={split_name} epoch={epoch} n=0"

        labels_float = labels_np.astype(np.float32, copy=False)
        pos_mask = labels_np == 1
        neg_mask = labels_np == 0

        pred_p01, pred_p50, pred_p99 = self._safe_quantiles(probs_np)
        logit_p01, logit_p50, logit_p99 = self._safe_quantiles(logits_np)
        pred_pos_mean = self._safe_mean(probs_np[pos_mask])
        pred_neg_mean = self._safe_mean(probs_np[neg_mask])
        logit_pos_mean = self._safe_mean(logits_np[pos_mask])
        logit_neg_mean = self._safe_mean(logits_np[neg_mask])
        pred_margin = pred_pos_mean - pred_neg_mean
        logit_margin = logit_pos_mean - logit_neg_mean
        brier_score = float(np.mean((probs_np - labels_float) ** 2))

        fmt = self._format_diag_value
        return (
            "VALID_DIAGNOSTICS"
            f" split={split_name}"
            f" epoch={epoch}"
            f" n={n}"
            f" pos={int(pos_mask.sum())}"
            f" neg={int(neg_mask.sum())}"
            f" label_rate={fmt(self._safe_mean(labels_float))}"
            f" pred_mean={fmt(self._safe_mean(probs_np))}"
            f" pred_std={fmt(self._safe_std(probs_np))}"
            f" pred_p01={fmt(pred_p01)}"
            f" pred_p50={fmt(pred_p50)}"
            f" pred_p99={fmt(pred_p99)}"
            f" logit_mean={fmt(self._safe_mean(logits_np))}"
            f" logit_std={fmt(self._safe_std(logits_np))}"
            f" logit_p01={fmt(logit_p01)}"
            f" logit_p50={fmt(logit_p50)}"
            f" logit_p99={fmt(logit_p99)}"
            f" pred_pos_mean={fmt(pred_pos_mean)}"
            f" pred_neg_mean={fmt(pred_neg_mean)}"
            f" pred_margin={fmt(pred_margin)}"
            f" logit_pos_mean={fmt(logit_pos_mean)}"
            f" logit_neg_mean={fmt(logit_neg_mean)}"
            f" logit_margin={fmt(logit_margin)}"
            f" brier={fmt(brier_score)}"
        )

    def _build_step_dir_name(
        self,
        global_step: int,
        is_best: bool = False,
        eval_index: Optional[int] = None,
        val_auc: Optional[float] = None,
        val_logloss: Optional[float] = None,
    ) -> str:
        """Build a checkpoint sub-directory name such as
        ``global_step2500.eval0001.layer=2.head=4.hidden=64.auc=0.860000``.
        """
        parts = [f"global_step{global_step}"]
        if eval_index is not None:
            parts.append(f"eval{eval_index:04d}")
        for key in ("layer", "head", "hidden"):
            if key in self.ckpt_params:
                parts.append(f"{key}={self.ckpt_params[key]}")
        if val_auc is not None:
            parts.append(f"auc={self._format_metric(val_auc)}")
        if val_logloss is not None:
            parts.append(f"logloss={self._format_metric(val_logloss)}")
        name = ".".join(parts)
        if is_best:
            name += ".best_model"
        return name

    def _write_sidecar_files(self, ckpt_dir: str) -> None:
        """Write sidecar files next to a ``model.pt``.

        Currently persists sidecar files next to the weights, all overwritten
        on every call:

        - ``schema.json`` (copied from ``self.schema_path``): feature layout
          metadata needed to rebuild the Parquet dataset.
        - ``ns_groups.json`` (copied from ``self.ns_groups_path`` when set
          and the file exists): NS-token grouping used to construct the
          tokenizer. Making a per-ckpt copy lets evaluation environments
          consume the checkpoint without having to ship the original
          project-level ``ns_groups.json``.
        - ``train_config.json`` (serialized from ``self.train_config``):
          full set of training-time hyperparameters. When ``ns_groups.json``
          is copied into ``ckpt_dir``, the ``ns_groups_json`` field is
          rewritten to the bare filename so that ``infer.py`` resolves it
          against ``ckpt_dir`` rather than the original absolute path on
          the training machine.
        - custom time-bucket JSON (copied from ``time_bucket_boundaries_json``
          when it is a non-empty path): domain-specific time bucket boundaries
          used by both train and infer. Empty string means the built-in
          hardcoded boundaries are used and no external file is needed.
        """
        os.makedirs(ckpt_dir, exist_ok=True)
        if self.schema_path and os.path.exists(self.schema_path):
            shutil.copy2(self.schema_path, ckpt_dir)

        ns_groups_copied = False
        if self.ns_groups_path and os.path.exists(self.ns_groups_path):
            shutil.copy2(self.ns_groups_path, ckpt_dir)
            ns_groups_copied = True

        time_bucket_json_copied_path: Optional[str] = None
        if self.train_config:
            time_bucket_json_path = self.train_config.get(
                'time_bucket_boundaries_json')
            if time_bucket_json_path:
                if not os.path.exists(time_bucket_json_path):
                    raise FileNotFoundError(
                        "time_bucket_boundaries_json was set during training "
                        f"but the file does not exist: {time_bucket_json_path}"
                    )
                shutil.copy2(time_bucket_json_path, ckpt_dir)
                time_bucket_json_copied_path = os.path.basename(
                    time_bucket_json_path)

        if self.train_config:
            import json
            cfg_to_dump = dict(self.train_config)
            if ns_groups_copied:
                # Override the stored path to a filename relative to ckpt_dir;
                # infer.py already falls back to `<ckpt_dir>/<basename>` when
                # the recorded path is not absolute, which keeps the ckpt
                # portable across hosts.
                cfg_to_dump['ns_groups_json'] = os.path.basename(
                    self.ns_groups_path)
            if time_bucket_json_copied_path is not None:
                cfg_to_dump['time_bucket_boundaries_json'] = (
                    time_bucket_json_copied_path)
            with open(os.path.join(ckpt_dir, 'train_config.json'), 'w') as f:
                json.dump(cfg_to_dump, f, indent=2)

    @staticmethod
    def _write_metrics_file(ckpt_dir: str, metrics: Dict[str, Any]) -> None:
        """Write eval metrics next to checkpoint weights."""
        import json
        os.makedirs(ckpt_dir, exist_ok=True)
        with open(os.path.join(ckpt_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)

    def _save_step_checkpoint(
        self,
        global_step: int,
        is_best: bool = False,
        skip_model_file: bool = False,
        eval_index: Optional[int] = None,
        val_auc: Optional[float] = None,
        val_logloss: Optional[float] = None,
        valid_metrics: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Save ``model.pt`` plus sidecar files under a ``global_step`` sub-dir.

        Args:
            global_step: current global step used to name the directory.
            is_best: whether this is a new-best checkpoint.
            skip_model_file: if True, skip writing ``model.pt`` (because the
                caller, e.g. EarlyStopping, has already persisted it to the
                same path). Sidecar files are still (re)written.

        Returns:
            The absolute path of the checkpoint directory.
        """
        dir_name = self._build_step_dir_name(
            global_step,
            is_best=is_best,
            eval_index=eval_index,
            val_auc=val_auc,
            val_logloss=val_logloss,
        )
        ckpt_dir = os.path.join(self.save_dir, dir_name)
        if os.path.exists(ckpt_dir) and not is_best:
            raise FileExistsError(
                f"Refusing to overwrite existing eval checkpoint directory: {ckpt_dir}"
            )
        os.makedirs(ckpt_dir, exist_ok=True)
        if not skip_model_file:
            torch.save(self.raw_model.state_dict(), os.path.join(ckpt_dir, "model.pt"))
        self._write_sidecar_files(ckpt_dir)
        self._write_metrics_file(ckpt_dir, {
            "eval_index": eval_index,
            "global_step": global_step,
            "val_AUC": val_auc,
            "val_logloss": val_logloss,
            "valid_metrics": valid_metrics,
            "is_best": is_best,
        })
        logging.info(f"Saved checkpoint to {ckpt_dir}/model.pt")
        return ckpt_dir

    def _batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move all tensors in ``batch`` to ``self.device`` (``non_blocking=True``,
        to cooperate with ``pin_memory``). Non-tensor values pass through.
        """
        device_batch: Dict[str, Any] = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                device_batch[k] = v.to(self.device, non_blocking=True)
            else:
                device_batch[k] = v
        return device_batch

    @staticmethod
    def _optimizer_lr(optimizer: Optional[torch.optim.Optimizer]) -> Optional[float]:
        if optimizer is None or not optimizer.param_groups:
            return None
        return float(optimizer.param_groups[0].get('lr', 0.0))

    def _write_lr_scalars(self, total_step: int) -> None:
        if not self.writer:
            return
        dense_lr = self._optimizer_lr(self.dense_optimizer)
        if dense_lr is not None:
            self.writer.add_scalar('LR/dense', dense_lr, total_step)
        sparse_lr = self._optimizer_lr(self.sparse_optimizer)
        if sparse_lr is not None:
            self.writer.add_scalar('LR/sparse', sparse_lr, total_step)

    @staticmethod
    def _metrics_payload(
        results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        return [
            {
                "name": r["name"],
                "AUC": r["auc"],
                "logloss": r["logloss"],
            }
            for r in results
        ]

    def _write_validation_scalars(
        self,
        results: List[Dict[str, Any]],
        total_step: int,
    ) -> None:
        if not self.writer:
            return
        if self.multi_valid_enabled:
            for idx, result in enumerate(results, start=1):
                self.writer.add_scalar(
                    f'AUC{idx}/valid', result['auc'], total_step)
                self.writer.add_scalar(
                    f'LogLoss{idx}/valid', result['logloss'], total_step)
                safe_name = result['name'].replace('/', '_')
                self.writer.add_scalar(
                    f'AUC/{safe_name}', result['auc'], total_step)
                self.writer.add_scalar(
                    f'LogLoss/{safe_name}', result['logloss'], total_step)
        else:
            result = results[0]
            self.writer.add_scalar('AUC/valid', result['auc'], total_step)
            self.writer.add_scalar('LogLoss/valid', result['logloss'], total_step)

    def _log_validation_results(
        self,
        prefix: str,
        results: List[Dict[str, Any]],
    ) -> None:
        if self.multi_valid_enabled:
            summary = " | ".join(
                f"{r['name']}: AUC: {r['auc']}, LogLoss: {r['logloss']}"
                for r in results
            )
            logging.info(f"{prefix} Validation | {summary}")
        else:
            result = results[0]
            logging.info(
                f"{prefix} Validation | AUC: {result['auc']}, "
                f"LogLoss: {result['logloss']}")
        for result in results:
            diagnostics = result.get('diagnostics')
            if diagnostics:
                logging.info(diagnostics)

    def evaluate_all(self, epoch: Optional[int] = None) -> List[Dict[str, Any]]:
        results = []
        for name, loader in self.valid_loaders:
            auc, logloss = self.evaluate(
                epoch=epoch,
                valid_loader=loader,
                valid_name=name,
            )
            results.append({
                "name": name,
                "auc": auc,
                "logloss": logloss,
                "diagnostics": self._last_eval_diagnostics_log,
            })
        return results

    def _handle_validation_result(
        self,
        total_step: int,
        results: List[Dict[str, Any]],
    ) -> None:
        """Save every eval checkpoint and keep a separate best checkpoint."""
        if not results:
            raise ValueError("validation results must not be empty")
        primary = results[0]
        val_auc = float(primary['auc'])
        val_logloss = float(primary['logloss'])
        valid_metrics = self._metrics_payload(results)
        self.eval_checkpoint_index += 1
        self._save_step_checkpoint(
            total_step,
            eval_index=self.eval_checkpoint_index,
            val_auc=val_auc,
            val_logloss=val_logloss,
            valid_metrics=valid_metrics,
        )

        old_best = self.early_stopping.best_score
        best_dir = os.path.join(self.save_dir, "best_model")
        self.early_stopping.checkpoint_path = os.path.join(best_dir, "model.pt")

        self.early_stopping(val_auc, self.raw_model, {
            "best_val_AUC": val_auc,
            "best_val_logloss": val_logloss,
            "best_global_step": total_step,
            "best_eval_index": self.eval_checkpoint_index,
            "best_valid_name": primary['name'],
            "valid_metrics": valid_metrics,
        })

        if self.early_stopping.best_score != old_best and os.path.exists(
            self.early_stopping.checkpoint_path
        ):
            self._write_sidecar_files(best_dir)
            self._write_metrics_file(best_dir, {
                "eval_index": self.eval_checkpoint_index,
                "global_step": total_step,
                "best_val_AUC": val_auc,
                "best_val_logloss": val_logloss,
                "best_valid_name": primary['name'],
                "valid_metrics": valid_metrics,
                "is_best": True,
            })
            logging.info(
                f"Updated best checkpoint at {best_dir}/model.pt "
                f"(eval={self.eval_checkpoint_index}, step={total_step}, "
                f"{primary['name']} AUC={val_auc}, LogLoss={val_logloss})"
            )

    def train(self) -> None:
        """Main training loop: iterates over epochs, performs step-level and
        epoch-level validation, triggers EarlyStopping and the periodic sparse
        re-initialization strategy.
        """
        print("Start training (PCVRHyFormer)")
        self.model.train()
        self.raw_model.train()
        total_step = 0

        for epoch in range(1, self.num_epochs + 1):
            train_pbar = tqdm(enumerate(self.train_loader),
                              dynamic_ncols=True,
                              disable=not self.show_progress_bar)
            loss_sum = 0.0
            steps_in_epoch = 0

            for step, batch in train_pbar:
                loss = self._train_step(batch)
                total_step += 1
                steps_in_epoch += 1
                loss_sum += loss

                if self.writer:
                    self.writer.add_scalar('Loss/train', loss, total_step)
                    self._write_lr_scalars(total_step)

                train_pbar.set_postfix({"loss": f"{loss:.4f}"})

                # Step-level validation (only when eval_every_n_steps > 0).
                if self.eval_every_n_steps > 0 and total_step % self.eval_every_n_steps == 0:
                    logging.info(f"Evaluating at step {total_step}")
                    results = self.evaluate_all(epoch=epoch)
                    self.model.train()
                    self.raw_model.train()
                    torch.cuda.empty_cache()

                    self._log_validation_results(f"Step {total_step}", results)
                    self._write_validation_scalars(results, total_step)

                    self._handle_validation_result(total_step, results)

                    if self.early_stopping.early_stop:
                        logging.info(f"Early stopping at step {total_step}")
                        return

            if steps_in_epoch == 0:
                raise RuntimeError(
                    f"train_loader yielded no batches in epoch {epoch}; "
                    "check data_dir, split settings, and timestamp filters"
                )
            logging.info(
                f"Epoch {epoch}, Average Loss: {loss_sum / steps_in_epoch}")

            results = self.evaluate_all(epoch=epoch)
            self.model.train()
            self.raw_model.train()
            torch.cuda.empty_cache()

            self._log_validation_results(f"Epoch {epoch}", results)
            self._write_validation_scalars(results, total_step)

            self._handle_validation_result(total_step, results)

            if self.early_stopping.early_stop:
                logging.info(f"Early stopping at epoch {epoch}")
                break

            # After the configured epoch, reinitialize high-cardinality sparse
            # params (Embeddings) as a form of cold restart to reduce overfit.
            # Reference: KuaiShou Tech., "MultiEpoch: Reusing Training Data
            # for Click-Through Rate Prediction",
            # https://arxiv.org/pdf/2305.19531
            if epoch >= self.reinit_sparse_after_epoch and self.sparse_optimizer is not None:
                # Snapshot Adagrad state per parameter via data_ptr, so state
                # of low-cardinality embeddings can be preserved across rebuild.
                old_state: Dict[int, Any] = {}
                for group in self.sparse_optimizer.param_groups:
                    for p in group['params']:
                        if p.data_ptr() in self.sparse_optimizer.state:
                            old_state[p.data_ptr()] = self.sparse_optimizer.state[p]

                reinit_ptrs = self.raw_model.reinit_high_cardinality_params(self.reinit_cardinality_threshold)
                sparse_params = self.raw_model.get_sparse_params()
                self.sparse_optimizer = torch.optim.Adagrad(
                    sparse_params, lr=self.sparse_lr, weight_decay=self.sparse_weight_decay
                )
                # Restore optimizer state for low-cardinality embeddings only.
                restored = 0
                for p in sparse_params:
                    if p.data_ptr() not in reinit_ptrs and p.data_ptr() in old_state:
                        self.sparse_optimizer.state[p] = old_state[p.data_ptr()]
                        restored += 1
                logging.info(f"Rebuilt Adagrad optimizer after epoch {epoch}, "
                             f"restored optimizer state for {restored} low-cardinality params")

    def _make_model_input(self, device_batch: Dict[str, Any]) -> ModelInput:
        """Construct a ``ModelInput`` NamedTuple from a device_batch dict."""
        seq_domains = device_batch['_seq_domains']
        seq_data: Dict[str, torch.Tensor] = {}
        seq_lens: Dict[str, torch.Tensor] = {}
        seq_time_buckets: Dict[str, torch.Tensor] = {}
        seq_recency_stats: Dict[str, torch.Tensor] = {}
        for domain in seq_domains:
            seq_data[domain] = device_batch[domain]
            seq_lens[domain] = device_batch[f'{domain}_len']
            B = device_batch[domain].shape[0]
            L = device_batch[domain].shape[2]
            seq_time_buckets[domain] = device_batch.get(
                f'{domain}_time_bucket',
                torch.zeros(B, L, dtype=torch.long, device=self.device))
            if f'{domain}_recency_stats' in device_batch:
                seq_recency_stats[domain] = device_batch[f'{domain}_recency_stats']
        return ModelInput(
            user_int_feats=device_batch['user_int_feats'],
            item_int_feats=device_batch['item_int_feats'],
            user_dense_feats=device_batch['user_dense_feats'],
            item_dense_feats=device_batch['item_dense_feats'],
            timestamp=device_batch['timestamp'],
            seq_data=seq_data,
            seq_lens=seq_lens,
            seq_time_buckets=seq_time_buckets,
            seq_recency_stats=seq_recency_stats,
        )

    def _train_step(self, batch: Dict[str, Any]) -> float:
        """Run a single training step and return the scalar loss value."""
        device_batch = self._batch_to_device(batch)
        label = device_batch['label'].float()
        sample_weight = device_batch.get('sample_weight')
        if sample_weight is not None:
            sample_weight = sample_weight.float()
            if sample_weight.shape != label.shape:
                raise RuntimeError(
                    "sample_weight shape must match label shape, got "
                    f"{tuple(sample_weight.shape)} vs {tuple(label.shape)}")
            if not torch.isfinite(sample_weight).all():
                raise RuntimeError("sample_weight contains non-finite values")
            if (sample_weight <= 0).any():
                raise RuntimeError("sample_weight must be strictly positive")

        self.dense_optimizer.zero_grad(set_to_none=True)
        if self.sparse_optimizer is not None:
            self.sparse_optimizer.zero_grad(set_to_none=True)

        model_input = self._make_model_input(device_batch)
        with self._autocast_context():
            logits = self.model(model_input)  # (B, 1)
            logits = logits.squeeze(-1)  # (B,)

            if self.loss_type == 'focal':
                raw_loss = sigmoid_focal_loss(
                    logits,
                    label,
                    alpha=self.focal_alpha,
                    gamma=self.focal_gamma,
                    reduction='none',
                )
            else:
                raw_loss = F.binary_cross_entropy_with_logits(
                    logits,
                    label,
                    reduction='none',
                )
            if sample_weight is not None:
                denom = sample_weight.sum()
                if denom <= 0:
                    raise RuntimeError(
                        "sample_weight sum must be positive for weighted loss")
                loss = (raw_loss * sample_weight).sum() / denom
            else:
                loss = raw_loss.mean()
        loss.backward()
        # foreach=False: avoids a PyTorch _foreach_norm CUDA kernel bug observed
        # with certain tensor shapes in this project.
        torch.nn.utils.clip_grad_norm_(self.raw_model.parameters(), max_norm=1.0, foreach=False)

        self.dense_optimizer.step()
        if self.sparse_optimizer is not None:
            self.sparse_optimizer.step()

        return loss.item()

    def evaluate(
        self,
        epoch: Optional[int] = None,
        valid_loader: Optional[DataLoader] = None,
        valid_name: str = 'valid',
    ) -> Tuple[float, float]:
        """Run validation over one loader and return ``(AUC, logloss)``.

        NaN predictions (which can arise from exploding gradients) are filtered
        out before computing both metrics.
        """
        if valid_loader is None:
            valid_loader = self.valid_loader
        print(f"Start Evaluation (PCVRHyFormer) - {valid_name}")
        self.model.eval()
        self.raw_model.eval()
        if not epoch:
            epoch = -1

        pbar = tqdm(enumerate(valid_loader),
                    disable=not self.show_progress_bar)

        all_logits_list = []
        all_labels_list = []

        with torch.inference_mode():
            for step, batch in pbar:
                logits, labels = self._evaluate_step(batch)
                all_logits_list.append(logits.detach().cpu())
                all_labels_list.append(labels.detach().cpu())

        if not all_logits_list:
            raise RuntimeError(
                f"validation loader {valid_name} yielded no batches; "
                "check timestamp windows and dataset filters")

        # Autocast may produce bf16 logits; CPU numpy and sklearn metrics
        # require fp32/float64-compatible arrays.
        all_logits = torch.cat(all_logits_list, dim=0).float()
        all_labels = torch.cat(all_labels_list, dim=0).long()

        # Binary AUC via sklearn.
        probs = torch.sigmoid(all_logits).numpy()
        logits_np = all_logits.numpy()
        labels_np = all_labels.numpy()

        # Filter NaN predictions (may appear if gradients explode).
        nan_mask = np.isnan(probs)
        if nan_mask.any():
            n_nan = int(nan_mask.sum())
            logging.warning(
                f"[Evaluate:{valid_name}] {n_nan}/{len(probs)} predictions "
                "are NaN, filtering them out")
            valid_mask = ~nan_mask
            probs = probs[valid_mask]
            labels_np = labels_np[valid_mask]
            logits_np = logits_np[valid_mask]

        if len(probs) == 0 or len(np.unique(labels_np)) < 2:
            auc = 0.0
        else:
            auc = float(roc_auc_score(labels_np, probs))

        self._last_eval_diagnostics_log = self._build_eval_diagnostics_log(
            epoch, labels_np, logits_np, probs, split_name=valid_name)

        # Binary logloss (same NaN filtering).
        valid_logits = all_logits[~torch.isnan(all_logits)]
        valid_labels = all_labels[~torch.isnan(all_logits)]
        if len(valid_logits) > 0:
            logloss = F.binary_cross_entropy_with_logits(valid_logits, valid_labels.float()).item()
        else:
            logloss = float('inf')

        return auc, logloss

    def _evaluate_step(
        self, batch: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run a single validation step and return ``(logits, labels)``."""
        device_batch = self._batch_to_device(batch)
        label = device_batch['label']

        model_input = self._make_model_input(device_batch)
        with self._autocast_context():
            logits, _ = self.raw_model.predict(model_input)  # (B, 1), (B, D)
        logits = logits.squeeze(-1)  # (B,)

        return logits, label
