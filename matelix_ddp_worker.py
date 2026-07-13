#!/usr/bin/env python3
# matelix_ddp_worker.py
# Copyright 2026 TMP-SYSTEM-SERVICE GmbH
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gc
import hashlib
import json
import logging
import math
import multiprocessing as mp
import os
import pickle
import random
import re
import shutil
import signal
import sys
import time
import traceback
import threading
import unicodedata
from contextlib import nullcontext
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# Rueckwaertskompatibel: einige Umgebungen nutzen noch den alten (falschen) Schluessel.
os.environ.setdefault("PYTORCH_ALLOC_CONF", os.environ["PYTORCH_CUDA_ALLOC_CONF"])

if os.environ.get("MATELIX_NCCL_BLOCKING_WAIT", "0") == "1":
    os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from matelix_ngram_pipeline import (
    NgramConfig,
    build_or_load_ngram_state,
    ngram_summary_text,
)

try:
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    _PEFT_AVAILABLE = True
except Exception:
    LoraConfig = PeftModel = TaskType = get_peft_model = None
    _PEFT_AVAILABLE = False

try:
    from torch.amp import GradScaler
    _NEW_SCALER = True
except Exception:
    from torch.cuda.amp import GradScaler
    _NEW_SCALER = False

csv.field_size_limit(1024 * 1024 * 128)
LOGGER = logging.getLogger("matelix_ddp_worker")


@dataclass
class TrainConfig:
    model_dir: str
    csv_path: str

    save_dir: Optional[str] = None
    output_dir: Optional[str] = None

    device: str = "cuda"
    template_mode: str = "chat"
    column_name: str = "text"
    mixed_training: bool = False
    mixed_text_column: str = "Text"
    training_phase: str = "custom"
    text_token_weight: float = 0.7
    dialog_token_weight: float = 0.3
    max_mixture_oversample: float = 4.0

    chunk_long_texts: bool = True
    text_chunk_overlap: int = 128
    text_chunk_min_tokens: int = 32
    append_eos_to_text: bool = True
    pack_short_texts: bool = False
    pack_target_length: int = 1024

    deduplicate_exact: bool = True
    near_duplicate_action: str = "warn"
    near_duplicate_threshold: float = 0.92
    near_duplicate_max_shingles: int = 512
    quality_filter_mode: str = "warn"
    quality_min_chars: int = 24

    tokenizer_dir: Optional[str] = None
    train_scratch_tokenizer: bool = False
    scratch_tokenizer_vocab_size: int = 32000

    learning_rate: float = 2e-4
    lr_schedule: str = "cosine"
    lr_decay_factor: float = 1.0
    warmup_steps: int = 0
    warmup_ratio: float = 0.0
    min_lr_ratio: float = 0.0

    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    num_train_epochs: float = 3.0
    max_steps: Optional[int] = None
    max_seq_length: int = 4096
    chunk_size: Optional[int] = None
    max_history_turns: Optional[int] = None

    sort_by_length: bool = True
    sort_by_similarity: bool = False
    fixed_padding: bool = False
    dynamic_token_batching: bool = True
    max_tokens_per_batch: int = 0
    max_samples_per_batch: int = 0
    shuffle_batches: bool = True
    pad_batches_for_ddp: bool = True
    token_normalized_loss: bool = True
    dataloader_num_workers: int = -1
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01

    precision_mode: str = "auto"
    gradient_checkpointing: bool = False
    skip_oom_microbatches: bool = True

    train_mode: str = "full"
    train_from_scratch: bool = False
    include_prompt_loss: bool = False
    scratch_hidden_size: Optional[int] = None
    scratch_num_hidden_layers: Optional[int] = None
    scratch_num_attention_heads: Optional[int] = None
    scratch_intermediate_size: Optional[int] = None
    scratch_num_key_value_heads: Optional[int] = None
    scratch_max_position_embeddings: Optional[int] = None
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: Optional[List[str]] = None
    merge_lora_on_save: bool = True
    neftune_noise_alpha: float = 0.0

    dataset_audit: bool = True
    dataset_audit_strict: bool = False

    log_every_steps: int = 10
    save_every_epoch: bool = True
    keep_last_k_checkpoints: int = 3
    resume: Optional[str] = None
    val_split: float = 0.05
    split_seed: int = 42
    validate_every_epoch: bool = True
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 0.0

    ddp_find_unused_parameters: bool = False
    ddp_static_graph: bool = False
    ddp_broadcast_buffers: bool = False
    ddp_timeout_minutes: int = 30

    seed: int = 42
    deterministic: bool = False
    allow_tf32: bool = True
    use_ngrams: bool = False
    force_template: bool = True
    prefetch_factor: int = 4
    persistent_workers: bool = True

    use_dataset_cache: bool = True
    rebuild_dataset_cache: bool = False
    dataset_cache_dir: Optional[str] = None
    dataset_num_workers: int = -1
    dataset_tokenize_batch_size: int = 32
    tokenized_shard_size: int = 5000

    ngram_max: int = 12
    ngram_top_k: int = 1500
    ngram_min_chars: int = 16
    ngram_min_words: int = 2
    ngram_max_samples: int = 4000
    ngram_budgeted: bool = True
    ngram_target_fit: float = 0.98
    ngram_eval_samples: int = 512
    ngram_add_batch: int = 64
    ngram_min_count: int = 2
    ngram_max_token_chars: int = 384
    ngram_max_tokens_per_text: int = 4096

    log_cuda_memory: bool = True
    cuda_memory_log_interval_steps: int = 25
    cuda_empty_cache_interval_steps: int = 0

    def normalize(self) -> None:
        if not self.output_dir:
            self.output_dir = self.save_dir
        if not self.chunk_size:
            self.chunk_size = self.max_seq_length
        self.max_seq_length = int(self.chunk_size or self.max_seq_length)
        self.per_device_train_batch_size = max(1, int(self.per_device_train_batch_size))
        self.gradient_accumulation_steps = max(1, int(self.gradient_accumulation_steps))
        self.max_grad_norm = float(self.max_grad_norm)
        self.learning_rate = float(self.learning_rate)
        self.weight_decay = float(self.weight_decay)
        self.num_train_epochs = float(self.num_train_epochs)
        self.lr_decay_factor = max(0.01, float(self.lr_decay_factor))
        self.warmup_steps = max(0, int(self.warmup_steps))
        self.warmup_ratio = max(0.0, float(self.warmup_ratio))
        self.min_lr_ratio = min(1.0, max(0.0, float(self.min_lr_ratio)))
        self.seed = int(self.seed)
        self.mixed_training = bool(self.mixed_training)
        self.mixed_text_column = str(self.mixed_text_column or "Text").strip() or "Text"
        self.training_phase = str(self.training_phase or "custom").strip().lower()
        if self.training_phase not in {"custom", "pretrain", "mixed", "sft"}:
            raise ValueError(
                "training_phase muss custom, pretrain, mixed oder sft sein"
            )
        if self.training_phase == "mixed":
            self.mixed_training = True
        if self.training_phase == "sft":
            self.include_prompt_loss = False
        elif self.training_phase == "pretrain" or (
            self.training_phase == "mixed" and self.train_from_scratch
        ):
            self.include_prompt_loss = True
        self.text_token_weight = max(0.0, float(self.text_token_weight))
        self.dialog_token_weight = max(0.0, float(self.dialog_token_weight))
        if self.text_token_weight + self.dialog_token_weight <= 0.0:
            raise ValueError("Mindestens ein Datengewicht muss größer als 0 sein")
        self.max_mixture_oversample = max(1.0, float(self.max_mixture_oversample))
        self.chunk_long_texts = bool(self.chunk_long_texts)
        self.text_chunk_overlap = max(0, int(self.text_chunk_overlap))
        self.text_chunk_min_tokens = max(1, int(self.text_chunk_min_tokens))
        self.append_eos_to_text = bool(self.append_eos_to_text)
        self.pack_short_texts = bool(self.pack_short_texts)
        self.pack_target_length = min(
            self.max_seq_length,
            max(64, int(self.pack_target_length or min(1024, self.max_seq_length))),
        )
        self.deduplicate_exact = bool(self.deduplicate_exact)
        self.near_duplicate_action = str(self.near_duplicate_action or "warn").strip().lower()
        if self.near_duplicate_action not in {"off", "warn", "exclude"}:
            raise ValueError("near_duplicate_action muss off, warn oder exclude sein")
        self.near_duplicate_threshold = min(0.999, max(0.5, float(self.near_duplicate_threshold)))
        self.near_duplicate_max_shingles = max(32, int(self.near_duplicate_max_shingles))
        self.quality_filter_mode = str(self.quality_filter_mode or "warn").strip().lower()
        if self.quality_filter_mode not in {"off", "warn", "exclude"}:
            raise ValueError("quality_filter_mode muss off, warn oder exclude sein")
        self.quality_min_chars = max(1, int(self.quality_min_chars))
        self.tokenizer_dir = str(self.tokenizer_dir).strip() if self.tokenizer_dir else None
        self.train_scratch_tokenizer = bool(self.train_scratch_tokenizer)
        self.scratch_tokenizer_vocab_size = max(256, int(self.scratch_tokenizer_vocab_size))
        if (
            self.train_from_scratch
            and self.training_phase in {"pretrain", "mixed"}
            and self.warmup_steps <= 0
            and self.warmup_ratio <= 0.0
        ):
            self.warmup_ratio = 0.02
        self.dynamic_token_batching = bool(self.dynamic_token_batching)
        self.max_tokens_per_batch = max(0, int(self.max_tokens_per_batch))
        self.max_samples_per_batch = max(0, int(self.max_samples_per_batch))
        self.shuffle_batches = bool(self.shuffle_batches)
        self.pad_batches_for_ddp = bool(self.pad_batches_for_ddp)
        self.token_normalized_loss = bool(self.token_normalized_loss)
        self.dataloader_num_workers = int(self.dataloader_num_workers)
        self.fixed_padding = bool(self.fixed_padding)
        self.ddp_timeout_minutes = int(self.ddp_timeout_minutes)
        self.ddp_find_unused_parameters = bool(self.ddp_find_unused_parameters)
        self.ddp_static_graph = bool(self.ddp_static_graph)
        self.ddp_broadcast_buffers = bool(self.ddp_broadcast_buffers)
        self.force_template = bool(self.force_template)
        self.train_from_scratch = bool(self.train_from_scratch)
        self.include_prompt_loss = bool(self.include_prompt_loss)
        self.lora_r = max(1, int(self.lora_r))
        self.lora_alpha = max(1, int(self.lora_alpha))
        self.lora_dropout = min(0.95, max(0.0, float(self.lora_dropout)))
        if self.lora_target_modules is not None:
            self.lora_target_modules = sorted({str(x).strip() for x in self.lora_target_modules if str(x).strip()}) or None
        self.log_every_steps = max(1, int(self.log_every_steps))
        self.save_every_epoch = bool(self.save_every_epoch)
        self.keep_last_k_checkpoints = max(1, int(self.keep_last_k_checkpoints))
        self.resume = str(self.resume).strip() if self.resume else None
        self.neftune_noise_alpha = max(0.0, float(self.neftune_noise_alpha))
        self.dataset_audit = bool(self.dataset_audit)
        self.dataset_audit_strict = bool(self.dataset_audit_strict)
        self.val_split = min(0.5, max(0.0, float(self.val_split)))
        self.split_seed = int(self.split_seed)
        self.validate_every_epoch = bool(self.validate_every_epoch)
        self.early_stopping_patience = max(0, int(self.early_stopping_patience))
        self.early_stopping_min_delta = max(0.0, float(self.early_stopping_min_delta))
        self.skip_oom_microbatches = bool(self.skip_oom_microbatches)
        if self.scratch_hidden_size is not None:
            self.scratch_hidden_size = max(1, int(self.scratch_hidden_size))
        if self.scratch_num_hidden_layers is not None:
            self.scratch_num_hidden_layers = max(1, int(self.scratch_num_hidden_layers))
        if self.scratch_num_attention_heads is not None:
            self.scratch_num_attention_heads = max(1, int(self.scratch_num_attention_heads))
        if self.scratch_intermediate_size is not None:
            self.scratch_intermediate_size = max(1, int(self.scratch_intermediate_size))
        if self.scratch_num_key_value_heads is not None:
            self.scratch_num_key_value_heads = max(1, int(self.scratch_num_key_value_heads))
        if self.scratch_max_position_embeddings is not None:
            self.scratch_max_position_embeddings = max(1, int(self.scratch_max_position_embeddings))
        self.deterministic = bool(self.deterministic)
        self.allow_tf32 = bool(self.allow_tf32)
        self.prefetch_factor = max(1, int(self.prefetch_factor))
        self.persistent_workers = bool(self.persistent_workers)
        self.use_dataset_cache = bool(self.use_dataset_cache)
        self.rebuild_dataset_cache = bool(self.rebuild_dataset_cache)
        self.dataset_cache_dir = (
            str(Path(self.dataset_cache_dir).expanduser())
            if self.dataset_cache_dir else None
        )
        self.dataset_num_workers = min(32, max(-1, int(self.dataset_num_workers)))
        self.dataset_tokenize_batch_size = min(
            1024, max(1, int(self.dataset_tokenize_batch_size))
        )
        self.tokenized_shard_size = max(100, int(self.tokenized_shard_size))
        self.ngram_max = max(2, int(self.ngram_max))
        self.ngram_top_k = max(1, int(self.ngram_top_k))
        self.ngram_min_chars = max(1, int(self.ngram_min_chars))
        self.ngram_min_words = max(1, int(self.ngram_min_words))
        self.ngram_max_samples = max(1, int(self.ngram_max_samples))
        self.ngram_target_fit = float(self.ngram_target_fit)
        self.ngram_eval_samples = max(1, int(self.ngram_eval_samples))
        self.ngram_add_batch = max(1, int(self.ngram_add_batch))
        self.ngram_min_count = max(1, int(self.ngram_min_count))
        self.ngram_max_token_chars = max(8, int(self.ngram_max_token_chars))
        self.ngram_max_tokens_per_text = max(32, int(self.ngram_max_tokens_per_text))
        self.log_cuda_memory = bool(self.log_cuda_memory)
        self.cuda_memory_log_interval_steps = max(1, int(self.cuda_memory_log_interval_steps))
        self.cuda_empty_cache_interval_steps = max(0, int(self.cuda_empty_cache_interval_steps))
        if self.max_history_turns is not None:
            self.max_history_turns = max(1, int(self.max_history_turns))


def _coerce_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(payload or {})

    if "save_dir" in payload and "output_dir" not in payload:
        payload["output_dir"] = payload["save_dir"]

    ignore_keys = {
        "nproc_per_node", "nnodes", "node_rank", "master_addr", "master_port",
        "world_size", "local_rank", "rank", "run_name", "experiment_name",
        "monitor_metric", "monitor_mode",
        "use_tensorboard", "val_csv",
        "compile_model", "compile_mode",
        "scheduler",
    }
    for k in list(payload.keys()):
        if k in ignore_keys:
            payload.pop(k, None)

    valid = {f.name for f in fields(TrainConfig)}
    return {k: v for k, v in payload.items() if k in valid}


def load_cfg(path: str) -> TrainConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    payload = _coerce_payload(payload)
    cfg = TrainConfig(**payload)
    cfg.normalize()
    return cfg


@dataclass
class DistContext:
    rank: int
    local_rank: int
    world_size: int
    is_distributed: bool
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


class ShutdownFlag:
    def __init__(self) -> None:
        self.stop = False
        self.reason = ""

    def request(self, reason: str) -> None:
        self.stop = True
        self.reason = reason


SHUTDOWN = ShutdownFlag()


def register_signal_handlers() -> None:
    def _handler(signum, _frame):
        try:
            name = signal.Signals(signum).name
        except Exception:
            name = str(signum)
        SHUTDOWN.request(name)

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        try:
            torch.use_deterministic_algorithms(False)
        except Exception:
            pass


def init_dist(cfg: TrainConfig) -> DistContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_distributed = world_size > 1

    device_name = (cfg.device or "").lower().strip()
    if torch.cuda.is_available() and device_name in {"cuda", "auto", ""}:
        device = torch.device("cuda", local_rank if is_distributed else 0)
        torch.cuda.set_device(device)
    elif device_name == "cpu":
        device = torch.device("cpu")
    elif device_name == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda", local_rank if is_distributed else 0)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    ctx = DistContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        is_distributed=is_distributed,
        device=device,
    )

    if is_distributed:
        backend = "nccl" if device.type == "cuda" else "gloo"
        from datetime import timedelta
        init_kwargs = dict(
            backend=backend,
            init_method="env://",
            timeout=timedelta(minutes=cfg.ddp_timeout_minutes),
        )

        if device.type == "cuda":
            try:
                dist.init_process_group(**init_kwargs, device_id=device)
            except TypeError:
                dist.init_process_group(**init_kwargs)
        else:
            dist.init_process_group(**init_kwargs)

    return ctx


def cleanup_dist() -> None:
    if dist.is_available() and dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception:
            pass


def barrier(ctx: DistContext) -> None:
    if ctx.is_distributed and dist.is_initialized():
        if ctx.device.type == "cuda":
            try:
                dist.barrier(device_ids=[ctx.local_rank])
                return
            except Exception:
                pass
        dist.barrier()


def all_reduce_mean(value: float, ctx: DistContext) -> float:
    if not ctx.is_distributed:
        return float(value)
    t = torch.tensor(float(value), device=ctx.device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= ctx.world_size
    return float(t.item())


def sync_stop(local_stop: bool, ctx: DistContext) -> bool:
    if not ctx.is_distributed:
        return local_stop
    t = torch.tensor([1 if local_stop else 0], device=ctx.device, dtype=torch.int32)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return bool(t.item())


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


class JsonStatusWriter:
    def __init__(self, path: Path, ctx: DistContext):
        self.path = path
        self.ctx = ctx

    def write(self, payload: Dict[str, Any]) -> None:
        if not self.ctx.is_main:
            return
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)


class JsonPreviewWriter:
    def __init__(self, path: Path, ctx: DistContext):
        self.path = path
        self.ctx = ctx

    def write(self, preview: str, preview_full: Optional[str] = None) -> None:
        if not self.ctx.is_main:
            return
        payload = {
            "preview": (preview or "")[:4000],
            "preview_full": (preview_full if preview_full is not None else preview or "")[:20000],
        }
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)


def setup_logging(log_path: Path, ctx: DistContext) -> None:
    LOGGER.handlers.clear()
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | rank=%(rank)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    class RankFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            record.rank = ctx.rank
            return True

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(formatter)
    fh.addFilter(RankFilter())
    fh.setLevel(logging.INFO)
    LOGGER.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    sh.addFilter(RankFilter())
    sh.setLevel(logging.INFO if ctx.is_main else logging.ERROR)
    LOGGER.addHandler(sh)


def normalize_id(val: Any) -> str:
    if val is None:
        return ""
    s = str(val).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


def get_chat_template(template_mode: str) -> str:
    mode = (template_mode or "chat").strip().lower()

    if mode == "plain":
        return """{% for message in messages %}
{{ message.content }}
{% endfor %}"""

    if mode == "instruct":
        return """{{ bos_token }}
{% for message in messages %}
{% if message.role == 'system' %}[SYSTEM]
{{ message.content }}
{% elif message.role == 'user' %}[USER]
{{ message.content }}
{% elif message.role == 'assistant' %}[ASSISTANT]
{{ message.content }}
{% endif %}
{% endfor %}
{% if add_generation_prompt %}[ASSISTANT]
{% endif %}"""

    if mode in {"chat", "dialogplus"}:
        return """{% for message in messages %}{% if loop.index0 != 0 and message['role'] == 'system' %}{{ raise_exception('Conversation roles must alternate system(optional)/user/assistant/user/assistant/...') }}{% elif messages[0]['role'] == 'system' and ((message['role'] == 'user' and (loop.index0 % 2 == 0)) or (message['role'] == 'assistant' and (loop.index0 % 2 == 1))) %}{{ raise_exception('Conversation roles must alternate system(optional)/user/assistant/user/assistant/...') }}{% elif messages[0]['role'] != 'system' and ((message['role'] == 'user' and (loop.index0 % 2 != 0)) or (message['role'] == 'assistant' and (loop.index0 % 2 != 1))) %}{{ raise_exception('Conversation roles must alternate system(optional)/user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '<|Benutzer|>' + message['content'].strip() + eos_token }}{% elif message['role'] == 'system' %}{{ '<|System|>' + message['content'].strip() + eos_token }}{% elif message['role'] == 'assistant' %}{{ '<|Assistentin|>' + message['content'].strip() + eos_token }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|Assistentin|>' }}{% endif %}"""

    return """{% for message in messages %}{% if loop.index0 != 0 and message['role'] == 'system' %}{{ raise_exception('Conversation roles must alternate system(optional)/user/assistant/user/assistant/...') }}{% elif messages[0]['role'] == 'system' and ((message['role'] == 'user' and (loop.index0 % 2 == 0)) or (message['role'] == 'assistant' and (loop.index0 % 2 == 1))) %}{{ raise_exception('Conversation roles must alternate system(optional)/user/assistant/user/assistant/...') }}{% elif messages[0]['role'] != 'system' and ((message['role'] == 'user' and (loop.index0 % 2 != 0)) or (message['role'] == 'assistant' and (loop.index0 % 2 != 1))) %}{{ raise_exception('Conversation roles must alternate system(optional)/user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '<|Benutzer|>' + message['content'].strip() + eos_token }}{% elif message['role'] == 'system' %}{{ '<|System|>' + message['content'].strip() + eos_token }}{% elif message['role'] == 'assistant' %}{{ '<|Assistentin|>' + message['content'].strip() + eos_token }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|Assistentin|>' }}{% endif %}"""


def prepare_tokenizer(tokenizer, template_mode: str = "chat", force_template: bool = True) -> bool:
    need_resize = False
    if tokenizer.eos_token_id is None:
        tokenizer.add_special_tokens({"eos_token": "<|eos|>"})
        need_resize = True
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        need_resize = True

    added = tokenizer.add_tokens(["<|System|>", "<|Benutzer|>", "<|Assistentin|>"], special_tokens=False)
    if added > 0:
        need_resize = True

    tokenizer.padding_side = "left"

    if force_template or not getattr(tokenizer, "chat_template", None):
        tokenizer.chat_template = get_chat_template(template_mode)

    return need_resize


@dataclass
class StructuredTurn:
    role: str
    content: str


@dataclass
class StructuredChatSample:
    system: str
    turns: List[StructuredTurn]
    target_answer: str
    split_key: str


@dataclass
class PlainTextSample:
    text: str
    split_key: str


def column_iter(csv_path: str, column_name: str) -> Iterator[str]:
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            txt = (row.get(column_name) or "").strip()
            if txt:
                yield txt


def _load_thread_rows(csv_path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows: List[Dict[str, Any]] = []
        for idx, row in enumerate(reader):
            row = dict(row)
            row["_rowidx"] = idx
            row["id"] = normalize_id(row.get("id", ""))
            row["parent_id"] = normalize_id(row.get("parent_id", ""))
            rows.append(row)
    id2row = {r["id"]: r for r in rows if r.get("id")}
    return rows, id2row


def _resolve_root_and_depth(
    rid: str, id2row: Dict[str, Dict[str, Any]],
) -> Tuple[str, int]:
    cur = id2row.get(rid)
    if not cur:
        return "", 0
    depth = 0
    seen: set[str] = set()
    while cur:
        current_id = normalize_id(cur.get("id"))
        if not current_id:
            return "", depth
        if current_id in seen:
            return min(seen), depth
        seen.add(current_id)
        parent_id = normalize_id(cur.get("parent_id"))
        if not parent_id or parent_id not in id2row:
            return current_id, depth
        cur = id2row[parent_id]
        depth += 1
    return rid, depth


def _build_root_depth_lookup(
    rows: List[Dict[str, Any]], id2row: Dict[str, Dict[str, Any]],
) -> Dict[str, Tuple[str, int]]:
    """Resolve every thread root once with path compression."""
    resolved: Dict[str, Tuple[str, int]] = {}
    for row in rows:
        start = normalize_id(row.get("id"))
        if not start or start in resolved:
            continue
        path: List[str] = []
        positions: Dict[str, int] = {}
        current = start
        while current and current in id2row and current not in resolved:
            if current in positions:
                cycle_nodes = path[positions[current]:]
                cycle_root = min(cycle_nodes) if cycle_nodes else current
                for node in cycle_nodes:
                    resolved[node] = (cycle_root, 0)
                break
            positions[current] = len(path)
            path.append(current)
            parent = normalize_id(id2row[current].get("parent_id"))
            if not parent or parent not in id2row:
                resolved[current] = (current, 0)
                break
            current = parent

        for node in reversed(path):
            if node in resolved:
                continue
            parent = normalize_id(id2row[node].get("parent_id"))
            parent_root, parent_depth = resolved.get(parent, (node, -1))
            resolved[node] = (parent_root, parent_depth + 1)
    return resolved


def _iter_candidate_chains_from_rows(
    rows: List[Dict[str, Any]],
    id2row: Dict[str, Dict[str, Any]],
    root_depth_lookup: Optional[Dict[str, Tuple[str, int]]] = None,
) -> Iterator[Tuple[str, List[Dict[str, Any]]]]:
    candidates = [r for r in rows if (r.get("Assistentin") or "").strip() and r.get("id")]
    if not candidates:
        return

    root_depth_lookup = root_depth_lookup or _build_root_depth_lookup(rows, id2row)

    threads: Dict[str, List[Tuple[int, int, Dict[str, Any]]]] = {}
    for r in candidates:
        root_id, depth = root_depth_lookup.get(r.get("id", ""), ("", 0))
        threads.setdefault(root_id, []).append((depth, int(r["_rowidx"]), r))

    order = list(threads.keys())

    for root_id in order:
        items = sorted(threads[root_id], key=lambda x: (x[0], x[1]))
        for _, _, target in items:
            chain: List[Dict[str, Any]] = []
            cur = target
            seen = set()
            while cur.get("id") and cur["id"] not in seen:
                seen.add(cur["id"])
                chain.append(cur)
                pid = cur.get("parent_id", "")
                if pid and pid in id2row:
                    cur = id2row[pid]
                else:
                    break
            chain.reverse()
            if chain:
                yield root_id, chain


def _iter_candidate_chains(csv_path: str) -> Iterator[Tuple[str, List[Dict[str, Any]]]]:
    rows, id2row = _load_thread_rows(csv_path)
    yield from _iter_candidate_chains_from_rows(rows, id2row)


def _chat_structured_iter_from_rows(
    rows: List[Dict[str, Any]],
    id2row: Dict[str, Dict[str, Any]],
    root_depth_lookup: Optional[Dict[str, Tuple[str, int]]] = None,
) -> Iterator[StructuredChatSample]:
    for root_id, chain in _iter_candidate_chains_from_rows(
        rows, id2row, root_depth_lookup,
    ):
        target_idx = len(chain) - 1
        answer = (chain[target_idx].get("Assistentin") or "").strip()
        if not answer:
            continue

        system_text = (chain[0].get("system") or "").strip()
        turns: List[StructuredTurn] = []

        for j in range(target_idx + 1):
            turn = chain[j]
            user = (turn.get("Benutzer") or "").strip()
            ctx = (turn.get("Kontext") or "").strip()
            asst = (turn.get("Assistentin") or "").strip()

            if user:
                turns.append(
                    StructuredTurn(
                        role="user",
                        content=f"{ctx}\n{user}".strip() if ctx else user,
                    )
                )

            if j < target_idx and asst:
                turns.append(StructuredTurn(role="assistant", content=asst))

        yield StructuredChatSample(
            system=system_text,
            turns=turns,
            target_answer=answer,
            split_key=f"thread:{root_id}",
        )


def chat_structured_iter(csv_path: str) -> Iterator[StructuredChatSample]:
    rows, id2row = _load_thread_rows(csv_path)
    yield from _chat_structured_iter_from_rows(rows, id2row)


def dialogplus_structured_iter(csv_path: str) -> Iterator[StructuredChatSample]:
    for item in chat_structured_iter(csv_path):
        yield item


def _build_mixed_split_groups_from_rows(
    rows: List[Dict[str, Any]],
    id2row: Dict[str, Dict[str, Any]],
    column_name: str,
    root_depth_lookup: Optional[Dict[str, Tuple[str, int]]] = None,
) -> Dict[str, str]:
    root_depth_lookup = root_depth_lookup or _build_root_depth_lookup(rows, id2row)
    parents: Dict[str, str] = {}

    def find(node: str) -> str:
        parents.setdefault(node, node)
        while parents[node] != node:
            parents[node] = parents[parents[node]]
            node = parents[node]
        return node

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[max(left_root, right_root)] = min(left_root, right_root)

    for row in rows:
        text = (row.get(column_name) or "").strip()
        if not text:
            continue
        text_node = "text:" + hashlib.sha256(text.encode("utf-8")).hexdigest()
        find(text_node)
        rid = normalize_id(row.get("id"))
        root_id, _ = root_depth_lookup.get(rid, ("", 0)) if rid else ("", 0)
        if root_id:
            union(f"thread:{root_id}", text_node)

    components: Dict[str, List[str]] = {}
    for node in parents:
        components.setdefault(find(node), []).append(node)

    node_to_group: Dict[str, str] = {}
    for nodes in components.values():
        payload = "\0".join(sorted(nodes)).encode("utf-8")
        group_key = "mixed:" + hashlib.sha256(payload).hexdigest()
        for node in nodes:
            node_to_group[node] = group_key
    return node_to_group


def build_mixed_split_groups(csv_path: str, column_name: str) -> Dict[str, str]:
    rows, id2row = _load_thread_rows(csv_path)
    return _build_mixed_split_groups_from_rows(rows, id2row, column_name)


def remap_structured_split_keys(
    samples: Iterator[StructuredChatSample], split_groups: Dict[str, str],
) -> Iterator[StructuredChatSample]:
    for sample in samples:
        sample.split_key = split_groups.get(sample.split_key, sample.split_key)
        yield sample


def _mixed_text_iter_from_rows(
    rows: List[Dict[str, Any]],
    id2row: Dict[str, Dict[str, Any]],
    column_name: str,
    split_groups: Optional[Dict[str, str]] = None,
    root_depth_lookup: Optional[Dict[str, Tuple[str, int]]] = None,
) -> Iterator[PlainTextSample]:
    split_groups = split_groups or {}
    root_depth_lookup = root_depth_lookup or _build_root_depth_lookup(rows, id2row)
    for row in rows:
        text = (row.get(column_name) or "").strip()
        if not text:
            continue
        text_node = "text:" + hashlib.sha256(text.encode("utf-8")).hexdigest()
        rid = normalize_id(row.get("id"))
        root_id, _ = root_depth_lookup.get(rid, ("", 0)) if rid else ("", 0)
        thread_node = f"thread:{root_id}" if root_id else ""
        split_key = split_groups.get(
            text_node,
            split_groups.get(thread_node, thread_node or text_node),
        )
        yield PlainTextSample(text=text, split_key=split_key)


def mixed_text_iter(
    csv_path: str, column_name: str, split_groups: Optional[Dict[str, str]] = None,
) -> Iterator[PlainTextSample]:
    rows, id2row = _load_thread_rows(csv_path)
    yield from _mixed_text_iter_from_rows(rows, id2row, column_name, split_groups)


def interleave_examples(*iterables: Iterator[Any]) -> Iterator[Any]:
    active = [iter(iterator) for iterator in iterables]
    while active:
        remaining = []
        for iterator in active:
            try:
                yield next(iterator)
                remaining.append(iterator)
            except StopIteration:
                continue
        active = remaining


def normalize_for_dedup(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").casefold()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def canonical_example_text(item: StructuredChatSample | PlainTextSample | str) -> str:
    if isinstance(item, PlainTextSample):
        return item.text
    if isinstance(item, str):
        return item
    parts = [f"system:{item.system}"] if item.system else []
    parts.extend(f"{turn.role}:{turn.content}" for turn in item.turns)
    parts.append(f"assistant:{item.target_answer}")
    return "\n".join(parts)


def quality_issues_for_example(
    item: StructuredChatSample | PlainTextSample | str,
    min_chars: int,
) -> List[str]:
    text = canonical_example_text(item)
    issues: List[str] = []
    if isinstance(item, (PlainTextSample, str)) and len(text.strip()) < min_chars:
        issues.append("too_short")
    if "\ufffd" in text:
        issues.append("replacement_character")
    if re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", text):
        issues.append("control_characters")
    if re.search(r"(?is)<(?:html|body|script|style|nav|footer)\b", text):
        issues.append("html_boilerplate")
    lines = [normalize_for_dedup(line) for line in text.splitlines() if line.strip()]
    if len(lines) >= 6 and len(set(lines)) / float(len(lines)) < 0.55:
        issues.append("repetitive_lines")
    return issues


def _simhash64(text: str, max_shingles: int = 512) -> int:
    words = re.findall(r"\w+", normalize_for_dedup(text), flags=re.UNICODE)
    if not words:
        return 0
    width = 5 if len(words) >= 5 else max(1, len(words))
    shingle_count = max(1, len(words) - width + 1)
    sample_count = min(shingle_count, max(32, int(max_shingles)))
    if sample_count >= shingle_count:
        shingle_indexes = range(shingle_count)
    else:
        shingle_indexes = (
            (sample_index * shingle_count) // sample_count
            for sample_index in range(sample_count)
        )
    vector = [0] * 64
    for index in shingle_indexes:
        shingle = " ".join(words[index:index + width])
        value = int.from_bytes(
            hashlib.blake2b(shingle.encode("utf-8"), digest_size=8).digest(),
            "big",
        )
        for bit in range(64):
            vector[bit] += 1 if value & (1 << bit) else -1
    result = 0
    for bit, score in enumerate(vector):
        if score >= 0:
            result |= 1 << bit
    return result


class NearDuplicateTracker:
    """Bounded LSH tracker for approximate near-duplicate detection."""

    def __init__(
        self, threshold: float, bucket_limit: int = 256, max_shingles: int = 512,
    ):
        self.max_hamming = max(1, int(round((1.0 - float(threshold)) * 64)))
        self.bucket_limit = max(16, int(bucket_limit))
        self.max_shingles = max(32, int(max_shingles))
        self.hashes: List[int] = []
        self.group_keys: List[Optional[str]] = []
        self.bands: Dict[Tuple[int, int], List[int]] = defaultdict(list)

    def add_and_find(
        self, text: str, group_key: Optional[str] = None,
    ) -> Tuple[bool, Optional[str]]:
        value = _simhash64(text, self.max_shingles)
        candidate_ids: set[int] = set()
        for band in range(4):
            band_value = (value >> (band * 16)) & 0xFFFF
            candidate_ids.update(self.bands.get((band, band_value), []))
        matching_ids = [
            index for index in sorted(candidate_ids)
            if (value ^ self.hashes[index]).bit_count() <= self.max_hamming
        ]
        is_near = bool(matching_ids)
        representative_key = (
            self.group_keys[matching_ids[0]] if matching_ids else group_key
        )
        index = len(self.hashes)
        self.hashes.append(value)
        self.group_keys.append(representative_key)
        for band in range(4):
            key = (band, (value >> (band * 16)) & 0xFFFF)
            bucket = self.bands[key]
            bucket.append(index)
            if len(bucket) > self.bucket_limit:
                del bucket[:len(bucket) - self.bucket_limit]
        return is_near, (representative_key if is_near else None)

    def check_and_add(self, text: str) -> bool:
        is_near, _ = self.add_and_find(text)
        return is_near


def _build_role_block(role: str, content: str, template_mode: str, eos_token: str) -> str:
    content = (content or "").strip()
    if role == "system":
        return f"<|System|>{content}{eos_token}"
    if role == "user":
        return f"<|Benutzer|>{content}{eos_token}"
    if role == "assistant":
        return f"<|Assistentin|>{content}{eos_token}"
    return content


def _build_assistant_prefix(template_mode: str) -> str:
    return "<|Assistentin|>"


def _apply_history_limit(turns: List[StructuredTurn], max_history_turns: Optional[int]) -> List[StructuredTurn]:
    if max_history_turns is None:
        return turns
    if len(turns) <= max_history_turns:
        return turns
    return turns[-max_history_turns:]


def _csv_has_column(csv_path: str, column_name: str) -> bool:
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            names = reader.fieldnames or []
            return column_name in names
    except Exception:
        return False


STRUCTURED_CSV_COLUMNS = (
    "id", "parent_id", "system", "Benutzer", "Kontext", "Assistentin",
)


def _mixed_dataset_errors(
    csv_path: str, text_column: str, require_text_sample: bool = False,
) -> List[str]:
    with open(csv_path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        required = (*STRUCTURED_CSV_COLUMNS, text_column)
        missing = [name for name in required if name not in fieldnames]
        if missing:
            return ["CSV-Spalten fehlen: " + ", ".join(missing)]
        if require_text_sample and not any(
            (row.get(text_column) or "").strip() for row in reader
        ):
            return [f"Keine nutzbaren Texte in {text_column}"]
    return []


def build_examples_stream(cfg: TrainConfig) -> Iterator[Any]:
    if cfg.template_mode in {"chat", "dialogplus"}:
        if cfg.mixed_training and cfg.training_phase != "sft":
            errors = _mixed_dataset_errors(cfg.csv_path, cfg.mixed_text_column)
            if errors:
                raise ValueError(
                    "Gemischtes Training ist nicht möglich: " + "; ".join(errors)
                )
        has_structured_columns = (
            _csv_has_column(cfg.csv_path, "id")
            and _csv_has_column(cfg.csv_path, "Assistentin")
        )
        if has_structured_columns:
            if cfg.mixed_training:
                rows, id2row = _load_thread_rows(cfg.csv_path)
                root_depth_lookup = _build_root_depth_lookup(rows, id2row)
                chat_examples = _chat_structured_iter_from_rows(
                    rows, id2row, root_depth_lookup,
                )
                split_groups = _build_mixed_split_groups_from_rows(
                    rows, id2row, cfg.mixed_text_column, root_depth_lookup,
                )
                text_examples = _mixed_text_iter_from_rows(
                    rows, id2row, cfg.mixed_text_column, split_groups, root_depth_lookup,
                )
                if cfg.training_phase == "pretrain":
                    return text_examples
                if cfg.training_phase == "sft":
                    return chat_examples
                return interleave_examples(
                    remap_structured_split_keys(chat_examples, split_groups),
                    text_examples,
                )
            chat_examples = (
                chat_structured_iter(cfg.csv_path)
                if cfg.template_mode == "chat"
                else dialogplus_structured_iter(cfg.csv_path)
            )
            return chat_examples
        if cfg.training_phase == "sft":
            raise ValueError(
                "Die SFT-Phase benötigt ein strukturiertes Dialog-Dataset"
            )
        return column_iter(cfg.csv_path, cfg.column_name)
    return column_iter(cfg.csv_path, cfg.column_name)


def pack_dialog_from_blocks_strict(
    prompt_blocks: List[List[int]],
    answer_ids: List[int],
    max_seq_length: int,
    include_prompt_loss: bool = False,
) -> Optional[Tuple[List[int], List[int]]]:
    if max_seq_length <= 0:
        raise ValueError("max_seq_length muss > 0 sein")

    if len(answer_ids) > max_seq_length:
        return None

    kept_prompt_blocks: List[List[int]] = []
    used = len(answer_ids)

    for block_ids in reversed(prompt_blocks):
        if not block_ids:
            continue
        if used + len(block_ids) <= max_seq_length:
            kept_prompt_blocks.insert(0, block_ids)
            used += len(block_ids)
        else:
            break

    input_ids = [tok for part in kept_prompt_blocks for tok in part] + answer_ids
    if include_prompt_loss:
        labels = input_ids.copy()
    else:
        labels = ([-100] * (len(input_ids) - len(answer_ids))) + answer_ids.copy()

    if not input_ids or len(input_ids) != len(labels):
        return None

    return input_ids, labels


def _text_token_chunks(
    text: str,
    tokenizer: Any,
    max_tokens: int,
    overlap_tokens: int,
    min_tokens: int,
) -> Tuple[List[List[int]], int]:
    """Split text on token boundaries and prefer nearby sentence/paragraph ends."""
    max_tokens = max(1, int(max_tokens))
    overlap_tokens = min(max(0, int(overlap_tokens)), max(0, max_tokens // 2))
    min_tokens = min(max_tokens, max(1, int(min_tokens)))

    offsets: Optional[List[Tuple[int, int]]] = None
    try:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        token_ids = list(encoded["input_ids"])
        raw_offsets = encoded.get("offset_mapping")
        if raw_offsets and len(raw_offsets) == len(token_ids):
            offsets = [(int(start), int(end)) for start, end in raw_offsets]
    except Exception:
        token_ids = list(tokenizer(text, add_special_tokens=False)["input_ids"])

    if len(token_ids) <= max_tokens:
        return ([token_ids] if token_ids else []), len(token_ids)

    boundary_chars = {
        match.end()
        for match in re.finditer(r"(?:\n\s*\n|[.!?](?:[\"'»”)]*)\s+|[;:]\s+)", text)
    }
    chunks: List[List[int]] = []
    start = 0
    while start < len(token_ids):
        desired_end = min(len(token_ids), start + max_tokens)
        end = desired_end
        if offsets is not None and desired_end < len(token_ids):
            earliest = min(desired_end, start + min_tokens)
            candidates: List[int] = []
            for token_end in range(earliest, desired_end + 1):
                char_end = offsets[token_end - 1][1]
                if char_end in boundary_chars:
                    candidates.append(token_end)
                    continue
                following = text[char_end: min(len(text), char_end + 3)]
                if "\n" in following:
                    candidates.append(token_end)
            if candidates:
                end = candidates[-1]

        if end <= start:
            end = min(len(token_ids), start + max_tokens)
        chunks.append(token_ids[start:end])
        if end >= len(token_ids):
            break
        start = max(start + 1, end - overlap_tokens)

    return chunks, len(token_ids)


def tokenize_text_examples(
    item: PlainTextSample | str,
    tokenizer: Any,
    max_seq_length: int,
    *,
    chunk_long_texts: bool,
    text_chunk_overlap: int,
    text_chunk_min_tokens: int,
    append_eos_to_text: bool,
) -> List[Dict[str, Any]]:
    text = item if isinstance(item, str) else item.text
    text = (text or "").strip()
    if not text:
        return []
    split_key = (
        "text:" + hashlib.sha256(text.encode("utf-8")).hexdigest()
        if isinstance(item, str)
        else item.split_key
    )
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        eos_id = tokenizer.pad_token_id
    reserve_eos = 1 if append_eos_to_text and eos_id is not None else 0
    content_limit = max(1, int(max_seq_length) - reserve_eos)

    if chunk_long_texts:
        chunks, original_token_count = _text_token_chunks(
            text,
            tokenizer,
            content_limit,
            text_chunk_overlap,
            text_chunk_min_tokens,
        )
    else:
        ids = list(tokenizer(text, add_special_tokens=False)["input_ids"])
        original_token_count = len(ids)
        chunks = [ids] if ids and len(ids) <= content_limit else []

    samples: List[Dict[str, Any]] = []
    chunk_count = len(chunks)
    for chunk_index, content_ids in enumerate(chunks):
        ids = list(content_ids)
        if append_eos_to_text and eos_id is not None and (not ids or ids[-1] != eos_id):
            ids.append(int(eos_id))
        if len(ids) < 2:
            ids.append(int(eos_id or tokenizer.pad_token_id or 0))
        if len(ids) > max_seq_length:
            continue
        samples.append({
            "input_ids": ids,
            "attention_mask": [1] * len(ids),
            "labels": ids.copy(),
            "seq_len": len(ids),
            "split_key": split_key,
            "sample_type": "text",
            "chunk_index": int(chunk_index),
            "chunk_count": int(chunk_count),
            "original_token_count": int(original_token_count),
        })
    return samples


def tokenize_example(
    item: StructuredChatSample | PlainTextSample | str,
    tokenizer,
    max_seq_length: int,
    template_mode: str,
    max_history_turns: Optional[int],
    include_prompt_loss: bool = False,
) -> Optional[Dict[str, Any]]:
    if isinstance(item, (str, PlainTextSample)):
        samples = tokenize_text_examples(
            item,
            tokenizer,
            max_seq_length,
            chunk_long_texts=False,
            text_chunk_overlap=0,
            text_chunk_min_tokens=1,
            append_eos_to_text=True,
        )
        return samples[0] if samples else None

    prompt_blocks: List[List[int]] = []
    eos_token = tokenizer.eos_token or "</s>"

    if item.system:
        system_block = _build_role_block("system", item.system, template_mode, eos_token)
        system_ids = tokenizer(system_block, add_special_tokens=False)["input_ids"]
        if len(system_ids) > max_seq_length:
            return None
        prompt_blocks.append(system_ids)

    limited_turns = _apply_history_limit(item.turns, max_history_turns)

    for turn in limited_turns:
        block = _build_role_block(turn.role, turn.content, template_mode, eos_token)
        block_ids = tokenizer(block, add_special_tokens=False)["input_ids"]
        if len(block_ids) > max_seq_length:
            return None
        prompt_blocks.append(block_ids)

    assistant_prefix = _build_assistant_prefix(template_mode)
    assistant_prefix_ids = tokenizer(assistant_prefix, add_special_tokens=False)["input_ids"]
    if len(assistant_prefix_ids) > max_seq_length:
        return None
    prompt_blocks.append(assistant_prefix_ids)

    answer_ids = tokenizer(
        (item.target_answer or "").strip() + eos_token,
        add_special_tokens=False,
    )["input_ids"]

    packed = pack_dialog_from_blocks_strict(
        prompt_blocks=prompt_blocks,
        answer_ids=answer_ids,
        max_seq_length=max_seq_length,
        include_prompt_loss=bool(include_prompt_loss),
    )
    if packed is None:
        return None

    input_ids, labels = packed

    if len(input_ids) < 2:
        eos_or_pad = tokenizer.eos_token_id or tokenizer.pad_token_id or 0
        input_ids = input_ids + [eos_or_pad]
        labels = labels + [eos_or_pad]

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
        "seq_len": len(input_ids),
        "split_key": item.split_key,
        "sample_type": "dialog",
        "chunk_index": 0,
        "chunk_count": 1,
        "original_token_count": len(input_ids),
    }


class ShortTextPacker:
    """Pack short text documents without crossing train/validation boundaries."""

    def __init__(self, target_length: int):
        self.target_length = max(64, int(target_length))
        self.buffers: Dict[str, Dict[str, Any]] = {}
        self.input_segments = 0
        self.output_sequences = 0

    def _start(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        packed = dict(sample)
        packed["input_ids"] = list(sample["input_ids"])
        packed["attention_mask"] = list(sample["attention_mask"])
        packed["labels"] = list(sample["labels"])
        packed["packed_segment_count"] = 1
        packed["_packed_keys"] = [str(sample.get("split_key") or "")]
        return packed

    def _finish(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        keys = sample.pop("_packed_keys", [])
        if len(keys) > 1:
            payload = "\0".join(keys).encode("utf-8")
            sample["split_key"] = "packed:" + hashlib.sha256(payload).hexdigest()
        sample["seq_len"] = len(sample["input_ids"])
        self.output_sequences += 1
        return sample

    def add(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        if sample.get("sample_type") != "text" or len(sample["input_ids"]) >= self.target_length:
            self.input_segments += 1
            self.output_sequences += 1
            sample["packed_segment_count"] = 1
            return [sample]

        self.input_segments += 1
        assigned_split = str(sample.get("assigned_split") or "train")
        current = self.buffers.get(assigned_split)
        if current is None:
            self.buffers[assigned_split] = self._start(sample)
            return []

        if len(current["input_ids"]) + len(sample["input_ids"]) > self.target_length:
            finished = self._finish(current)
            self.buffers[assigned_split] = self._start(sample)
            return [finished]

        next_labels = list(sample["labels"])
        if next_labels:
            next_labels[0] = -100
        current["input_ids"].extend(sample["input_ids"])
        current["attention_mask"].extend(sample["attention_mask"])
        current["labels"].extend(next_labels)
        current["packed_segment_count"] += 1
        current["original_token_count"] = int(current.get("original_token_count", 0)) + int(
            sample.get("original_token_count", len(sample["input_ids"]))
        )
        current["_packed_keys"].append(str(sample.get("split_key") or ""))
        return []

    def flush(self) -> List[Dict[str, Any]]:
        samples = [self._finish(sample) for sample in self.buffers.values()]
        self.buffers.clear()
        return samples


def count_examples_fast(cfg: TrainConfig) -> int:
    count = 0
    with open(cfg.csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        has_chat_columns = bool(reader.fieldnames) and ("id" in reader.fieldnames) and ("Assistentin" in reader.fieldnames)
        if cfg.template_mode in {"chat", "dialogplus"} and has_chat_columns:
            for row in reader:
                rid = normalize_id(row.get("id", ""))
                asst = (row.get("Assistentin") or "").strip()
                if rid and asst and cfg.training_phase != "pretrain":
                    count += 1
                if (
                    cfg.mixed_training
                    and cfg.training_phase != "sft"
                    and (row.get(cfg.mixed_text_column) or "").strip()
                ):
                    count += 1
        else:
            for row in reader:
                txt = (row.get(cfg.column_name) or "").strip()
                if txt:
                    count += 1
    return count


def estimate_examples_from_audit(cfg: TrainConfig, report: Dict[str, Any]) -> int:
    """Use the already completed audit instead of scanning the CSV again."""
    if not report:
        return 0
    if "usable_text_samples" in report:
        return max(0, int(report.get("usable_text_samples", 0)))

    assistant_samples = max(0, int(report.get("usable_assistant_samples", 0)))
    mixed_text_samples = max(0, int(report.get("usable_mixed_text_samples", 0)))
    if cfg.training_phase == "pretrain":
        return mixed_text_samples
    if cfg.training_phase == "sft":
        return assistant_samples
    if cfg.mixed_training:
        return assistant_samples + mixed_text_samples
    return assistant_samples


def audit_dataset(
    cfg: TrainConfig,
    outdir: Path,
    is_main: bool,
    progress_cb: Optional[Callable[[int], None]] = None,
) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "csv_path": str(Path(cfg.csv_path).expanduser().resolve()),
        "template_mode": cfg.template_mode,
        "training_phase": cfg.training_phase,
        "mixed_training": bool(cfg.mixed_training),
        "mixed_text_column": cfg.mixed_text_column,
        "errors": [],
        "warnings": [],
    }
    with open(cfg.csv_path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        report["columns"] = fieldnames
        is_threaded = (
            cfg.template_mode in {"chat", "dialogplus"}
            and {"id", "Assistentin"}.issubset(fieldnames)
        )
        mixed_column_present = cfg.mixed_text_column in fieldnames
        mixed_text_required = bool(
            cfg.mixed_training and cfg.training_phase != "sft"
        )
        missing_mixed_columns = [
            name
            for name in (*STRUCTURED_CSV_COLUMNS, cfg.mixed_text_column)
            if name not in fieldnames
        ]
        mixed_training_ready = bool(
            cfg.training_phase == "sft"
            or (mixed_text_required and not missing_mixed_columns)
        )
        report["mixed_text_column_present"] = mixed_column_present
        report["mixed_training_ready"] = mixed_training_ready
        if mixed_text_required and missing_mixed_columns:
            report["errors"].append(
                "CSV-Spalten für gemischtes Training fehlen: "
                + ", ".join(missing_mixed_columns)
            )
        if cfg.training_phase == "pretrain" and is_threaded and not cfg.mixed_training:
            report["errors"].append(
                "Die Pretrain-Phase benötigt eine Plain-Text-CSV oder den aktivierten Mischmodus"
            )
        if cfg.training_phase == "sft" and not is_threaded:
            report["errors"].append(
                "Die SFT-Phase benötigt ein strukturiertes Dialog-Dataset"
            )
        report["training_phase_ready"] = not any(
            "Phase benötigt" in error for error in report["errors"]
        )
        row_count = 0

        if is_threaded or cfg.mixed_training:
            seen_ids: set[str] = set()
            seen_pairs: set[str] = set()
            parent_by_id: Dict[str, str] = {}
            duplicate_ids = duplicate_pairs = 0
            missing_id = missing_user = missing_answer = 0
            usable_assistant_samples = 0
            mixed_text_samples = mixed_text_duplicates = 0
            seen_mixed_texts: set[str] = set()
            quality_issue_counts: Counter[str] = Counter()
            near_duplicate_samples = 0
            potential_conflicting_answers = 0
            near_tracker = NearDuplicateTracker(
                cfg.near_duplicate_threshold, max_shingles=cfg.near_duplicate_max_shingles,
            )
            prompt_answers: Dict[str, set[str]] = defaultdict(set)
            for row in reader:
                row_count += 1
                if progress_cb is not None and (row_count == 1 or row_count % 1000 == 0):
                    progress_cb(row_count)
                rid = normalize_id(row.get("id"))
                parent = normalize_id(row.get("parent_id"))
                user = (row.get("Benutzer") or "").strip()
                answer = (row.get("Assistentin") or "").strip()
                mixed_text = (
                    (row.get(cfg.mixed_text_column) or "").strip()
                    if cfg.mixed_training and mixed_column_present
                    else ""
                )
                text_only_row = bool(mixed_text and not answer)
                if not rid and not text_only_row:
                    missing_id += 1
                else:
                    if rid:
                        if rid in seen_ids:
                            duplicate_ids += 1
                        seen_ids.add(rid)
                        parent_by_id[rid] = parent
                if not user and not text_only_row:
                    missing_user += 1
                if not answer and not mixed_text:
                    missing_answer += 1
                if rid and answer:
                    usable_assistant_samples += 1
                if user and answer:
                    pair_hash = hashlib.sha256(f"{user}\0{answer}".encode("utf-8")).hexdigest()
                    if pair_hash in seen_pairs:
                        duplicate_pairs += 1
                    seen_pairs.add(pair_hash)
                    prompt_key = normalize_for_dedup(
                        "\n".join([
                            (row.get("system") or "").strip(),
                            (row.get("Kontext") or "").strip(),
                            user,
                        ])
                    )
                    answer_key = normalize_for_dedup(answer)
                    known_answers = prompt_answers[prompt_key]
                    if known_answers and answer_key not in known_answers:
                        potential_conflicting_answers += 1
                    known_answers.add(answer_key)
                if mixed_text:
                    mixed_text_samples += 1
                    normalized_text = normalize_for_dedup(mixed_text)
                    text_hash = hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()
                    if text_hash in seen_mixed_texts:
                        mixed_text_duplicates += 1
                    seen_mixed_texts.add(text_hash)
                    if cfg.near_duplicate_action != "off" and near_tracker.check_and_add(mixed_text):
                        near_duplicate_samples += 1
                    quality_issue_counts.update(
                        quality_issues_for_example(mixed_text, cfg.quality_min_chars)
                    )

            cycles = 0
            visit_state: Dict[str, int] = {}
            for rid in parent_by_id:
                if visit_state.get(rid, 0) == 2:
                    continue
                path: List[str] = []
                cur = rid
                while cur and cur in parent_by_id and visit_state.get(cur, 0) == 0:
                    visit_state[cur] = 1
                    path.append(cur)
                    cur = parent_by_id.get(cur, "")
                if cur and visit_state.get(cur, 0) == 1:
                    cycles += 1
                for node in path:
                    visit_state[node] = 2

            report.update({
                "usable_assistant_samples": usable_assistant_samples,
                "missing_id": missing_id,
                "missing_user": missing_user,
                "missing_answer": missing_answer,
                "duplicate_ids": duplicate_ids,
                "duplicate_prompt_answer_pairs": duplicate_pairs,
                "thread_cycles": cycles,
                "usable_mixed_text_samples": mixed_text_samples,
                "duplicate_mixed_text_samples": mixed_text_duplicates,
                "approximate_near_duplicate_samples": near_duplicate_samples,
                "potential_conflicting_answers": potential_conflicting_answers,
                "quality_issues": dict(sorted(quality_issue_counts.items())),
            })
            if duplicate_ids:
                report["errors"].append(f"{duplicate_ids} doppelte IDs")
            if cycles:
                report["errors"].append(f"{cycles} zyklische Parent-Ketten")
            if missing_id:
                report["warnings"].append(f"{missing_id} Zeilen ohne ID")
            if missing_user:
                report["warnings"].append(f"{missing_user} Zeilen ohne Benutzertext")
            if missing_answer:
                report["warnings"].append(f"{missing_answer} Zeilen ohne Assistentinnen-Antwort")
            if duplicate_pairs:
                report["warnings"].append(f"{duplicate_pairs} exakte Prompt-Antwort-Duplikate")
            if mixed_text_duplicates:
                report["warnings"].append(
                    f"{mixed_text_duplicates} exakte Duplikate in {cfg.mixed_text_column}"
                )
            if near_duplicate_samples:
                report["warnings"].append(
                    f"{near_duplicate_samples} mögliche Near-Duplikate"
                )
            if potential_conflicting_answers:
                report["warnings"].append(
                    f"{potential_conflicting_answers} möglicherweise widersprüchliche Antworten"
                )
            if quality_issue_counts:
                report["warnings"].append(
                    "Qualitätshinweise: "
                    + ", ".join(
                        f"{name}={count}" for name, count in sorted(quality_issue_counts.items())
                    )
                )
            if mixed_text_required and mixed_text_samples == 0:
                report["mixed_training_ready"] = False
                report["errors"].append(
                    f"Keine nutzbaren Texte in {cfg.mixed_text_column}"
                )
        else:
            if cfg.column_name not in fieldnames:
                report["errors"].append(f"Textspalte fehlt: {cfg.column_name}")
            seen_values: set[str] = set()
            usable_count = empty_count = duplicate_count = 0
            near_duplicate_samples = 0
            quality_issue_counts: Counter[str] = Counter()
            near_tracker = NearDuplicateTracker(
                cfg.near_duplicate_threshold, max_shingles=cfg.near_duplicate_max_shingles,
            )
            for row in reader:
                row_count += 1
                if progress_cb is not None and (row_count == 1 or row_count % 1000 == 0):
                    progress_cb(row_count)
                value = (row.get(cfg.column_name) or "").strip()
                if not value:
                    empty_count += 1
                    continue
                usable_count += 1
                normalized_value = normalize_for_dedup(value)
                if normalized_value in seen_values:
                    duplicate_count += 1
                seen_values.add(normalized_value)
                if cfg.near_duplicate_action != "off" and near_tracker.check_and_add(value):
                    near_duplicate_samples += 1
                quality_issue_counts.update(
                    quality_issues_for_example(value, cfg.quality_min_chars)
                )
            report.update({
                "usable_text_samples": usable_count,
                "empty_text_samples": empty_count,
                "duplicate_text_samples": duplicate_count,
                "approximate_near_duplicate_samples": near_duplicate_samples,
                "quality_issues": dict(sorted(quality_issue_counts.items())),
            })
            if not usable_count:
                report["errors"].append("Keine nutzbaren Textsamples")
            if duplicate_count:
                report["warnings"].append(f"{duplicate_count} exakte Textduplikate")
            if near_duplicate_samples:
                report["warnings"].append(
                    f"{near_duplicate_samples} mögliche Near-Duplikate"
                )
            if quality_issue_counts:
                report["warnings"].append(
                    "Qualitätshinweise: "
                    + ", ".join(
                        f"{name}={count}" for name, count in sorted(quality_issue_counts.items())
                    )
                )

    report["rows"] = row_count

    report["ok"] = not report["errors"]
    if is_main:
        (outdir / "dataset_audit.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        LOGGER.info("Dataset-Audit: %s", json.dumps(report, ensure_ascii=False))
    return report


def dataset_cache_root(cfg: TrainConfig) -> Path:
    if cfg.dataset_cache_dir:
        root = Path(cfg.dataset_cache_dir).expanduser().resolve()
    else:
        output_dir = Path(
            cfg.output_dir or cfg.save_dir or "./training_outputs/worker_run"
        ).expanduser().resolve()
        root = output_dir.parent / "_dataset_cache"
    root.mkdir(parents=True, exist_ok=True)
    return root


def dataset_audit_cache_path(cfg: TrainConfig) -> Path:
    csv_path = Path(cfg.csv_path).expanduser().resolve()
    csv_stat = csv_path.stat()
    payload = {
        "audit_schema": 2,
        "csv_path": str(csv_path),
        "csv_mtime_ns": int(csv_stat.st_mtime_ns),
        "csv_size": int(csv_stat.st_size),
        "template_mode": cfg.template_mode,
        "column_name": cfg.column_name,
        "mixed_training": bool(cfg.mixed_training),
        "mixed_text_column": cfg.mixed_text_column,
        "training_phase": cfg.training_phase,
        "near_duplicate_action": cfg.near_duplicate_action,
        "near_duplicate_threshold": float(cfg.near_duplicate_threshold),
        "near_duplicate_max_shingles": int(cfg.near_duplicate_max_shingles),
        "quality_min_chars": int(cfg.quality_min_chars),
    }
    key = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:24]
    audit_dir = dataset_cache_root(cfg) / "audits"
    audit_dir.mkdir(parents=True, exist_ok=True)
    return audit_dir / f"audit_{key}.json"


def audit_dataset_cached(
    cfg: TrainConfig,
    outdir: Path,
    status_writer: Optional[JsonStatusWriter] = None,
) -> Dict[str, Any]:
    cache_path = dataset_audit_cache_path(cfg)
    output_path = outdir / "dataset_audit.json"
    if cfg.use_dataset_cache and cache_path.exists() and not cfg.rebuild_dataset_cache:
        report = json.loads(cache_path.read_text(encoding="utf-8"))
        _atomic_write_json(output_path, report)
        LOGGER.info("Dataset-Audit aus gemeinsamem Cache geladen: %s", cache_path)
        return report

    def report_progress(rows: int) -> None:
        if SHUTDOWN.stop:
            raise RuntimeError("Dataset-Audit durch Stop-Signal abgebrochen.")
        if status_writer is not None:
            status_writer.write({
                "running": True,
                "status": "auditing_dataset",
                "eta": f"Dataset-Audit: {rows} Zeilen geprüft",
                "dataset_progress": {
                    "phase": "audit",
                    "seen_samples": int(rows),
                    "expected_samples": 0,
                    "done": False,
                    "updated_at": time.time(),
                },
                "dataset_ready": False,
            })

    report = audit_dataset(cfg, outdir, True, progress_cb=report_progress)
    if cfg.use_dataset_cache:
        _atomic_write_json(cache_path, report)
    return report


def format_raw_example_preview(item: Any) -> Tuple[str, str]:
    try:
        if isinstance(item, StructuredChatSample):
            parts = []
            if item.system:
                parts.append(f"[SYSTEM]\n{item.system}")
            for turn in item.turns:
                parts.append(f"[{turn.role.upper()}]\n{turn.content}")
            parts.append(f"[TARGET_ASSISTANT]\n{item.target_answer}")
            preview = "\n\n".join(parts)
        elif isinstance(item, PlainTextSample):
            preview = item.text
        else:
            preview = str(item)
    except Exception:
        preview = ""
    return preview[:4000], preview[:20000]


def iter_tokenizer_training_texts(cfg: TrainConfig) -> Iterator[str]:
    for item in build_examples_stream(cfg):
        text = canonical_example_text(item).strip()
        if text:
            yield text


def prepare_scratch_tokenizer_if_requested(
    cfg: TrainConfig, outdir: Path, ctx: DistContext, example_count: Optional[int] = None,
) -> None:
    if cfg.resume:
        cfg.tokenizer_dir = str(Path(cfg.resume).expanduser().resolve())
        return
    if not cfg.train_scratch_tokenizer:
        return
    if not cfg.train_from_scratch:
        raise ValueError(
            "train_scratch_tokenizer ist nur zusammen mit train_from_scratch erlaubt"
        )

    tokenizer_dir = (
        compute_scratch_tokenizer_cache_dir(cfg)
        if cfg.use_dataset_cache else outdir / "scratch_tokenizer"
    )
    error_path = outdir / "scratch_tokenizer_error.txt"
    if ctx.is_main:
        error_path.unlink(missing_ok=True)
        try:
            cache_complete = (
                (tokenizer_dir / "training_report.json").is_file()
                and (tokenizer_dir / "tokenizer_config.json").is_file()
            )
            if cache_complete and not cfg.rebuild_dataset_cache:
                LOGGER.info("Scratch-Tokenizer aus gemeinsamem Cache geladen: %s", tokenizer_dir)
            else:
                if tokenizer_dir.exists():
                    shutil.rmtree(tokenizer_dir, ignore_errors=True)
                tokenizer_dir.mkdir(parents=True, exist_ok=True)
                base_tokenizer = AutoTokenizer.from_pretrained(
                    cfg.tokenizer_dir or cfg.model_dir,
                    trust_remote_code=False,
                    use_fast=True,
                )
                if not hasattr(base_tokenizer, "train_new_from_iterator"):
                    raise RuntimeError("Der gewählte Tokenizer unterstützt kein Scratch-Training")
                example_count = int(example_count or count_examples_fast(cfg))
                trained_tokenizer = base_tokenizer.train_new_from_iterator(
                    iter_tokenizer_training_texts(cfg),
                    vocab_size=cfg.scratch_tokenizer_vocab_size,
                    length=max(1, example_count),
                )
                prepare_tokenizer(
                    trained_tokenizer,
                    template_mode=cfg.template_mode,
                    force_template=bool(cfg.force_template),
                )
                trained_tokenizer.save_pretrained(tokenizer_dir)
                _atomic_write_json(tokenizer_dir / "training_report.json", {
                    "source": cfg.tokenizer_dir or cfg.model_dir,
                    "vocab_size_requested": int(cfg.scratch_tokenizer_vocab_size),
                    "vocab_size_effective": int(len(trained_tokenizer)),
                    "examples_seen_estimate": int(example_count),
                    "training_phase": cfg.training_phase,
                })
                LOGGER.info(
                    "Scratch-Tokenizer trainiert | source=%s vocab=%s path=%s",
                    cfg.tokenizer_dir or cfg.model_dir,
                    len(trained_tokenizer),
                    tokenizer_dir,
                )
        except Exception as exc:
            error_path.write_text(
                f"{exc.__class__.__name__}: {exc}\n\n{traceback.format_exc()}",
                encoding="utf-8",
            )
    barrier(ctx)
    if error_path.exists():
        raise RuntimeError(error_path.read_text(encoding="utf-8"))
    if not tokenizer_dir.is_dir():
        raise RuntimeError(f"Scratch-Tokenizer wurde nicht erzeugt: {tokenizer_dir}")
    cfg.tokenizer_dir = str(tokenizer_dir)


def tokenizer_source_signature(source: str) -> Dict[str, Any]:
    source_path = Path(source).expanduser()
    if not source_path.exists() or not source_path.is_dir():
        return {"kind": "model_id", "value": str(source)}

    tokenizer_files = (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
        "spiece.model",
        "tokenizer.model",
        "sentencepiece.bpe.model",
    )
    file_digests: Dict[str, str] = {}
    for filename in tokenizer_files:
        path = source_path / filename
        if not path.is_file():
            continue
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        file_digests[filename] = digest.hexdigest()
    if not file_digests:
        return {"kind": "local_path", "value": str(source_path.resolve())}
    return {"kind": "local_tokenizer", "files": file_digests}


def compute_scratch_tokenizer_cache_dir(cfg: TrainConfig) -> Path:
    csv_path = Path(cfg.csv_path).expanduser().resolve()
    csv_stat = csv_path.stat()
    payload = {
        "scratch_tokenizer_schema": 1,
        "csv_path": str(csv_path),
        "csv_mtime_ns": int(csv_stat.st_mtime_ns),
        "csv_size": int(csv_stat.st_size),
        "source": tokenizer_source_signature(cfg.tokenizer_dir or cfg.model_dir),
        "template_mode": cfg.template_mode,
        "column_name": cfg.column_name,
        "mixed_training": bool(cfg.mixed_training),
        "mixed_text_column": cfg.mixed_text_column,
        "training_phase": cfg.training_phase,
        "vocab_size": int(cfg.scratch_tokenizer_vocab_size),
    }
    key = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:24]
    return dataset_cache_root(cfg) / "tokenizers" / f"scratch_{key}"


def compute_shard_cache_dir(cfg: TrainConfig) -> Path:
    csv_path = Path(cfg.csv_path).expanduser().resolve()
    csv_stat = csv_path.stat()

    payload = {
        "cache_schema": 4,
        "csv_path": str(csv_path),
        "csv_mtime_ns": int(csv_stat.st_mtime_ns),
        "csv_size": int(csv_stat.st_size),
        "tokenizer_source": tokenizer_source_signature(cfg.tokenizer_dir or cfg.model_dir),
        "template_mode": cfg.template_mode,
        "column_name": cfg.column_name,
        "mixed_training": bool(cfg.mixed_training),
        "mixed_text_column": cfg.mixed_text_column,
        "training_phase": cfg.training_phase,
        "chunk_long_texts": bool(cfg.chunk_long_texts),
        "text_chunk_overlap": int(cfg.text_chunk_overlap),
        "text_chunk_min_tokens": int(cfg.text_chunk_min_tokens),
        "append_eos_to_text": bool(cfg.append_eos_to_text),
        "pack_short_texts": bool(cfg.pack_short_texts),
        "pack_target_length": int(cfg.pack_target_length),
        "deduplicate_exact": bool(cfg.deduplicate_exact),
        "near_duplicate_action": cfg.near_duplicate_action,
        "near_duplicate_threshold": float(cfg.near_duplicate_threshold),
        "near_duplicate_max_shingles": int(cfg.near_duplicate_max_shingles),
        "quality_filter_mode": cfg.quality_filter_mode,
        "quality_min_chars": int(cfg.quality_min_chars),
        "val_split": float(cfg.val_split),
        "split_seed": int(cfg.split_seed),
        "max_seq_length": int(cfg.max_seq_length),
        "sort_by_length": bool(cfg.sort_by_length),
        "max_history_turns": cfg.max_history_turns,
        "strict_whole_turns": True,
        "use_ngrams": bool(cfg.use_ngrams),
        "ngram_max": int(cfg.ngram_max),
        "ngram_top_k": int(cfg.ngram_top_k),
        "ngram_min_chars": int(cfg.ngram_min_chars),
        "ngram_min_words": int(cfg.ngram_min_words),
        "ngram_max_samples": int(cfg.ngram_max_samples),
        "ngram_budgeted": bool(cfg.ngram_budgeted),
        "ngram_target_fit": float(cfg.ngram_target_fit),
        "ngram_eval_samples": int(cfg.ngram_eval_samples),
        "ngram_add_batch": int(cfg.ngram_add_batch),
        "ngram_min_count": int(cfg.ngram_min_count),
        "ngram_max_token_chars": int(cfg.ngram_max_token_chars),
        "ngram_max_tokens_per_text": int(cfg.ngram_max_tokens_per_text),
    }
    key = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:24]
    return dataset_cache_root(cfg) / f"shards_{key}"


def shard_file_path(cache_dir: Path, shard_idx: int) -> Path:
    return cache_dir / f"shard_{shard_idx:06d}.pkl"


def dataset_ready_path(cache_dir: Path) -> Path:
    return cache_dir / "_dataset_ready.json"


def dataset_error_path(cache_dir: Path) -> Path:
    return cache_dir / "_dataset_error.txt"


def dataset_meta_path(cache_dir: Path) -> Path:
    return cache_dir / "_dataset_meta.json"


def dataset_progress_path(cache_dir: Path) -> Path:
    return cache_dir / "_dataset_progress.json"


def dataset_preview_path(cache_dir: Path) -> Path:
    return cache_dir / "_dataset_preview.json"


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _write_shard(cache_dir: Path, shard_idx: int, global_start: int, samples: List[Dict[str, Any]]) -> None:
    payload = {
        "shard_idx": shard_idx,
        "global_start": global_start,
        "num_samples": len(samples),
        "samples": samples,
    }
    tmp = shard_file_path(cache_dir, shard_idx).with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(shard_file_path(cache_dir, shard_idx))


def _flush_pending_samples(
    *,
    cache_dir: Path,
    pending_samples: List[Dict[str, Any]],
    current_samples: List[Dict[str, Any]],
    shard_idx: int,
    global_start: int,
    shard_size: int,
    sort_by_length: bool,
    sort_by_similarity: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int, int]:
    if (sort_by_length or sort_by_similarity) and pending_samples:
        def sample_sort_key(sample: Dict[str, Any]) -> Tuple[int, int]:
            seq_len = int(sample.get("seq_len") or len(sample["input_ids"]))
            if not sort_by_similarity:
                return (seq_len, 0)
            labels = sample.get("labels")
            input_ids = sample.get("input_ids") or []
            if labels and len(labels) == len(input_ids):
                prompt_ids = [int(tid) for tid, lab in zip(input_ids, labels) if int(lab) == -100]
            else:
                prompt_ids = [int(tid) for tid in input_ids[: min(128, len(input_ids))]]
            if not prompt_ids:
                return (seq_len, 0)
            signature = 0
            for tid in prompt_ids[:128]:
                signature ^= ((tid * 2654435761) & 0xFFFFFFFF)
            signature ^= (len(prompt_ids) & 0xFFFFFFFF)
            return (seq_len, signature)

        pending_samples.sort(key=sample_sort_key)

    while pending_samples:
        free_slots = shard_size - len(current_samples)
        if free_slots <= 0:
            _write_shard(cache_dir, shard_idx, global_start, current_samples)
            global_start += len(current_samples)
            shard_idx += 1
            current_samples = []
            free_slots = shard_size

        take = min(free_slots, len(pending_samples))
        current_samples.extend(pending_samples[:take])
        del pending_samples[:take]

        if len(current_samples) >= shard_size:
            _write_shard(cache_dir, shard_idx, global_start, current_samples)
            global_start += len(current_samples)
            shard_idx += 1
            current_samples = []

    return pending_samples, current_samples, shard_idx, global_start


def _write_dataset_progress(
    cache_dir: Path,
    *,
    seen_samples: int,
    tokenized_samples: int,
    skipped_samples: int,
    shard_idx: int,
    done: bool,
    status_writer: Optional[JsonStatusWriter] = None,
    expected_samples: int = 0,
    started_at: Optional[float] = None,
    num_workers: int = 1,
    phase: str = "tokenizing",
    processed_samples: Optional[int] = None,
) -> None:
    processed = int(seen_samples if processed_samples is None else processed_samples)
    elapsed_seconds = max(0.0, time.time() - started_at) if started_at else 0.0
    samples_per_second = (
        float(processed) / elapsed_seconds if elapsed_seconds > 0.0 else 0.0
    )
    remaining_samples = max(0, int(expected_samples) - processed)
    eta_seconds = (
        float(remaining_samples) / samples_per_second
        if samples_per_second > 0.0 and expected_samples > 0 else None
    )
    payload = {
        "seen_samples": int(seen_samples),
        "processed_samples": processed,
        "tokenized_samples": int(tokenized_samples),
        "skipped_samples": int(skipped_samples),
        "num_shards_written": int(shard_idx),
        "done": bool(done),
        "phase": str(phase),
        "expected_samples": int(expected_samples),
        "num_workers": int(num_workers),
        "elapsed_seconds": float(elapsed_seconds),
        "samples_per_second": float(samples_per_second),
        "eta_seconds": eta_seconds,
        "cache_hit": False,
        "updated_at": time.time(),
    }
    _atomic_write_json(dataset_progress_path(cache_dir), payload)
    if status_writer is not None:
        status_writer.write({
            "running": True,
            "status": "dataset_ready" if done else "building_dataset",
            "eta": (
                "Dataset vollständig aufgebaut"
                if done else (
                    f"Dataset-Aufbau: {format_eta(eta_seconds)}"
                    if eta_seconds is not None else "Dataset wird vollständig aufgebaut"
                )
            ),
            "dataset_progress": payload,
            "dataset_ready": bool(done),
            "total_samples_real": int(tokenized_samples) if done else None,
            "skipped_samples": int(skipped_samples),
        })


def _histogram_percentile(histogram: Counter[int], quantile: float) -> int:
    total = sum(histogram.values())
    if total <= 0:
        return 0
    target = max(1, int(math.ceil(total * min(1.0, max(0.0, quantile)))))
    running = 0
    for upper_bound, count in sorted(histogram.items()):
        running += count
        if running >= target:
            return int(upper_bound)
    return int(max(histogram))


_DATASET_TOKENIZER_LOCAL = threading.local()


def resolve_dataset_num_workers(cfg: TrainConfig, estimated_samples: int = 0) -> int:
    if cfg.dataset_num_workers >= 0:
        return max(1, int(cfg.dataset_num_workers))
    if 0 < int(estimated_samples) < 256:
        return 1
    cpu_count = max(1, os.cpu_count() or 1)
    return min(8, max(1, cpu_count // 2))


def _get_dataset_thread_tokenizer(spec: Dict[str, Any]) -> Any:
    cache_key = str(spec["cache_key"])
    if getattr(_DATASET_TOKENIZER_LOCAL, "cache_key", None) == cache_key:
        return _DATASET_TOKENIZER_LOCAL.tokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        spec["tokenizer_source"],
        trust_remote_code=False,
        use_fast=True,
        local_files_only=True,
    )
    prepare_tokenizer(
        tokenizer,
        template_mode=spec["template_mode"],
        force_template=bool(spec["force_template"]),
    )
    ngram_tokens = list(spec.get("ngram_tokens") or [])
    if ngram_tokens:
        tokenizer.add_tokens(ngram_tokens, special_tokens=False)

    _DATASET_TOKENIZER_LOCAL.cache_key = cache_key
    _DATASET_TOKENIZER_LOCAL.tokenizer = tokenizer
    return tokenizer


def _tokenize_dataset_item(
    item: Any, tokenizer: Any, options: Dict[str, Any],
) -> List[Dict[str, Any]]:
    if isinstance(item, (PlainTextSample, str)):
        return tokenize_text_examples(
            item,
            tokenizer,
            int(options["max_seq_length"]),
            chunk_long_texts=bool(options["chunk_long_texts"]),
            text_chunk_overlap=int(options["text_chunk_overlap"]),
            text_chunk_min_tokens=int(options["text_chunk_min_tokens"]),
            append_eos_to_text=bool(options["append_eos_to_text"]),
        )

    dialog_sample = tokenize_example(
        item=item,
        tokenizer=tokenizer,
        max_seq_length=int(options["max_seq_length"]),
        template_mode=str(options["template_mode"]),
        max_history_turns=options.get("max_history_turns"),
        include_prompt_loss=bool(options["include_prompt_loss"]),
    )
    return [dialog_sample] if dialog_sample is not None else []


def _tokenize_dataset_batch(
    items: List[Any], tokenizer_spec: Dict[str, Any], options: Dict[str, Any],
) -> List[List[Dict[str, Any]]]:
    tokenizer = _get_dataset_thread_tokenizer(tokenizer_spec)
    return [_tokenize_dataset_item(item, tokenizer, options) for item in items]


def build_shard_dataset(
    cfg_dict: Dict[str, Any],
    cache_dir_str: str,
    status_writer: Optional[JsonStatusWriter] = None,
    preview_writer: Optional[JsonPreviewWriter] = None,
    estimated_samples: int = 0,
) -> None:
    try:
        cfg = TrainConfig(**cfg_dict)
        cfg.normalize()
        cache_dir = Path(cache_dir_str)
        cache_dir.mkdir(parents=True, exist_ok=True)

        tokenizer_source = cfg.tokenizer_dir or cfg.model_dir
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=False, use_fast=True)
        prepare_tokenizer(tokenizer, template_mode=cfg.template_mode, force_template=bool(cfg.force_template))

        ngram_cfg = NgramConfig(
            use_ngrams=cfg.use_ngrams,
            ngram_max=cfg.ngram_max,
            ngram_top_k=cfg.ngram_top_k,
            ngram_min_chars=cfg.ngram_min_chars,
            ngram_min_words=cfg.ngram_min_words,
            ngram_max_samples=cfg.ngram_max_samples,
            ngram_budgeted=cfg.ngram_budgeted,
            ngram_target_fit=cfg.ngram_target_fit,
            ngram_eval_samples=cfg.ngram_eval_samples,
            ngram_add_batch=cfg.ngram_add_batch,
            ngram_min_count=cfg.ngram_min_count,
            ngram_max_token_chars=cfg.ngram_max_token_chars,
            ngram_max_tokens_per_text=cfg.ngram_max_tokens_per_text,
            template_mode=cfg.template_mode,
            column_name=cfg.column_name,
            mixed_training=cfg.mixed_training,
            mixed_text_column=cfg.mixed_text_column,
            training_phase=cfg.training_phase,
            csv_path=cfg.csv_path,
        )

        def report_ngram_progress(stage: str, current: int, total: int) -> None:
            if SHUTDOWN.stop:
                raise RuntimeError("N-Gram-Analyse durch Stop-Signal abgebrochen.")
            LOGGER.info(
                "NGRAM Fortschritt [%s]: %s/%s (%.1f%%)",
                stage,
                current,
                total,
                (100.0 * current / max(1, total)),
            )
            if status_writer is not None:
                status_writer.write({
                    "running": True,
                    "status": "building_ngrams",
                    "eta": f"N-Gram-Analyse: {current}/{total}",
                    "dataset_progress": {
                        "phase": "ngrams",
                        "seen_samples": int(current),
                        "expected_samples": int(total),
                        "done": False,
                        "updated_at": time.time(),
                    },
                    "dataset_ready": False,
                })

        ngram_state = build_or_load_ngram_state(
            tokenizer=tokenizer,
            cfg=ngram_cfg,
            outdir=cache_dir,
            rebuild=bool(cfg.rebuild_dataset_cache),
            progress_cb=report_ngram_progress,
        )

        shard_size = int(cfg.tokenized_shard_size)
        shard_idx = 0
        global_start = 0
        total_samples = 0
        skipped_samples = 0
        seen_samples = 0
        completed_source_samples = 0
        raw_samples_by_type: Counter[str] = Counter()
        tokenized_segments_by_type: Counter[str] = Counter()
        output_samples_by_type: Counter[str] = Counter()
        output_tokens_by_type: Counter[str] = Counter()
        supervised_tokens_by_type: Counter[str] = Counter()
        assigned_split_counts: Counter[str] = Counter()
        skipped_reasons: Counter[str] = Counter()
        quality_issue_counts: Counter[str] = Counter()
        exact_duplicates_removed = 0
        near_duplicates_found = 0
        chunks_created = 0
        length_histogram: Counter[int] = Counter()
        seen_fingerprints: set[str] = set()
        near_tracker = NearDuplicateTracker(
            cfg.near_duplicate_threshold, max_shingles=cfg.near_duplicate_max_shingles,
        )
        packer = ShortTextPacker(cfg.pack_target_length) if cfg.pack_short_texts else None

        current_samples: List[Dict[str, Any]] = []
        pending_samples: List[Dict[str, Any]] = []

        if cfg.sort_by_length:
            sort_buffer_size = max(512, min(shard_size * 2, 20000))
        else:
            sort_buffer_size = shard_size

        examples_iter = build_examples_stream(cfg)
        build_started_at = time.time()
        effective_dataset_workers = resolve_dataset_num_workers(cfg, estimated_samples)
        tokenize_batch_size = max(1, int(cfg.dataset_tokenize_batch_size))
        tokenizer_spec = {
            "tokenizer_source": tokenizer_source,
            "template_mode": cfg.template_mode,
            "force_template": bool(cfg.force_template),
            "ngram_tokens": list(ngram_state.tokens if ngram_state is not None else []),
        }
        tokenizer_spec["cache_key"] = hashlib.sha256(
            json.dumps(tokenizer_spec, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        tokenize_options = {
            "max_seq_length": int(cfg.max_seq_length),
            "chunk_long_texts": bool(cfg.chunk_long_texts),
            "text_chunk_overlap": int(cfg.text_chunk_overlap),
            "text_chunk_min_tokens": int(cfg.text_chunk_min_tokens),
            "append_eos_to_text": bool(cfg.append_eos_to_text),
            "template_mode": cfg.template_mode,
            "max_history_turns": cfg.max_history_turns,
            "include_prompt_loss": bool(cfg.include_prompt_loss),
        }
        LOGGER.info(
            "Dataset-Tokenisierung | workers=%s batch_size=%s estimated_samples=%s",
            effective_dataset_workers,
            tokenize_batch_size,
            estimated_samples,
        )

        def queue_output_sample(sample: Dict[str, Any]) -> None:
            nonlocal pending_samples, current_samples, shard_idx, global_start, total_samples
            sample_type = str(sample.get("sample_type") or "unknown")
            seq_len = int(sample.get("seq_len") or len(sample["input_ids"]))
            output_samples_by_type[sample_type] += 1
            output_tokens_by_type[sample_type] += seq_len
            supervised_tokens_by_type[sample_type] += sum(
                1 for label in sample.get("labels", [])[1:] if int(label) != -100
            )
            assigned_split_counts[str(sample.get("assigned_split") or "train")] += 1
            length_histogram[int(math.ceil(seq_len / 64.0) * 64)] += 1
            pending_samples.append(sample)
            total_samples += 1

            if len(pending_samples) >= sort_buffer_size:
                pending_samples, current_samples, shard_idx, global_start = _flush_pending_samples(
                    cache_dir=cache_dir,
                    pending_samples=pending_samples,
                    current_samples=current_samples,
                    shard_idx=shard_idx,
                    global_start=global_start,
                    shard_size=shard_size,
                    sort_by_length=bool(cfg.sort_by_length),
                    sort_by_similarity=bool(cfg.sort_by_similarity),
                )

        def write_periodic_progress() -> None:
            if seen_samples % 100 == 0:
                _write_dataset_progress(
                    cache_dir,
                    seen_samples=seen_samples,
                    tokenized_samples=total_samples,
                    skipped_samples=skipped_samples,
                    shard_idx=shard_idx,
                    done=False,
                    status_writer=status_writer,
                    expected_samples=estimated_samples,
                    started_at=build_started_at,
                    num_workers=effective_dataset_workers,
                    processed_samples=completed_source_samples,
                )

        def process_tokenized_result(
            samples: List[Dict[str, Any]],
            near_representative_split_key: Optional[str],
            sample_type: str,
        ) -> None:
            nonlocal skipped_samples, chunks_created, completed_source_samples
            completed_source_samples += 1
            if sample_type == "text":
                chunks_created += max(0, len(samples) - 1)
            if not samples:
                skipped_samples += 1
                skipped_reasons[
                    "oversize_text" if sample_type == "text" else "oversize_dialog"
                ] += 1
                return

            tokenized_segments_by_type[sample_type] += len(samples)
            for sample in samples:
                split_key_for_assignment = (
                    near_representative_split_key or sample.get("split_key")
                )
                is_validation = belongs_to_dataset_split(
                    split="validation",
                    val_split=cfg.val_split,
                    split_seed=cfg.split_seed,
                    global_idx=0,
                    split_key=split_key_for_assignment,
                )
                sample["assigned_split"] = "validation" if is_validation else "train"
                output_samples = packer.add(sample) if packer is not None else [sample]
                for output_sample in output_samples:
                    queue_output_sample(output_sample)

        executor: Optional[ThreadPoolExecutor] = None
        pending_work: List[
            Tuple[Future[List[List[Dict[str, Any]]]], List[Tuple[Any, Optional[str], str]]]
        ] = []
        tokenization_batch: List[Tuple[Any, Optional[str], str]] = []
        tokenizer_fallback_batches = 0

        if effective_dataset_workers > 1:
            executor = ThreadPoolExecutor(
                max_workers=effective_dataset_workers,
                thread_name_prefix="dataset-tokenizer",
            )

        def submit_tokenization_batch() -> None:
            nonlocal tokenization_batch
            if not tokenization_batch:
                return
            metadata = tokenization_batch
            tokenization_batch = []
            if executor is None:
                for item, near_key, sample_type in metadata:
                    samples = _tokenize_dataset_item(item, tokenizer, tokenize_options)
                    process_tokenized_result(samples, near_key, sample_type)
                return
            future = executor.submit(
                _tokenize_dataset_batch,
                [item for item, _near_key, _sample_type in metadata],
                tokenizer_spec,
                tokenize_options,
            )
            pending_work.append((future, metadata))

        def consume_tokenization_batch() -> None:
            nonlocal tokenizer_fallback_batches
            future, metadata = pending_work.pop(0)
            try:
                batch_results = future.result()
            except Exception as exc:
                tokenizer_fallback_batches += 1
                LOGGER.warning(
                    "Parallele Dataset-Tokenisierung fehlgeschlagen; Batch läuft synchron weiter: %s",
                    exc,
                )
                batch_results = [
                    _tokenize_dataset_item(item, tokenizer, tokenize_options)
                    for item, _near_key, _sample_type in metadata
                ]
            for (_item, near_key, sample_type), samples in zip(metadata, batch_results):
                process_tokenized_result(samples, near_key, sample_type)
            _write_dataset_progress(
                cache_dir,
                seen_samples=seen_samples,
                tokenized_samples=total_samples,
                skipped_samples=skipped_samples,
                shard_idx=shard_idx,
                done=False,
                status_writer=status_writer,
                expected_samples=estimated_samples,
                started_at=build_started_at,
                num_workers=effective_dataset_workers,
                processed_samples=completed_source_samples,
            )

        _write_dataset_progress(
            cache_dir,
            seen_samples=0,
            tokenized_samples=0,
            skipped_samples=0,
            shard_idx=0,
            done=False,
            status_writer=status_writer,
            expected_samples=estimated_samples,
            started_at=build_started_at,
            num_workers=effective_dataset_workers,
            processed_samples=completed_source_samples,
        )

        preview_saved = False
        try:
            for item in examples_iter:
                if SHUTDOWN.stop:
                    raise RuntimeError("Dataset-Aufbau durch Stop-Signal abgebrochen.")
                seen_samples += 1
                if not preview_saved:
                    preview, preview_full = format_raw_example_preview(item)
                    preview_payload = {"preview": preview, "preview_full": preview_full}
                    _atomic_write_json(dataset_preview_path(cache_dir), preview_payload)
                    if preview_writer is not None:
                        preview_writer.write(preview, preview_full)
                    preview_saved = True

                sample_type = "dialog" if isinstance(item, StructuredChatSample) else "text"
                raw_samples_by_type[sample_type] += 1
                canonical_text = canonical_example_text(item)
                source_split_key = (
                    item.split_key
                    if isinstance(item, (StructuredChatSample, PlainTextSample))
                    else "text:" + hashlib.sha256(canonical_text.encode("utf-8")).hexdigest()
                )
                normalized_text = normalize_for_dedup(canonical_text)
                fingerprint = hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()
                issues = quality_issues_for_example(item, cfg.quality_min_chars)
                quality_issue_counts.update(issues)

                if cfg.deduplicate_exact and fingerprint in seen_fingerprints:
                    skipped_samples += 1
                    completed_source_samples += 1
                    exact_duplicates_removed += 1
                    skipped_reasons["exact_duplicate"] += 1
                    write_periodic_progress()
                    continue
                seen_fingerprints.add(fingerprint)

                if issues and cfg.quality_filter_mode == "exclude":
                    skipped_samples += 1
                    completed_source_samples += 1
                    skipped_reasons["quality_filter"] += 1
                    write_periodic_progress()
                    continue

                is_near_duplicate = False
                near_representative_split_key: Optional[str] = None
                if cfg.near_duplicate_action != "off":
                    is_near_duplicate, near_representative_split_key = near_tracker.add_and_find(
                        canonical_text,
                        source_split_key,
                    )
                    if is_near_duplicate:
                        near_duplicates_found += 1
                if is_near_duplicate and cfg.near_duplicate_action == "exclude":
                    skipped_samples += 1
                    completed_source_samples += 1
                    skipped_reasons["near_duplicate"] += 1
                    write_periodic_progress()
                    continue

                tokenization_batch.append(
                    (item, near_representative_split_key, sample_type)
                )
                if len(tokenization_batch) >= tokenize_batch_size:
                    submit_tokenization_batch()
                if len(pending_work) >= max(1, effective_dataset_workers * 2):
                    consume_tokenization_batch()
                write_periodic_progress()

            submit_tokenization_batch()
            while pending_work:
                consume_tokenization_batch()
        finally:
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)

        if packer is not None:
            for output_sample in packer.flush():
                queue_output_sample(output_sample)

        if pending_samples:
            pending_samples, current_samples, shard_idx, global_start = _flush_pending_samples(
                cache_dir=cache_dir,
                pending_samples=pending_samples,
                current_samples=current_samples,
                shard_idx=shard_idx,
                global_start=global_start,
                shard_size=shard_size,
                sort_by_length=bool(cfg.sort_by_length),
                sort_by_similarity=bool(cfg.sort_by_similarity),
            )

        if current_samples:
            _write_shard(cache_dir, shard_idx, global_start, current_samples)
            shard_idx += 1

        _write_dataset_progress(
            cache_dir,
            seen_samples=seen_samples,
            tokenized_samples=total_samples,
            skipped_samples=skipped_samples,
            shard_idx=shard_idx,
            done=True,
            status_writer=status_writer,
            expected_samples=estimated_samples,
            started_at=build_started_at,
            num_workers=effective_dataset_workers,
            processed_samples=completed_source_samples,
        )

        meta = {
            "done": True,
            "num_shards": shard_idx,
            "total_samples": total_samples,
            "seen_samples": seen_samples,
            "skipped_samples": skipped_samples,
            "skipped_reasons": dict(sorted(skipped_reasons.items())),
            "raw_samples_by_type": dict(sorted(raw_samples_by_type.items())),
            "tokenized_segments_by_type": dict(sorted(tokenized_segments_by_type.items())),
            "output_samples_by_type": dict(sorted(output_samples_by_type.items())),
            "output_tokens_by_type": dict(sorted(output_tokens_by_type.items())),
            "supervised_tokens_by_type": dict(sorted(supervised_tokens_by_type.items())),
            "assigned_split_counts": dict(sorted(assigned_split_counts.items())),
            "exact_duplicates_removed": exact_duplicates_removed,
            "near_duplicates_found": near_duplicates_found,
            "quality_issues": dict(sorted(quality_issue_counts.items())),
            "chunks_created": chunks_created,
            "packing": {
                "enabled": bool(packer is not None),
                "target_length": int(cfg.pack_target_length),
                "input_segments": int(packer.input_segments if packer is not None else 0),
                "output_sequences": int(packer.output_sequences if packer is not None else 0),
            },
            "length_percentiles_approx": {
                "p50": _histogram_percentile(length_histogram, 0.50),
                "p95": _histogram_percentile(length_histogram, 0.95),
                "p99": _histogram_percentile(length_histogram, 0.99),
            },
            "training_phase": cfg.training_phase,
            "template_mode": cfg.template_mode,
            "max_seq_length": cfg.max_seq_length,
            "max_history_turns": cfg.max_history_turns,
            "strict_whole_turns": True,
            "sort_by_length": bool(cfg.sort_by_length),
            "sort_by_similarity": bool(cfg.sort_by_similarity),
            "sort_buffer_size": sort_buffer_size,
            "use_ngrams": bool(cfg.use_ngrams),
            "ngram_summary": ngram_summary_text(ngram_state),
            "ngram_selected_count": int((ngram_state.stats or {}).get("selected_count", 0)) if ngram_state else 0,
            "dataset_build": {
                "num_workers": int(effective_dataset_workers),
                "tokenize_batch_size": int(tokenize_batch_size),
                "fallback_batches": int(tokenizer_fallback_batches),
                "elapsed_seconds": float(time.time() - build_started_at),
                "samples_per_second": float(
                    seen_samples / max(1e-9, time.time() - build_started_at)
                ),
            },
        }
        _atomic_write_json(dataset_meta_path(cache_dir), meta)
        _atomic_write_json(dataset_ready_path(cache_dir), {"done": True})

    except Exception as e:
        dataset_error_path(Path(cache_dir_str)).write_text(
            f"{e.__class__.__name__}: {e}\n\n{traceback.format_exc()}",
            encoding="utf-8",
        )
        raise


def prepare_shard_dataset(
    cfg: TrainConfig,
    cache_dir: Path,
    ctx: DistContext,
    status_writer: Optional[JsonStatusWriter] = None,
    preview_writer: Optional[JsonPreviewWriter] = None,
    estimated_samples: int = 0,
) -> None:
    if not ctx.is_main:
        return

    if cfg.rebuild_dataset_cache and cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)

    cache_dir.mkdir(parents=True, exist_ok=True)

    done_file = dataset_ready_path(cache_dir)
    error_file = dataset_error_path(cache_dir)
    progress_file = dataset_progress_path(cache_dir)

    if error_file.exists():
        error_file.unlink(missing_ok=True)

    if cfg.use_dataset_cache and done_file.exists() and not cfg.rebuild_dataset_cache:
        LOGGER.info("Vollständig gebautes Dataset bereits vorhanden: %s", cache_dir)
        if status_writer is not None:
            cached_progress = read_json_if_exists(progress_file) or {}
            cached_progress.update({
                "phase": "cache",
                "cache_hit": True,
                "done": True,
                "updated_at": time.time(),
            })
            _atomic_write_json(progress_file, cached_progress)
            cached_meta = read_json_if_exists(dataset_meta_path(cache_dir)) or {}
            status_writer.write({
                "running": True,
                "status": "dataset_ready",
                "eta": "Dataset-Cache vollständig geladen",
                "dataset_progress": cached_progress,
                "dataset_meta": cached_meta,
                "dataset_ready": True,
                "total_samples_real": cached_meta.get("total_samples"),
                "skipped_samples": cached_meta.get("skipped_samples"),
            })
        cached_preview = read_json_if_exists(dataset_preview_path(cache_dir)) or {}
        if preview_writer is not None and cached_preview:
            preview_writer.write(
                str(cached_preview.get("preview") or ""),
                str(cached_preview.get("preview_full") or ""),
            )
        return

    for p in cache_dir.glob("shard_*.pkl"):
        p.unlink(missing_ok=True)
    done_file.unlink(missing_ok=True)
    dataset_meta_path(cache_dir).unlink(missing_ok=True)
    dataset_preview_path(cache_dir).unlink(missing_ok=True)
    progress_file.unlink(missing_ok=True)

    LOGGER.info("Baue tokenisiertes Dataset vollständig vor dem Training: %s", cache_dir)
    build_shard_dataset(
        cfg.__dict__.copy(),
        str(cache_dir),
        status_writer=status_writer,
        preview_writer=preview_writer,
        estimated_samples=estimated_samples,
    )
    LOGGER.info("Dataset vollständig gebaut: %s", cache_dir)


def wait_for_dataset_ready(cache_dir: Path, poll_sec: float = 0.5) -> None:
    while True:
        error_file = dataset_error_path(cache_dir)
        if error_file.exists():
            raise RuntimeError(error_file.read_text(encoding="utf-8"))
        if dataset_ready_path(cache_dir).exists():
            meta = read_json_if_exists(dataset_meta_path(cache_dir)) or {}
            if int(meta.get("num_shards", 0)) <= 0 or not shard_file_path(cache_dir, 0).exists():
                raise RuntimeError(
                    "Dataset-Aufbau abgeschlossen, aber kein Trainings-Shard erzeugt. "
                    f"skipped_samples={meta.get('skipped_samples', 0)}"
                )
            return
        if SHUTDOWN.stop:
            raise RuntimeError("Dataset-Aufbau durch Stop-Signal abgebrochen.")
        time.sleep(poll_sec)


def read_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return None


def cuda_memory_snapshot(device: torch.device) -> Optional[Dict[str, Any]]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    try:
        idx = device.index if device.index is not None else torch.cuda.current_device()
        return {
            "device_index": int(idx),
            "allocated_mb": round(torch.cuda.memory_allocated(idx) / (1024 * 1024), 2),
            "reserved_mb": round(torch.cuda.memory_reserved(idx) / (1024 * 1024), 2),
            "max_allocated_mb": round(torch.cuda.max_memory_allocated(idx) / (1024 * 1024), 2),
            "max_reserved_mb": round(torch.cuda.max_memory_reserved(idx) / (1024 * 1024), 2),
        }
    except Exception:
        return None


def hardware_training_profile(cfg: TrainConfig, ctx: DistContext) -> Dict[str, Any]:
    _, uses_fp16, uses_bf16 = pick_precision(cfg, ctx.device)
    profile: Dict[str, Any] = {
        "device": str(ctx.device),
        "world_size": ctx.world_size,
        "precision_requested": cfg.precision_mode,
        "precision_effective": "fp16" if uses_fp16 else ("bf16" if uses_bf16 else "fp32"),
        "tf32_enabled": bool(cfg.allow_tf32),
    }
    if ctx.device.type == "cuda":
        props = torch.cuda.get_device_properties(ctx.device)
        profile.update({
            "gpu_name": props.name,
            "compute_capability": f"{props.major}.{props.minor}",
            "memory_gb": round(props.total_memory / (1024 ** 3), 2),
            "tensor_core_fp16": props.major >= 7,
            "native_bf16": props.major >= 8,
            "native_tf32": props.major >= 8,
        })
        if props.major == 7 and props.minor == 0:
            profile["hardware_family"] = "NVIDIA Volta/V100"
            profile["recommended_precision"] = "fp16"
    return profile


def maybe_log_cuda_memory(
    *,
    cfg: TrainConfig,
    ctx: DistContext,
    global_step: int,
    prefix: str,
) -> Optional[Dict[str, Any]]:
    if not cfg.log_cuda_memory:
        return None
    if ctx.device.type != "cuda":
        return None
    if global_step < 0:
        return None
    if (global_step % cfg.cuda_memory_log_interval_steps) != 0:
        return None

    snap = cuda_memory_snapshot(ctx.device)
    if snap and ctx.is_main:
        LOGGER.info(
            "%s CUDA memory | dev=%s allocated=%s MB reserved=%s MB max_allocated=%s MB max_reserved=%s MB",
            prefix,
            snap["device_index"],
            snap["allocated_mb"],
            snap["reserved_mb"],
            snap["max_allocated_mb"],
            snap["max_reserved_mb"],
        )
    return snap


def maybe_empty_cuda_cache(cfg: TrainConfig, ctx: DistContext, global_step: int) -> None:
    if cfg.cuda_empty_cache_interval_steps <= 0:
        return
    if ctx.device.type != "cuda":
        return
    if global_step <= 0:
        return
    if (global_step % cfg.cuda_empty_cache_interval_steps) != 0:
        return
    try:
        torch.cuda.empty_cache()
        if ctx.is_main:
            LOGGER.info("torch.cuda.empty_cache() ausgeführt bei step=%s", global_step)
    except Exception as exc:
        if ctx.is_main:
            LOGGER.warning("empty_cache fehlgeschlagen bei step=%s: %s", global_step, exc)


def belongs_to_dataset_split(
    *, split: str, val_split: float, split_seed: int,
    global_idx: int, split_key: Optional[str] = None,
) -> bool:
    split = str(split).lower().strip()
    val_split = min(0.5, max(0.0, float(val_split)))
    if val_split <= 0.0:
        return split != "validation"
    stable_key = split_key or f"sample:{int(global_idx)}"
    payload = f"{int(split_seed)}:{stable_key}".encode("utf-8")
    bucket = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") / float(2**64)
    is_validation = bucket < val_split
    return is_validation if split == "validation" else not is_validation


def tokenized_sample_belongs_to_split(
    item: Dict[str, Any],
    *,
    split: str,
    val_split: float,
    split_seed: int,
    global_idx: int,
) -> bool:
    assigned_split = str(item.get("assigned_split") or "").strip().lower()
    if assigned_split in {"train", "validation"}:
        return assigned_split == str(split).lower().strip()
    return belongs_to_dataset_split(
        split=split,
        val_split=val_split,
        split_seed=split_seed,
        global_idx=global_idx,
        split_key=item.get("split_key"),
    )


class TokenizedShardIterableDataset(IterableDataset):
    def __init__(
        self,
        cache_dir: Path,
        rank: int,
        world_size: int,
        sort_by_length: bool = True,
        epoch: int = 0,
        split: str = "train",
        val_split: float = 0.0,
        split_seed: int = 42,
        sample_type_filter: Optional[str] = None,
    ):
        super().__init__()
        self.cache_dir = cache_dir
        self.rank = rank
        self.world_size = world_size
        self.sort_by_length = sort_by_length
        self.epoch = epoch
        self.split = str(split).lower().strip()
        self.val_split = min(0.5, max(0.0, float(val_split)))
        self.split_seed = int(split_seed)
        self.sample_type_filter = (
            str(sample_type_filter).strip().lower() if sample_type_filter else None
        )

    def _belongs_to_split(self, global_idx: int, split_key: Optional[str] = None) -> bool:
        return belongs_to_dataset_split(
            split=self.split,
            val_split=self.val_split,
            split_seed=self.split_seed,
            global_idx=global_idx,
            split_key=split_key,
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = int(worker_info.id) if worker_info is not None else 0
        num_workers = int(worker_info.num_workers) if worker_info is not None else 1
        combined_world_size = max(1, int(self.world_size) * max(1, num_workers))
        combined_rank = int(self.rank) + worker_id * max(1, int(self.world_size))

        if dataset_error_path(self.cache_dir).exists():
            raise RuntimeError(dataset_error_path(self.cache_dir).read_text(encoding="utf-8"))
        if not dataset_ready_path(self.cache_dir).exists():
            raise RuntimeError("Tokenisiertes Dataset ist noch nicht vollständig gebaut.")

        for path in sorted(self.cache_dir.glob("shard_*.pkl")):
            with open(path, "rb") as f:
                payload = pickle.load(f)

            samples = payload["samples"]
            global_start = int(payload["global_start"])

            order = list(range(len(samples)))

            for local_idx in order:
                global_idx = global_start + local_idx
                item = samples[local_idx]
                if not tokenized_sample_belongs_to_split(
                    item,
                    split=self.split,
                    val_split=self.val_split,
                    split_seed=self.split_seed,
                    global_idx=global_idx,
                ):
                    continue
                if (
                    self.sample_type_filter
                    and str(item.get("sample_type") or "unknown").lower()
                    != self.sample_type_filter
                ):
                    continue
                if (global_idx % combined_world_size) != combined_rank:
                    continue
                yield {
                    "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
                    "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
                    "labels": torch.tensor(item["labels"], dtype=torch.long),
                }



@dataclass(frozen=True)
class SampleRef:
    shard_idx: int
    local_idx: int
    global_idx: int
    seq_len: int
    sample_type: str = "unknown"
    repeat_index: int = 0


def apply_token_mixture_weights(
    refs: List[SampleRef], cfg: TrainConfig,
) -> Tuple[List[SampleRef], Dict[str, Any]]:
    raw_tokens: Counter[str] = Counter()
    groups: Dict[str, List[SampleRef]] = defaultdict(list)
    for ref in refs:
        groups[ref.sample_type].append(ref)
        raw_tokens[ref.sample_type] += ref.seq_len

    stats: Dict[str, Any] = {
        "enabled": False,
        "requested_weights": {
            "text": float(cfg.text_token_weight),
            "dialog": float(cfg.dialog_token_weight),
        },
        "raw_tokens": dict(sorted(raw_tokens.items())),
        "raw_samples": {name: len(values) for name, values in sorted(groups.items())},
    }
    should_mix = bool(
        cfg.mixed_training
        and cfg.training_phase in {"custom", "mixed"}
        and groups.get("text")
        and groups.get("dialog")
    )
    if not should_mix:
        stats["effective_tokens"] = dict(sorted(raw_tokens.items()))
        stats["effective_samples"] = {name: len(values) for name, values in sorted(groups.items())}
        return refs, stats

    weights = {
        "text": max(0.0, float(cfg.text_token_weight)),
        "dialog": max(0.0, float(cfg.dialog_token_weight)),
    }
    weight_sum = sum(weights.values())
    weights = {name: value / weight_sum for name, value in weights.items()}
    if any(value <= 0.0 for value in weights.values()):
        filtered = [ref for ref in refs if weights.get(ref.sample_type, 1.0) > 0.0]
        token_counts: Counter[str] = Counter()
        for ref in filtered:
            token_counts[ref.sample_type] += ref.seq_len
        stats.update({
            "enabled": True,
            "effective_tokens": dict(sorted(token_counts.items())),
            "effective_samples": dict(Counter(ref.sample_type for ref in filtered)),
        })
        return filtered, stats

    target_total = max(
        raw_tokens[name] / max(1e-12, weights[name])
        for name in ("text", "dialog")
    )
    weighted_refs = list(refs)
    effective_tokens = Counter(raw_tokens)
    effective_samples: Counter[str] = Counter(ref.sample_type for ref in refs)
    oversampled: Counter[str] = Counter()
    for name in ("text", "dialog"):
        desired_tokens = int(math.ceil(target_total * weights[name]))
        capped_tokens = int(raw_tokens[name] * cfg.max_mixture_oversample)
        target_tokens = min(desired_tokens, max(raw_tokens[name], capped_tokens))
        group = groups[name]
        index = 0
        while effective_tokens[name] < target_tokens and group:
            ref = group[index % len(group)]
            weighted_refs.append(SampleRef(
                shard_idx=ref.shard_idx,
                local_idx=ref.local_idx,
                global_idx=ref.global_idx,
                seq_len=ref.seq_len,
                sample_type=ref.sample_type,
                repeat_index=1 + index // len(group),
            ))
            effective_tokens[name] += ref.seq_len
            effective_samples[name] += 1
            oversampled[name] += 1
            index += 1

    stats.update({
        "enabled": True,
        "normalized_weights": weights,
        "effective_tokens": dict(sorted(effective_tokens.items())),
        "effective_samples": dict(sorted(effective_samples.items())),
        "oversampled_samples": dict(sorted(oversampled.items())),
    })
    return weighted_refs, stats


def build_token_batch_plan(
    cache_dir: Path, cfg: TrainConfig, ctx: DistContext,
) -> Tuple[List[List[SampleRef]], Dict[str, Any]]:
    refs: List[SampleRef] = []
    for path in sorted(cache_dir.glob("shard_*.pkl")):
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        shard_idx = int(payload.get("shard_idx", int(path.stem.split("_")[-1])))
        global_start = int(payload["global_start"])
        for local_idx, item in enumerate(payload["samples"]):
            global_idx = global_start + local_idx
            if not tokenized_sample_belongs_to_split(
                item,
                split="train",
                val_split=cfg.val_split,
                split_seed=cfg.split_seed,
                global_idx=global_idx,
            ):
                continue
            refs.append(SampleRef(
                shard_idx=shard_idx,
                local_idx=local_idx,
                global_idx=global_idx,
                seq_len=int(item.get("seq_len") or len(item["input_ids"])),
                sample_type=str(item.get("sample_type") or "unknown"),
            ))

    if not refs:
        raise RuntimeError("Der Trainingssplit enthält keine tokenisierten Samples.")

    raw_train_samples = len(refs)
    refs, mixture_stats = apply_token_mixture_weights(refs, cfg)
    if not refs:
        raise RuntimeError("Die Datengewichtung hat alle Trainingssamples ausgeschlossen.")

    if cfg.sort_by_length:
        refs.sort(key=lambda ref: (ref.seq_len, ref.repeat_index, ref.global_idx))

    token_budget = int(cfg.max_tokens_per_batch)
    if token_budget <= 0:
        token_budget = int(cfg.max_seq_length * cfg.per_device_train_batch_size)
    token_budget = max(int(math.ceil(cfg.max_seq_length / 8) * 8), token_budget)

    if cfg.dynamic_token_batching:
        max_samples = int(
            cfg.max_samples_per_batch
            or max(8, cfg.per_device_train_batch_size * 32)
        )
    else:
        max_samples = int(cfg.per_device_train_batch_size)

    batches: List[List[SampleRef]] = []
    current: List[SampleRef] = []
    current_max_len = 0
    for ref in refs:
        padded_seq_len = int(math.ceil(ref.seq_len / 8) * 8)
        candidate_max = max(current_max_len, padded_seq_len)
        candidate_count = len(current) + 1
        exceeds_tokens = (
            cfg.dynamic_token_batching
            and candidate_max * candidate_count > token_budget
        )
        exceeds_samples = candidate_count > max_samples
        if current and (exceeds_tokens or exceeds_samples):
            batches.append(current)
            current = []
            current_max_len = 0
        current.append(ref)
        current_max_len = max(current_max_len, padded_seq_len)
    if current:
        batches.append(current)

    original_batches = len(batches)
    padded_batches = 0
    dropped_batches = 0
    alignment = max(1, ctx.world_size * cfg.gradient_accumulation_steps)
    if ctx.is_distributed and batches and len(batches) % alignment:
        if cfg.pad_batches_for_ddp or len(batches) < alignment:
            target_count = int(math.ceil(len(batches) / alignment) * alignment)
            source = list(batches)
            while len(batches) < target_count:
                batches.append(list(source[len(batches) % len(source)]))
                padded_batches += 1
        else:
            target_count = int(len(batches) // alignment) * alignment
            dropped_batches = len(batches) - target_count
            batches = batches[:target_count]

    efficiency_batches = batches[:min(original_batches, len(batches))]
    actual_tokens = sum(ref.seq_len for batch in efficiency_batches for ref in batch)
    padded_tokens = sum(
        int(math.ceil(max(ref.seq_len for ref in batch) / 8) * 8) * len(batch)
        for batch in efficiency_batches if batch
    )
    stats = {
        "train_samples": len(refs),
        "train_samples_raw": raw_train_samples,
        "global_batches": len(batches),
        "original_batches": original_batches,
        "ddp_padding_batches": padded_batches,
        "ddp_dropped_batches": dropped_batches,
        "batches_per_rank": int(math.ceil(len(batches) / max(1, ctx.world_size))),
        "max_tokens_per_batch": token_budget,
        "max_samples_per_batch": max_samples,
        "dynamic_token_batching": bool(cfg.dynamic_token_batching),
        "padding_efficiency": float(actual_tokens / max(1, padded_tokens)),
        "training_tokens_per_epoch": int(actual_tokens),
        "mixture": mixture_stats,
    }
    return batches, stats


class PlannedBatchIterableDataset(IterableDataset):
    def __init__(
        self, cache_dir: Path, global_batches: List[List[SampleRef]],
        rank: int, world_size: int, seed: int, shuffle: bool,
        original_batch_count: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.cache_dir = cache_dir
        self.global_batches = global_batches
        self.rank = int(rank)
        self.world_size = max(1, int(world_size))
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.original_batch_count = min(
            len(global_batches),
            max(1, int(original_batch_count or len(global_batches))),
        )
        self._epoch = mp.Value("i", 0)

    def set_epoch(self, epoch: int) -> None:
        self._epoch.value = int(epoch)

    def __iter__(self):
        epoch = int(self._epoch.value)
        order = list(range(self.original_batch_count))
        if self.shuffle:
            random.Random(self.seed + epoch).shuffle(order)
        padding_count = len(self.global_batches) - len(order)
        if padding_count > 0:
            padding_source = list(order)
            offset = epoch % max(1, len(padding_source))
            for index in range(padding_count):
                order.append(padding_source[(offset + index) % len(padding_source)])
        local_order = order[self.rank::self.world_size]

        worker_info = get_worker_info()
        if worker_info is not None:
            local_order = local_order[int(worker_info.id)::int(worker_info.num_workers)]

        shard_cache: Dict[int, Dict[str, Any]] = {}
        for batch_idx in local_order:
            features: List[Dict[str, torch.Tensor]] = []
            batch_refs = sorted(
                self.global_batches[batch_idx],
                key=lambda ref: (ref.shard_idx, ref.local_idx),
            )
            for ref in batch_refs:
                if ref.shard_idx not in shard_cache:
                    with open(shard_file_path(self.cache_dir, ref.shard_idx), "rb") as handle:
                        shard_cache[ref.shard_idx] = pickle.load(handle)
                    while len(shard_cache) > 2:
                        shard_cache.pop(next(iter(shard_cache)))
                item = shard_cache[ref.shard_idx]["samples"][ref.local_idx]
                features.append({
                    "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
                    "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
                    "labels": torch.tensor(item["labels"], dtype=torch.long),
                })
            if features:
                yield features



class DataCollator:
    def __init__(
        self,
        pad_token_id: int,
        pad_to_multiple_of: int = 8,
        fixed_length: Optional[int] = None,
    ):
        self.pad_token_id = int(pad_token_id)
        self.pad_to_multiple_of = int(pad_to_multiple_of)
        self.fixed_length = int(fixed_length) if fixed_length and int(fixed_length) > 0 else None

    def __call__(self, features: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        max_len = self.fixed_length or max(int(x["input_ids"].numel()) for x in features)
        if self.pad_to_multiple_of > 1:
            max_len = int(math.ceil(max_len / self.pad_to_multiple_of) * self.pad_to_multiple_of)

        def _pad(x: torch.Tensor, value: int) -> torch.Tensor:
            pad_len = max_len - int(x.numel())
            if pad_len <= 0:
                return x
            return torch.nn.functional.pad(x, (0, pad_len), value=value)

        return {
            "input_ids": torch.stack([_pad(x["input_ids"], self.pad_token_id) for x in features], dim=0),
            "attention_mask": torch.stack([_pad(x["attention_mask"], 0) for x in features], dim=0),
            "labels": torch.stack([_pad(x["labels"], -100) for x in features], dim=0),
        }


def pick_precision(cfg: TrainConfig, device: torch.device) -> Tuple[Optional[torch.dtype], bool, bool]:
    want = (cfg.precision_mode or "auto").lower().strip()
    if device.type != "cuda":
        return None, False, False
    if want == "fp32":
        return None, False, False
    if want == "bf16":
        ok = torch.cuda.is_bf16_supported()
        return (torch.bfloat16 if ok else None), False, ok
    if want == "fp16":
        return torch.float16, True, False
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16, False, True
    return torch.float16, True, False


def apply_training_mode(model: nn.Module, cfg: TrainConfig) -> nn.Module:
    mode = (cfg.train_mode or "full").lower().strip()
    if mode == "full":
        for p in model.parameters():
            p.requires_grad = True
        return model

    if mode != "lora":
        raise ValueError(f"Unbekannter train_mode: {cfg.train_mode}")

    if not _PEFT_AVAILABLE:
        raise RuntimeError("LoRA angefordert, aber 'peft' ist nicht installiert.")

    target_modules = list(cfg.lora_target_modules or [])
    if not target_modules:
        for name, module in model.named_modules():
            leaf = name.split(".")[-1].lower()
            if isinstance(module, nn.Linear) and leaf in {
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
                "query", "key", "value", "dense",
                "fc1", "fc2", "wq", "wk", "wv", "wo",
            }:
                target_modules.append(name.split(".")[-1])
        target_modules = sorted(set(target_modules))
    if not target_modules:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    peft_cfg = LoraConfig(
        r=int(cfg.lora_r),
        lora_alpha=int(cfg.lora_alpha),
        lora_dropout=float(cfg.lora_dropout),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )
    model = get_peft_model(model, peft_cfg)
    try:
        model.print_trainable_parameters()
    except Exception:
        pass
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    LOGGER.info(
        "LoRA aktiv | r=%s alpha=%s dropout=%s targets=%s | trainable=%s/%s (%.4f%%)",
        cfg.lora_r, cfg.lora_alpha, cfg.lora_dropout, ",".join(target_modules),
        trainable, total, 100.0 * trainable / max(1, total),
    )
    return model


def build_model_and_tokenizer(
    cfg: TrainConfig, ctx: DistContext, ngram_cache_dir: Optional[Path] = None,
):
    resume_dir = Path(cfg.resume).expanduser().resolve() if cfg.resume else None
    tokenizer_source = (
        str(resume_dir)
        if resume_dir and resume_dir.exists()
        else (cfg.tokenizer_dir or cfg.model_dir)
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=False, use_fast=True)
    need_resize = prepare_tokenizer(
        tokenizer,
        template_mode=cfg.template_mode,
        force_template=bool(cfg.force_template),
    )

    ngram_cfg = NgramConfig(
        use_ngrams=cfg.use_ngrams,
        ngram_max=cfg.ngram_max,
        ngram_top_k=cfg.ngram_top_k,
        ngram_min_chars=cfg.ngram_min_chars,
        ngram_min_words=cfg.ngram_min_words,
        ngram_max_samples=cfg.ngram_max_samples,
        ngram_budgeted=cfg.ngram_budgeted,
        ngram_target_fit=cfg.ngram_target_fit,
        ngram_eval_samples=cfg.ngram_eval_samples,
        ngram_add_batch=cfg.ngram_add_batch,
        ngram_min_count=cfg.ngram_min_count,
        ngram_max_token_chars=cfg.ngram_max_token_chars,
        ngram_max_tokens_per_text=cfg.ngram_max_tokens_per_text,
        template_mode=cfg.template_mode,
        column_name=cfg.column_name,
        mixed_training=cfg.mixed_training,
        mixed_text_column=cfg.mixed_text_column,
        training_phase=cfg.training_phase,
        csv_path=cfg.csv_path,
    )

    ngram_state = build_or_load_ngram_state(
        tokenizer=tokenizer,
        cfg=ngram_cfg,
        outdir=(
            Path(ngram_cache_dir)
            if ngram_cache_dir is not None
            else Path(cfg.output_dir or cfg.save_dir or "./training_outputs/worker_run")
        ),
        rebuild=False if ngram_cache_dir is not None else bool(cfg.rebuild_dataset_cache),
        progress_cb=lambda stage, current, total: LOGGER.info(
            "NGRAM Fortschritt [%s]: %s/%s (%.1f%%)",
            stage,
            current,
            total,
            (100.0 * current / max(1, total)),
        ),
    )
    LOGGER.info(ngram_summary_text(ngram_state))

    load_dtype, fp16, bf16 = pick_precision(cfg, ctx.device)
    LOGGER.info("Precision: load_dtype=%s fp16=%s bf16=%s", load_dtype, fp16, bf16)

    train_from_scratch = bool(cfg.train_from_scratch)
    mode = (cfg.train_mode or "full").lower().strip()
    if train_from_scratch and mode == "lora":
        raise ValueError("train_from_scratch ist nicht mit LoRA kompatibel. Bitte train_mode='full' verwenden.")

    if train_from_scratch:
        model_config = AutoConfig.from_pretrained(cfg.model_dir, trust_remote_code=False)

        scratch_overrides = {
            "hidden_size": cfg.scratch_hidden_size,
            "num_hidden_layers": cfg.scratch_num_hidden_layers,
            "num_attention_heads": cfg.scratch_num_attention_heads,
            "intermediate_size": cfg.scratch_intermediate_size,
            "num_key_value_heads": cfg.scratch_num_key_value_heads,
            "max_position_embeddings": cfg.scratch_max_position_embeddings,
        }
        applied_overrides = {}
        for key, value in scratch_overrides.items():
            if value is None:
                continue
            if hasattr(model_config, key):
                setattr(model_config, key, int(value))
                applied_overrides[key] = int(value)
            else:
                LOGGER.warning("Scratch override ignoriert (Config kennt Feld nicht): %s", key)

        if hasattr(model_config, "max_position_embeddings"):
            current_positions = int(getattr(model_config, "max_position_embeddings", 0) or 0)
            if current_positions < cfg.max_seq_length:
                model_config.max_position_embeddings = int(cfg.max_seq_length)
                applied_overrides["max_position_embeddings"] = int(cfg.max_seq_length)
                LOGGER.info(
                    "Scratch-Positions automatisch auf max_seq_length erweitert: %s -> %s",
                    current_positions,
                    cfg.max_seq_length,
                )

        hidden_size = int(getattr(model_config, "hidden_size", 0) or 0)
        num_attention_heads = int(getattr(model_config, "num_attention_heads", 0) or 0)
        if hidden_size > 0 and num_attention_heads > 0 and (hidden_size % num_attention_heads) != 0:
            raise ValueError(
                f"Ungültige Scratch-Config: hidden_size ({hidden_size}) muss durch "
                f"num_attention_heads ({num_attention_heads}) teilbar sein."
            )

        num_key_value_heads = int(getattr(model_config, "num_key_value_heads", 0) or 0)
        if num_key_value_heads > 0 and num_attention_heads > 0 and num_attention_heads % num_key_value_heads != 0:
            raise ValueError(
                f"Ungültige Scratch-Config: num_attention_heads ({num_attention_heads}) muss durch "
                f"num_key_value_heads ({num_key_value_heads}) teilbar sein."
            )

        if load_dtype is not None:
            try:
                model_config.torch_dtype = load_dtype
            except Exception:
                pass
        model = AutoModelForCausalLM.from_config(
            model_config,
            trust_remote_code=False,
            attn_implementation="sdpa",
        )
        LOGGER.info(
            "Model init: scratch from config | source=%s | overrides=%s",
            cfg.model_dir,
            json.dumps(applied_overrides, ensure_ascii=False, sort_keys=True),
        )
    else:
        model_source = cfg.model_dir
        # Full checkpoints contain a complete model. LoRA checkpoints contain only
        # the adapter and must still be attached to the configured base model.
        if resume_dir and (resume_dir / "config.json").exists() and not (resume_dir / "adapter_config.json").exists():
            model_source = str(resume_dir)
        model = AutoModelForCausalLM.from_pretrained(
            model_source,
            trust_remote_code=False,
            torch_dtype=load_dtype,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        )
        LOGGER.info("Model init: pretrained weights | source=%s", model_source)

    if need_resize or model.get_input_embeddings().weight.shape[0] != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))

    if ctx.device.type in {"cpu", "mps"} or (cfg.precision_mode or "").lower() == "fp32":
        model = model.to(torch.float32)

    if hasattr(model, "config"):
        model.config.use_cache = False

    if resume_dir and (resume_dir / "adapter_config.json").exists():
        if not _PEFT_AVAILABLE:
            raise RuntimeError("LoRA-Resume angefordert, aber 'peft' ist nicht installiert.")
        model = PeftModel.from_pretrained(model, str(resume_dir), is_trainable=True)
        LOGGER.info("LoRA-Adapter fuer Resume geladen: %s", resume_dir)
    else:
        model = apply_training_mode(model, cfg)

    if cfg.gradient_checkpointing:
        _enable_gradient_checkpointing(model)

    if cfg.neftune_noise_alpha > 0.0:
        _enable_neftune(model, cfg.neftune_noise_alpha)

    model.to(ctx.device)
    return model, tokenizer, fp16, bf16, ngram_state


def _enable_gradient_checkpointing(model: nn.Module) -> None:
    if not hasattr(model, "gradient_checkpointing_enable"):
        LOGGER.info("Gradient Checkpointing nicht verfügbar.")
        return

    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        LOGGER.info("Gradient Checkpointing: ON (non-reentrant)")
    except TypeError:
        try:
            model.gradient_checkpointing_enable()
            LOGGER.info("Gradient Checkpointing: ON (legacy API)")
        except Exception as e:
            LOGGER.warning("Gradient Checkpointing konnte nicht aktiviert werden: %s", e)

    if hasattr(model, "enable_input_require_grads"):
        try:
            model.enable_input_require_grads()
        except Exception:
            pass


def _enable_neftune(model: nn.Module, noise_alpha: float) -> None:
    embeddings = model.get_input_embeddings()

    def _noise_hook(module: nn.Module, _inputs: Tuple[Any, ...], output: torch.Tensor):
        if not module.training or not torch.is_tensor(output):
            return output
        dims = max(1, int(output.shape[-2]) * int(output.shape[-1]))
        magnitude = float(noise_alpha) / math.sqrt(float(dims))
        noise = torch.empty_like(output).uniform_(-magnitude, magnitude)
        return output + noise

    handle = embeddings.register_forward_hook(_noise_hook)
    setattr(model, "_matelix_neftune_hook", handle)
    LOGGER.info("NEFTune aktiv | noise_alpha=%s", noise_alpha)


def build_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    no_decay_terms = ("bias", "LayerNorm.weight", "layernorm.weight", "norm.weight", "ln_f.weight")
    named_params = list(unwrap_model(model).named_parameters())
    decay = [p for n, p in named_params if p.requires_grad and not any(x in n for x in no_decay_terms)]
    no_decay = [p for n, p in named_params if p.requires_grad and any(x in n for x in no_decay_terms)]

    use_fused = bool(torch.cuda.is_available())
    try:
        return AdamW(
            [
                {"params": decay, "weight_decay": cfg.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=cfg.learning_rate,
            fused=use_fused,
        )
    except TypeError:
        return AdamW(
            [
                {"params": decay, "weight_decay": cfg.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=cfg.learning_rate,
        )


class FixedLRScheduler:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        base_lr: float,
        schedule: str,
        total_steps: int,
        warmup_steps: int = 0,
        min_lr_ratio: float = 0.0,
        lr_decay_factor: float = 1.0,
    ) -> None:
        self.optimizer = optimizer
        self.base_lr = float(base_lr)
        self.schedule = (schedule or "cosine").lower().strip()
        self.total_steps = max(1, int(total_steps))
        self.warmup_steps = max(0, int(warmup_steps))
        self.min_lr_ratio = min(1.0, max(0.0, float(min_lr_ratio)))
        self.lr_decay_factor = max(0.01, float(lr_decay_factor))
        self.last_lr = self.base_lr
        self._apply_lr(self.get_lr_for_step(0))

    def _apply_lr(self, lr: float) -> float:
        lr = float(max(0.0, lr))
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.last_lr = lr
        return lr

    def get_lr_scale(self, step: int) -> float:
        step = max(0, int(step))
        warmup_steps = max(0, min(self.warmup_steps, self.total_steps - 1 if self.total_steps > 1 else 0))

        if warmup_steps > 0 and step < warmup_steps:
            return max(1e-12, float(step + 1) / float(warmup_steps))

        if self.total_steps <= warmup_steps:
            return 1.0

        decay_span = max(1, int(math.ceil((self.total_steps - warmup_steps) * self.lr_decay_factor)))
        decay_step = max(0, step - warmup_steps)
        progress = min(1.0, decay_step / float(decay_span))

        if self.schedule == "linear":
            value = 1.0 - progress
        elif self.schedule == "cosine":
            value = 0.5 * (1.0 + math.cos(math.pi * progress))
        else:
            value = 1.0

        return max(self.min_lr_ratio, float(value))

    def get_lr_for_step(self, step: int) -> float:
        return self.base_lr * self.get_lr_scale(step)

    def step(self, global_step: int) -> float:
        return self._apply_lr(self.get_lr_for_step(global_step))

    def state_dict(self) -> Dict[str, Any]:
        return {
            "base_lr": self.base_lr,
            "schedule": self.schedule,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "min_lr_ratio": self.min_lr_ratio,
            "lr_decay_factor": self.lr_decay_factor,
            "last_lr": self.last_lr,
            "scheduler_type": "fixed_batch_plan",
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore scheduler state without replacing the current optimizer."""
        for key in (
            "base_lr", "schedule", "total_steps", "warmup_steps", "min_lr_ratio",
            "lr_decay_factor", "last_lr",
        ):
            if key in state:
                setattr(self, key, state[key])
        self.total_steps = max(1, int(self.total_steps))
        self.warmup_steps = max(0, int(self.warmup_steps))
        self._apply_lr(float(self.last_lr))


def estimate_total_steps_from_batch_plan(
    global_batches: int, cfg: TrainConfig, ctx: DistContext,
) -> int:
    local_batches = int(math.ceil(max(1, global_batches) / max(1, ctx.world_size)))
    updates_per_epoch = max(1, int(math.ceil(local_batches / cfg.gradient_accumulation_steps)))
    if cfg.max_steps is not None:
        return max(1, int(cfg.max_steps))
    return max(1, int(math.ceil(cfg.num_train_epochs * updates_per_epoch)))


def _build_dataset_runtime_fields(
    *,
    scheduler: FixedLRScheduler,
    cache_dir: Path,
    csv_total_samples_est: int,
) -> Dict[str, Any]:
    progress = read_json_if_exists(dataset_progress_path(cache_dir)) or {}
    meta = read_json_if_exists(dataset_meta_path(cache_dir)) or {}
    total_samples_real = meta.get("total_samples", progress.get("tokenized_samples"))
    skipped_samples = meta.get("skipped_samples", progress.get("skipped_samples"))

    return {
        "csv_total_samples_est": int(csv_total_samples_est),
        "total_samples_real": int(total_samples_real) if total_samples_real is not None else None,
        "skipped_samples": int(skipped_samples) if skipped_samples is not None else None,
        "dataset_progress": progress,
        "dataset_meta": meta,
        "dataset_ready": bool(progress.get("done") or meta.get("done")),
        "scheduler_state": scheduler.state_dict(),
        "scheduler_mode": f"fixed_{scheduler.schedule}",
    }


def wrap_ddp(model: nn.Module, cfg: TrainConfig, ctx: DistContext) -> nn.Module:
    if not ctx.is_distributed:
        return model

    kwargs: Dict[str, Any] = {
        "broadcast_buffers": cfg.ddp_broadcast_buffers,
        "find_unused_parameters": cfg.ddp_find_unused_parameters,
    }

    if "static_graph" in DDP.__init__.__code__.co_varnames:
        kwargs["static_graph"] = cfg.ddp_static_graph
    if "gradient_as_bucket_view" in DDP.__init__.__code__.co_varnames:
        kwargs["gradient_as_bucket_view"] = True

    if ctx.device.type == "cuda":
        kwargs["device_ids"] = [ctx.local_rank]
        kwargs["output_device"] = ctx.local_rank

    LOGGER.info(
        "DDP: find_unused_parameters=%s | broadcast_buffers=%s | static_graph=%s",
        cfg.ddp_find_unused_parameters,
        cfg.ddp_broadcast_buffers,
        cfg.ddp_static_graph,
    )
    return DDP(model, **kwargs)


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=(device.type == "cuda")) for k, v in batch.items()}


def format_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def make_scaler(fp16: bool, device: torch.device):
    enabled = bool(device.type == "cuda" and fp16)
    if _NEW_SCALER:
        try:
            return GradScaler("cuda", enabled=enabled)
        except TypeError:
            return GradScaler(enabled=enabled)
    return GradScaler(enabled=enabled)


def count_causal_target_tokens(labels: torch.Tensor) -> int:
    """Count labels that contribute after the causal language-model shift."""
    if labels.ndim == 0 or labels.shape[-1] <= 1:
        return 0
    return int((labels[..., 1:] != -100).sum().item())


def iter_accumulation_batches(
    loader: DataLoader, cfg: TrainConfig, ctx: DistContext,
) -> Iterator[Tuple[Dict[str, torch.Tensor], float, bool, int]]:
    """Yield full accumulation windows with globally token-normalized weights."""
    iterator = iter(loader)
    while True:
        window: List[Dict[str, torch.Tensor]] = []
        for _ in range(cfg.gradient_accumulation_steps):
            try:
                window.append(next(iterator))
            except StopIteration:
                break
        if not window:
            return

        local_counts = [count_causal_target_tokens(batch["labels"]) for batch in window]
        local_total = sum(local_counts)
        global_total = float(local_total)
        if cfg.token_normalized_loss and ctx.is_distributed:
            count_tensor = torch.tensor(float(local_total), device=ctx.device, dtype=torch.float64)
            dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
            global_total = float(count_tensor.item())

        for idx, (batch, target_tokens) in enumerate(zip(window, local_counts)):
            is_window_end = idx == len(window) - 1
            if cfg.token_normalized_loss:
                if ctx.is_distributed:
                    loss_weight = (float(target_tokens) * ctx.world_size) / max(1.0, global_total)
                else:
                    loss_weight = float(target_tokens) / max(1.0, float(local_total))
            else:
                loss_weight = 1.0 / max(1, len(window))
            yield batch, loss_weight, is_window_end, target_tokens


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: FixedLRScheduler,
    scaler: Any,
    cfg: TrainConfig,
    ctx: DistContext,
    epoch: int,
    global_step: int,
    total_steps_ref: Dict[str, int],
    micro_step: int,
    csv_total_samples_est: int,
    cache_dir: Path,
    train_start_time: float,
    status_writer: JsonStatusWriter,
    preview_writer: Optional[JsonPreviewWriter],
    tokenizer,
    tokens_seen_ref: Dict[str, int],
) -> Tuple[float, int, int, bool]:
    model.train()

    _, fp16, bf16 = pick_precision(cfg, ctx.device)
    amp_dtype = torch.float16 if fp16 else (torch.bfloat16 if bf16 else None)
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=amp_dtype)
        if (ctx.device.type == "cuda" and amp_dtype is not None)
        else nullcontext()
    )

    optimizer.zero_grad(set_to_none=True)

    running_loss_sum = 0.0
    running_target_tokens = 0.0
    reached_max_steps = False
    accum_counter = 0
    accum_tokens_local = 0
    last_micro_loss_value: Optional[float] = None
    accum_loss_sum_local = 0.0
    accum_target_tokens_local = 0
    window_invalid = False

    join_ctx = model.join() if (ctx.is_distributed and isinstance(model, DDP)) else nullcontext()

    with join_ctx:
        for batch, token_loss_weight, window_should_step, target_tokens in iter_accumulation_batches(loader, cfg, ctx):
            # DDP-sicherer Stop:
            # Kein sync_stop()/dist.all_reduce() im Microbatch-Loop verwenden,
            # weil das mit DDP-Gradient-Allreduces kollidieren kann.
            # Der Launcher sendet SIGTERM an alle Worker; lokales Flag reicht.
            if SHUTDOWN.stop:
                reached_max_steps = True
                break

            if window_invalid:
                if window_should_step:
                    window_invalid = False
                continue

            batch = move_batch(batch, ctx.device)
            accum_tokens_local += int(batch["attention_mask"].sum().item())
            micro_step += 1

            if ctx.is_main and preview_writer is not None:
                try:
                    input_ids_cpu = batch["input_ids"].detach().to("cpu")
                    attention_mask_cpu = batch["attention_mask"].detach().to("cpu")
                    texts = []
                    for ids, mask in zip(input_ids_cpu, attention_mask_cpu):
                        valid_len = int(mask.sum().item())
                        trimmed_ids = ids[:valid_len]
                        txt = tokenizer.decode(trimmed_ids.tolist(), skip_special_tokens=False)
                        texts.append(txt)
                    preview_text = "\n\n---\n\n".join(texts)
                    preview_writer.write(preview_text[:4000], preview_text[:20000])
                except Exception:
                    pass

            try:
                # Der geplante Batch-Sampler garantiert gleich viele vollständige
                # Accumulation-Fenster je Rank. Deshalb kann no_sync() sicher für alle
                # Microbatches außer dem letzten im Fenster genutzt werden.
                sync_now = bool(window_should_step)

                if ctx.is_distributed and isinstance(model, DDP) and not sync_now:
                    backward_sync_ctx = model.no_sync()
                else:
                    backward_sync_ctx = nullcontext()

                with backward_sync_ctx:
                    with autocast_ctx:
                        outputs = model(**batch)
                        loss = outputs.loss

                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Nicht-finite Loss erkannt: {float(loss.detach().item())}")

                    loss_value = float(loss.detach().item())
                    last_micro_loss_value = loss_value
                    loss_to_backprop = loss * float(token_loss_weight)

                    if scaler is not None and getattr(scaler, "is_enabled", lambda: False)():
                        scaler.scale(loss_to_backprop).backward()
                    else:
                        loss_to_backprop.backward()
            except torch.OutOfMemoryError as oom:
                try:
                    if ctx.device.type == "cuda":
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                if cfg.skip_oom_microbatches and not ctx.is_distributed:
                    LOGGER.warning(
                        "CUDA OOM im Trainingsschritt (microbatch wird uebersprungen) | "
                        "batch_size=%s max_seq_length=%s grad_accum=%s gradient_checkpointing=%s | original=%s",
                        cfg.per_device_train_batch_size,
                        cfg.max_seq_length,
                        cfg.gradient_accumulation_steps,
                        cfg.gradient_checkpointing,
                        oom,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    accum_counter = 0
                    accum_tokens_local = 0
                    accum_loss_sum_local = 0.0
                    accum_target_tokens_local = 0
                    window_invalid = not window_should_step
                    continue
                raise RuntimeError(
                    "CUDA OOM im Trainingsschritt. Empfehlung: "
                    "kleinere max_seq_length oder per_device_train_batch_size, "
                    "ggf. gradient_checkpointing aktivieren. "
                    f"Original: {oom}"
                )

            accum_counter += 1
            accum_loss_sum_local += loss_value * max(0, target_tokens)
            accum_target_tokens_local += max(0, target_tokens)
            should_step = bool(window_should_step)

            if should_step:
                if scaler is not None and getattr(scaler, "is_enabled", lambda: False)():
                    scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)

                optimizer_step_succeeded = True
                if scaler is not None and getattr(scaler, "is_enabled", lambda: False)():
                    scale_before = float(scaler.get_scale())
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer_step_succeeded = float(scaler.get_scale()) >= scale_before
                else:
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)
                if not optimizer_step_succeeded:
                    if ctx.is_main:
                        LOGGER.warning("FP16-Optimizer-Step wegen Gradient-Overflow übersprungen.")
                    accum_counter = 0
                    accum_tokens_local = 0
                    accum_loss_sum_local = 0.0
                    accum_target_tokens_local = 0
                    continue
                global_step += 1
                accum_counter = 0

                lr = scheduler.step(global_step)
                total_steps_ref["value"] = scheduler.total_steps

                maybe_empty_cuda_cache(cfg, ctx, global_step)
                cuda_mem = maybe_log_cuda_memory(
                    cfg=cfg,
                    ctx=ctx,
                    global_step=global_step,
                    prefix=f"step={global_step}",
                )

                loss_stats = torch.tensor(
                    [
                        accum_loss_sum_local,
                        float(accum_target_tokens_local),
                        float(accum_tokens_local),
                    ],
                    device=ctx.device,
                    dtype=torch.float64,
                )
                if ctx.is_distributed:
                    dist.all_reduce(loss_stats, op=dist.ReduceOp.SUM)
                reduced_loss = float(loss_stats[0].item() / max(1.0, loss_stats[1].item()))
                running_loss_sum += float(loss_stats[0].item())
                running_target_tokens += float(loss_stats[1].item())

                if ctx.is_main:
                    step_tokens = int(loss_stats[2].item())
                    tokens_seen_ref["value"] = int(tokens_seen_ref.get("value", 0)) + step_tokens
                    elapsed = max(1e-6, time.time() - train_start_time)
                    steps_done = max(1, global_step)
                    steps_left = max(0, int(total_steps_ref["value"]) - int(global_step))
                    sec_per_step = elapsed / steps_done
                    eta = format_eta(sec_per_step * steps_left)

                    if global_step == 1 or (global_step % cfg.log_every_steps) == 0:
                        LOGGER.info(
                            "Step %d | Loss: %.6f | LR: %s | total_steps=%s",
                            global_step, reduced_loss, lr, total_steps_ref["value"]
                        )

                    payload = {
                        "running": True,
                        "step": global_step,
                        "micro_step": micro_step,
                        "loss": reduced_loss,
                        "learning_rate": lr,
                        "eta": eta,
                        "tokens_per_step": step_tokens,
                        "total_tokens": int(tokens_seen_ref["value"]),
                        "epoch": epoch,
                        "total_steps": int(total_steps_ref["value"]),
                        "scheduler_total_steps": int(total_steps_ref["value"]),
                        "cuda_memory": cuda_mem,
                    }
                    payload.update(
                        _build_dataset_runtime_fields(
                            scheduler=scheduler,
                            cache_dir=cache_dir,
                            csv_total_samples_est=csv_total_samples_est,
                        )
                    )
                    status_writer.write(payload)
                accum_tokens_local = 0
                accum_loss_sum_local = 0.0
                accum_target_tokens_local = 0

                if SHUTDOWN.stop:
                    reached_max_steps = True
                    break
                if cfg.max_steps is not None and global_step >= int(cfg.max_steps):
                    reached_max_steps = True
                    break
                if global_step >= int(total_steps_ref["value"]):
                    reached_max_steps = True
                    break

    do_tail_step = (
        accum_counter > 0
        and not reached_max_steps
        and not SHUTDOWN.stop
        and not ctx.is_distributed
    )

    if ctx.is_distributed and accum_counter > 0 and not reached_max_steps and ctx.is_main:
        LOGGER.info(
            "DDP: überspringe unvollständigen Gradient-Accumulation-Tail am Epoch-Ende "
            "(accum_counter=%s, gradient_accumulation_steps=%s), um Collective-Mismatch zu vermeiden.",
            accum_counter,
            cfg.gradient_accumulation_steps,
        )

    if do_tail_step:
        if scaler is not None and getattr(scaler, "is_enabled", lambda: False)():
            scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)

        optimizer_step_succeeded = True
        if scaler is not None and getattr(scaler, "is_enabled", lambda: False)():
            scale_before = float(scaler.get_scale())
            scaler.step(optimizer)
            scaler.update()
            optimizer_step_succeeded = float(scaler.get_scale()) >= scale_before
        else:
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)
        if not optimizer_step_succeeded:
            if ctx.is_main:
                LOGGER.warning("FP16-Tail-Step wegen Gradient-Overflow übersprungen.")
            avg_loss = running_loss_sum / max(1.0, running_target_tokens)
            return avg_loss, global_step, micro_step, reached_max_steps
        global_step += 1

        lr = scheduler.step(global_step)
        total_steps_ref["value"] = scheduler.total_steps

        maybe_empty_cuda_cache(cfg, ctx, global_step)
        cuda_mem = maybe_log_cuda_memory(
            cfg=cfg,
            ctx=ctx,
            global_step=global_step,
            prefix=f"step={global_step}",
        )

        reduced_loss = float(last_micro_loss_value or 0.0)
        running_loss_sum += float(accum_loss_sum_local)
        running_target_tokens += float(accum_target_tokens_local)

        if ctx.is_main:
            step_tokens = int(accum_tokens_local * max(1, ctx.world_size))
            tokens_seen_ref["value"] = int(tokens_seen_ref.get("value", 0)) + step_tokens
            elapsed = max(1e-6, time.time() - train_start_time)
            steps_done = max(1, global_step)
            steps_left = max(0, int(total_steps_ref["value"]) - int(global_step))
            sec_per_step = elapsed / steps_done
            eta = format_eta(sec_per_step * steps_left)

            LOGGER.info(
                "Step %d | Loss: %.6f | LR: %s | total_steps=%s",
                global_step, reduced_loss, lr, total_steps_ref["value"]
            )

            payload = {
                "running": True,
                "step": global_step,
                "micro_step": micro_step,
                "loss": reduced_loss,
                "learning_rate": lr,
                "eta": eta,
                "tokens_per_step": step_tokens,
                "total_tokens": int(tokens_seen_ref["value"]),
                "epoch": epoch,
                "total_steps": int(total_steps_ref["value"]),
                "scheduler_total_steps": int(total_steps_ref["value"]),
                "cuda_memory": cuda_mem,
            }
            payload.update(
                _build_dataset_runtime_fields(
                    scheduler=scheduler,
                    cache_dir=cache_dir,
                    csv_total_samples_est=csv_total_samples_est,
                )
            )
            status_writer.write(payload)

    avg_loss = running_loss_sum / max(1.0, running_target_tokens)
    return avg_loss, global_step, micro_step, reached_max_steps


@torch.no_grad()
def evaluate_model(
    model: nn.Module, loader: DataLoader, cfg: TrainConfig, ctx: DistContext,
) -> Tuple[Optional[float], Optional[float], int]:
    """Return globally token-weighted validation loss and perplexity."""
    evaluation_model = unwrap_model(model)
    evaluation_model.eval()
    _, fp16, bf16 = pick_precision(cfg, ctx.device)
    amp_dtype = torch.float16 if fp16 else (torch.bfloat16 if bf16 else None)
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=amp_dtype)
        if ctx.device.type == "cuda" and amp_dtype is not None else nullcontext()
    )
    loss_sum = torch.zeros(1, dtype=torch.float64, device=ctx.device)
    token_count = torch.zeros(1, dtype=torch.float64, device=ctx.device)
    sample_count = torch.zeros(1, dtype=torch.long, device=ctx.device)

    for batch in loader:
        if SHUTDOWN.stop:
            break
        batch = move_batch(batch, ctx.device)
        target_tokens = count_causal_target_tokens(batch["labels"])
        if target_tokens <= 0:
            continue
        with autocast_ctx:
            outputs = evaluation_model(**batch)
        if torch.isfinite(outputs.loss):
            loss_sum += outputs.loss.detach().double() * target_tokens
            token_count += target_tokens
            sample_count += int(batch["input_ids"].shape[0])

    if ctx.is_distributed:
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(sample_count, op=dist.ReduceOp.SUM)
    model.train()
    if token_count.item() <= 0:
        return None, None, int(sample_count.item())
    val_loss = float((loss_sum / token_count).item())
    perplexity = float(math.exp(min(20.0, val_loss)))
    return val_loss, perplexity, int(sample_count.item())


def save_best_model(model: nn.Module, tokenizer: Any, outdir: Path, ctx: DistContext) -> Optional[Path]:
    if not ctx.is_main:
        return None
    best_dir = outdir / "best_model"
    tmp_dir = outdir / "best_model.tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    unwrap_model(model).save_pretrained(tmp_dir)
    tokenizer.save_pretrained(tmp_dir)
    if best_dir.exists():
        shutil.rmtree(best_dir)
    tmp_dir.replace(best_dir)
    LOGGER.info("Neues bestes Modell gespeichert: %s", best_dir)
    return best_dir



def save_model_artifacts(
    *,
    model: nn.Module,
    tokenizer,
    outdir: Path,
    cfg: TrainConfig,
    ctx: DistContext,
    ngram_state,
    skipped_samples: Optional[int] = None,
) -> None:
    """Speichert Modell, Tokenizer, optional gemergtes LoRA-Modell und Template-Metadaten.

    Wichtig: Diese Funktion wird nur auf rank 0 ausgefuehrt. Sie ist bewusst
    sowohl fuer normalen Abschluss als auch fuer vorzeitigen Stop nutzbar.
    """
    if not ctx.is_main:
        return

    outdir.mkdir(parents=True, exist_ok=True)

    save_target = unwrap_model(model)
    mode = (cfg.train_mode or "full").lower().strip()

    if mode == "lora" and _PEFT_AVAILABLE:
        save_target.save_pretrained(outdir)
        tokenizer.save_pretrained(outdir)

        if cfg.merge_lora_on_save:
            try:
                merge_source = save_target

                if hasattr(merge_source, "merge_and_unload"):
                    merged_target = merge_source.merge_and_unload()
                elif hasattr(unwrap_model(model), "merge_and_unload"):
                    merged_target = unwrap_model(model).merge_and_unload()
                else:
                    raise AttributeError(
                        f"{merge_source.__class__.__name__} unterstuetzt merge_and_unload() nicht"
                    )

                merged_dir = outdir / "merged"
                merged_dir.mkdir(parents=True, exist_ok=True)
                merged_target.save_pretrained(merged_dir)
                tokenizer.save_pretrained(merged_dir)
                LOGGER.info("Gemergtes LoRA-Modell gespeichert nach: %s", merged_dir)
            except Exception as merge_exc:
                LOGGER.warning("LoRA-Merge beim Speichern fehlgeschlagen: %s", merge_exc)
                LOGGER.warning("Adapter wurde trotzdem normal gespeichert: %s", outdir)
    else:
        save_target.save_pretrained(outdir)
        tokenizer.save_pretrained(outdir)

    template_info = {
        "template_mode": cfg.template_mode,
        "mixed_training": bool(cfg.mixed_training),
        "mixed_text_column": cfg.mixed_text_column,
        "training_phase": cfg.training_phase,
        "text_token_weight": float(cfg.text_token_weight),
        "dialog_token_weight": float(cfg.dialog_token_weight),
        "chunk_long_texts": bool(cfg.chunk_long_texts),
        "append_eos_to_text": bool(cfg.append_eos_to_text),
        "pack_short_texts": bool(cfg.pack_short_texts),
        "tokenizer_source": cfg.tokenizer_dir or cfg.model_dir,
        "force_template": cfg.force_template,
        "chat_template": getattr(tokenizer, "chat_template", "") or "",
        "special_tokens": {
            "pad_token": tokenizer.pad_token,
            "eos_token": tokenizer.eos_token,
            "bos_token": tokenizer.bos_token,
        },
        "max_history_turns": cfg.max_history_turns,
        "strict_whole_turns": True,
        "sort_by_length": bool(cfg.sort_by_length),
        "use_ngrams": bool(cfg.use_ngrams),
        "ngram_summary": ngram_summary_text(ngram_state),
    }
    if skipped_samples is not None:
        template_info["skipped_samples"] = int(skipped_samples)


    (outdir / "template_info.json").write_text(
        json.dumps(template_info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_training_checkpoint(
    *, model: nn.Module, tokenizer: Any, optimizer: torch.optim.Optimizer,
    scheduler: FixedLRScheduler, scaler: Any, outdir: Path,
    epoch: int, global_step: int, cfg: TrainConfig, ctx: DistContext,
    best_val_loss: Optional[float] = None, epochs_without_improvement: int = 0,
    total_tokens_seen: int = 0,
) -> Optional[Path]:
    """Save a resumable checkpoint atomically at epoch boundaries."""
    local_rng = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    rng_by_rank: Optional[List[Any]] = None
    if ctx.is_distributed:
        rng_by_rank = [None] * ctx.world_size if ctx.is_main else None
        dist.gather_object(local_rng, rng_by_rank, dst=0)
    else:
        rng_by_rank = [local_rng]
    if not ctx.is_main:
        return None
    checkpoint_dir = outdir / "checkpoints" / f"checkpoint-{global_step:08d}"
    tmp_dir = checkpoint_dir.with_name(checkpoint_dir.name + ".tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    target = unwrap_model(model)
    target.save_pretrained(tmp_dir)
    tokenizer.save_pretrained(tmp_dir)
    state = {
        "version": 1,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None and hasattr(scaler, "state_dict") else None,
        "rng_by_rank": rng_by_rank,
        "config": cfg.__dict__,
        "best_val_loss": best_val_loss,
        "epochs_without_improvement": int(epochs_without_improvement),
        "total_tokens_seen": int(total_tokens_seen),
    }
    torch.save(state, tmp_dir / "trainer_state.pt")
    tmp_dir.replace(checkpoint_dir)

    checkpoints = sorted(
        (p for p in checkpoint_dir.parent.glob("checkpoint-*") if p.is_dir()),
        key=lambda p: p.name,
    )
    for old in checkpoints[:-cfg.keep_last_k_checkpoints]:
        shutil.rmtree(old, ignore_errors=True)
    LOGGER.info("Checkpoint gespeichert: %s", checkpoint_dir)
    return checkpoint_dir


def load_training_state(
    resume_dir: Path, optimizer: torch.optim.Optimizer,
    scheduler: FixedLRScheduler, scaler: Any, device: torch.device, rank: int = 0,
) -> Tuple[int, int, Dict[str, Any]]:
    state_path = resume_dir / "trainer_state.pt"
    if not state_path.exists():
        raise FileNotFoundError(f"Resume-State fehlt: {state_path}")
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    for optimizer_state in optimizer.state.values():
        for key, value in optimizer_state.items():
            if torch.is_tensor(value):
                optimizer_state[key] = value.to(device)
    scheduler.load_state_dict(state.get("scheduler") or {})
    if scaler is not None and state.get("scaler") and hasattr(scaler, "load_state_dict"):
        scaler.load_state_dict(state["scaler"])
    rank_states = state.get("rng_by_rank") or []
    rank_state = rank_states[min(max(0, rank), len(rank_states) - 1)] if rank_states else None
    if rank_state:
        random.setstate(rank_state["python"])
        torch.set_rng_state(rank_state["torch"])
        if torch.cuda.is_available() and rank_state.get("cuda") is not None:
            torch.cuda.set_rng_state_all(rank_state["cuda"])
    return int(state.get("epoch", -1)) + 1, int(state.get("global_step", 0)), state


def release_training_memory(
    *,
    ctx: DistContext,
    model: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    loader: Optional[Any] = None,
    dataset: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
) -> None:
    """Versucht Trainings-RAM/VRAM vor Prozessende explizit freizugeben.

    Hinweis: Die Funktion leert interne Strukturen und verschiebt das Modell auf CPU.
    Die aufrufende Funktion setzt danach ihre lokalen Variablen auf None, damit die
    letzten starken Referenzen verschwinden und gc/empty_cache wirklich greifen.
    """
    try:
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            for group in getattr(optimizer, "param_groups", []):
                group["params"] = []
            try:
                optimizer.state.clear()
            except Exception:
                pass
    except Exception:
        pass

    try:
        if model is not None:
            try:
                unwrapped = unwrap_model(model)
                unwrapped.to(torch.device("cpu"))
            except Exception:
                pass
            try:
                model.to(torch.device("cpu"))
            except Exception:
                pass
    except Exception:
        pass

    try:
        del scheduler
    except Exception:
        pass
    try:
        del scaler
    except Exception:
        pass
    try:
        del loader
    except Exception:
        pass
    try:
        del dataset
    except Exception:
        pass
    try:
        del tokenizer
    except Exception:
        pass
    try:
        del model
    except Exception:
        pass

    gc.collect()

    try:
        if ctx.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(ctx.device)
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
            try:
                torch.cuda.reset_peak_memory_stats(ctx.device)
            except Exception:
                pass
    except Exception:
        pass

    try:
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass

    if ctx.is_main:
        LOGGER.info("Training-Speicherfreigabe ausgefuehrt.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    args = parser.parse_args()

    register_signal_handlers()
    cfg = load_cfg(args.config)
    ctx = init_dist(cfg)

    # Volta/V100 has Tensor Cores for FP16, but no native TF32 execution path.
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(ctx.device)
        if props.major < 8:
            cfg.allow_tf32 = False

    if torch.cuda.is_available():
        if cfg.allow_tf32:
            if hasattr(torch.backends, "fp32_precision"):
                torch.backends.fp32_precision = "ieee"
            if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
                torch.backends.cuda.matmul.fp32_precision = "tf32"
            if hasattr(torch.backends.cudnn, "fp32_precision"):
                torch.backends.cudnn.fp32_precision = "ieee"
            if hasattr(torch.backends.cudnn, "conv") and hasattr(torch.backends.cudnn.conv, "fp32_precision"):
                torch.backends.cudnn.conv.fp32_precision = "tf32"
            if hasattr(torch.backends.cudnn, "rnn") and hasattr(torch.backends.cudnn.rnn, "fp32_precision"):
                torch.backends.cudnn.rnn.fp32_precision = "tf32"
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass
        else:
            if hasattr(torch.backends, "fp32_precision"):
                torch.backends.fp32_precision = "ieee"
            if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
                torch.backends.cuda.matmul.fp32_precision = "ieee"
            if hasattr(torch.backends.cudnn, "fp32_precision"):
                torch.backends.cudnn.fp32_precision = "ieee"
            if hasattr(torch.backends.cudnn, "conv") and hasattr(torch.backends.cudnn.conv, "fp32_precision"):
                torch.backends.cudnn.conv.fp32_precision = "ieee"
            if hasattr(torch.backends.cudnn, "rnn") and hasattr(torch.backends.cudnn.rnn, "fp32_precision"):
                torch.backends.cudnn.rnn.fp32_precision = "ieee"
            try:
                torch.set_float32_matmul_precision("highest")
            except Exception:
                pass

    outdir = Path(cfg.output_dir or cfg.save_dir or "./training_outputs/worker_run").expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    log_path = outdir / "training.log"
    status_path = outdir / "status.json"
    preview_path = outdir / "livepreview.json"

    setup_logging(log_path, ctx)
    status_writer = JsonStatusWriter(status_path, ctx)
    preview_writer = JsonPreviewWriter(preview_path, ctx)

    ngram_state = None

    try:
        set_seed(cfg.seed + ctx.rank, deterministic=cfg.deterministic)

        LOGGER.info(
            "Worker gestartet | rank=%s local_rank=%s world_size=%s device=%s",
            ctx.rank, ctx.local_rank, ctx.world_size, ctx.device
        )
        LOGGER.info("Train mode: %s | train_from_scratch=%s | include_prompt_loss=%s", cfg.train_mode, cfg.train_from_scratch, cfg.include_prompt_loss)
        LOGGER.info("Config: %s", json.dumps(cfg.__dict__, ensure_ascii=False))
        hardware_profile = hardware_training_profile(cfg, ctx)
        LOGGER.info("Hardware-Profil: %s", json.dumps(hardware_profile, ensure_ascii=False))

        if cfg.dataset_audit:
            audit_error_path = outdir / "dataset_audit_error.txt"
            if ctx.is_main:
                audit_error_path.unlink(missing_ok=True)
                try:
                    status_writer.write({
                        "running": True,
                        "status": "auditing_dataset",
                        "eta": "Dataset wird geprüft (Cache wird verwendet, falls vorhanden)",
                        "dataset_ready": False,
                    })
                    dataset_audit_report = audit_dataset_cached(
                        cfg, outdir, status_writer=status_writer,
                    )
                except Exception as exc:
                    audit_error_path.write_text(
                        f"{exc.__class__.__name__}: {exc}\n\n{traceback.format_exc()}",
                        encoding="utf-8",
                    )
            barrier(ctx)
            if audit_error_path.exists():
                raise RuntimeError(audit_error_path.read_text(encoding="utf-8"))
            if not ctx.is_main:
                dataset_audit_report = json.loads(
                    (outdir / "dataset_audit.json").read_text(encoding="utf-8")
                )
            if cfg.mixed_training and not dataset_audit_report.get("mixed_training_ready"):
                raise RuntimeError(
                    "Gemischtes Training kann nicht gestartet werden: "
                    + "; ".join(dataset_audit_report.get("errors") or ["Dataset nicht kompatibel"])
                )
            if cfg.training_phase != "custom" and not dataset_audit_report.get("training_phase_ready"):
                raise RuntimeError(
                    "Trainingsphase kann nicht gestartet werden: "
                    + "; ".join(dataset_audit_report.get("errors") or ["Dataset nicht kompatibel"])
                )
            if cfg.dataset_audit_strict and dataset_audit_report.get("errors"):
                raise RuntimeError(
                    "Dataset-Audit fehlgeschlagen: " + "; ".join(dataset_audit_report["errors"])
                )
        else:
            if cfg.mixed_training and cfg.training_phase != "sft":
                errors = _mixed_dataset_errors(
                    cfg.csv_path,
                    cfg.mixed_text_column,
                    require_text_sample=True,
                )
                if errors:
                    raise RuntimeError(
                        "Gemischtes Training kann nicht gestartet werden: "
                        + "; ".join(errors)
                    )
            has_structured_columns = (
                _csv_has_column(cfg.csv_path, "id")
                and _csv_has_column(cfg.csv_path, "Assistentin")
            )
            if cfg.training_phase == "sft" and not has_structured_columns:
                raise RuntimeError("Die SFT-Phase benötigt ein strukturiertes Dialog-Dataset")
            if (
                cfg.training_phase == "pretrain"
                and has_structured_columns
                and not cfg.mixed_training
            ):
                raise RuntimeError(
                    "Die Pretrain-Phase benötigt eine Plain-Text-CSV oder den Mischmodus"
                )
            dataset_audit_report = {"enabled": False, "ok": True}

        total_samples_est = estimate_examples_from_audit(cfg, dataset_audit_report)
        if total_samples_est <= 0:
            total_samples_est = count_examples_fast(cfg)
        if total_samples_est <= 0:
            raise RuntimeError("Kein Trainingssample gefunden.")

        if ctx.is_main and cfg.train_scratch_tokenizer and not cfg.resume:
            status_writer.write({
                "running": True,
                "status": "building_tokenizer",
                "eta": "Scratch-Tokenizer wird aufgebaut oder aus Cache geladen",
                "dataset_progress": {
                    "phase": "scratch_tokenizer",
                    "seen_samples": 0,
                    "expected_samples": int(total_samples_est),
                    "done": False,
                    "updated_at": time.time(),
                },
                "dataset_ready": False,
            })
        prepare_scratch_tokenizer_if_requested(
            cfg, outdir, ctx, example_count=total_samples_est,
        )

        cache_path_state = outdir / "dataset_cache_path.json"
        cache_path_error = outdir / "dataset_cache_path_error.txt"
        if ctx.is_main:
            cache_path_state.unlink(missing_ok=True)
            cache_path_error.unlink(missing_ok=True)
            try:
                cache_dir = compute_shard_cache_dir(cfg)
                _atomic_write_json(cache_path_state, {"cache_dir": str(cache_dir)})
            except Exception as exc:
                cache_path_error.write_text(
                    f"{exc.__class__.__name__}: {exc}\n\n{traceback.format_exc()}",
                    encoding="utf-8",
                )
        barrier(ctx)
        if cache_path_error.exists():
            raise RuntimeError(cache_path_error.read_text(encoding="utf-8"))
        if not ctx.is_main:
            cache_dir = Path(
                json.loads(cache_path_state.read_text(encoding="utf-8"))["cache_dir"]
            )

        if ctx.is_main:
            status_writer.write({
                "running": True,
                "status": "building_dataset",
                "eta": "Dataset wird vollständig aufgebaut",
                "csv_total_samples_est": int(total_samples_est),
                "dataset_cache_dir": str(cache_dir),
            })
            prepare_shard_dataset(
                cfg,
                cache_dir,
                ctx,
                status_writer=status_writer,
                preview_writer=preview_writer,
                estimated_samples=total_samples_est,
            )
        wait_for_dataset_ready(cache_dir)

        barrier(ctx)
        if ctx.is_main:
            cache_path_state.unlink(missing_ok=True)
            cache_path_error.unlink(missing_ok=True)
        model, tokenizer, fp16, bf16, ngram_state = build_model_and_tokenizer(
            cfg, ctx, ngram_cache_dir=cache_dir,
        )

        serialized_plan_path = outdir / "batch_plan_state.pkl"
        plan_error_path = outdir / "batch_plan_error.txt"
        if ctx.is_main:
            plan_error_path.unlink(missing_ok=True)
            try:
                global_batch_plan, batch_plan_stats = build_token_batch_plan(cache_dir, cfg, ctx)
                model_parameters = int(sum(parameter.numel() for parameter in model.parameters()))
                if cfg.max_steps is not None:
                    estimated_training_tokens = int(
                        max(1, cfg.max_steps)
                        * batch_plan_stats["max_tokens_per_batch"]
                        * cfg.gradient_accumulation_steps
                        * max(1, ctx.world_size)
                    )
                    token_estimate_source = "max_steps_token_budget_upper_bound"
                else:
                    estimated_training_tokens = int(
                        batch_plan_stats["training_tokens_per_epoch"]
                        * max(0.0, cfg.num_train_epochs)
                    )
                    token_estimate_source = "epochs_from_unpadded_tokens"
                batch_plan_stats.update({
                    "model_parameters": model_parameters,
                    "estimated_training_tokens": estimated_training_tokens,
                    "estimated_tokens_per_parameter": float(
                        estimated_training_tokens / max(1, model_parameters)
                    ),
                    "token_estimate_source": token_estimate_source,
                })
                if cfg.train_from_scratch and batch_plan_stats["estimated_tokens_per_parameter"] < 10.0:
                    LOGGER.warning(
                        "Scratch-Tokenbudget ist klein: ca. %.2f Tokens/Parameter "
                        "(%s Tokens, %s Parameter). Mehr hochwertige Daten, Epochen oder "
                        "ein kleineres Modell einplanen; dies ist ein Hinweis, kein Abbruch.",
                        batch_plan_stats["estimated_tokens_per_parameter"],
                        estimated_training_tokens,
                        model_parameters,
                    )
                tmp_plan_path = serialized_plan_path.with_suffix(".tmp")
                with open(tmp_plan_path, "wb") as handle:
                    pickle.dump(
                        {"global_batches": global_batch_plan, "stats": batch_plan_stats},
                        handle,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )
                tmp_plan_path.replace(serialized_plan_path)
            except Exception as exc:
                plan_error_path.write_text(
                    f"{exc.__class__.__name__}: {exc}\n\n{traceback.format_exc()}",
                    encoding="utf-8",
                )
        barrier(ctx)
        if plan_error_path.exists():
            raise RuntimeError(plan_error_path.read_text(encoding="utf-8"))
        if not ctx.is_main:
            with open(serialized_plan_path, "rb") as handle:
                serialized_plan = pickle.load(handle)
            global_batch_plan = serialized_plan["global_batches"]
            batch_plan_stats = serialized_plan["stats"]
        barrier(ctx)
        if ctx.is_main:
            serialized_plan_path.unlink(missing_ok=True)
            plan_error_path.unlink(missing_ok=True)
        if ctx.is_main:
            (outdir / "batch_plan.json").write_text(
                json.dumps(batch_plan_stats, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            LOGGER.info("Batch-Plan: %s", json.dumps(batch_plan_stats, ensure_ascii=False))
        dataset = PlannedBatchIterableDataset(
            cache_dir=cache_dir,
            global_batches=global_batch_plan,
            rank=ctx.rank,
            world_size=ctx.world_size,
            seed=cfg.seed,
            shuffle=cfg.shuffle_batches,
            original_batch_count=batch_plan_stats["original_batches"],
        )
        validation_dataset = None
        validation_type_datasets: Dict[str, TokenizedShardIterableDataset] = {}
        if cfg.val_split > 0.0 and cfg.validate_every_epoch:
            validation_dataset = TokenizedShardIterableDataset(
                cache_dir=cache_dir,
                rank=ctx.rank,
                world_size=ctx.world_size,
                sort_by_length=False,
                epoch=0,
                split="validation",
                val_split=cfg.val_split,
                split_seed=cfg.split_seed,
            )
            for sample_type in ("text", "dialog"):
                validation_type_datasets[sample_type] = TokenizedShardIterableDataset(
                    cache_dir=cache_dir,
                    rank=ctx.rank,
                    world_size=ctx.world_size,
                    sort_by_length=False,
                    epoch=0,
                    split="validation",
                    val_split=cfg.val_split,
                    split_seed=cfg.split_seed,
                    sample_type_filter=sample_type,
                )
        collator = DataCollator(
            tokenizer.pad_token_id or tokenizer.eos_token_id or 0,
            pad_to_multiple_of=8,
            fixed_length=(cfg.max_seq_length if cfg.fixed_padding else None),
        )

        initial_total_steps = estimate_total_steps_from_batch_plan(
            len(global_batch_plan), cfg, ctx
        )

        effective_warmup_steps = cfg.warmup_steps
        if effective_warmup_steps <= 0 and cfg.warmup_ratio > 0.0:
            effective_warmup_steps = int(math.ceil(initial_total_steps * cfg.warmup_ratio))
        effective_warmup_steps = max(0, min(effective_warmup_steps, max(0, initial_total_steps - 1)))

        if cfg.dataloader_num_workers < 0:
            cpu_count = max(1, os.cpu_count() or 1)
            available_cpus = max(1, cpu_count - 2)
            num_loader_workers = min(
                4,
                max(1, available_cpus // max(1, ctx.world_size)),
            )
        else:
            num_loader_workers = max(0, int(cfg.dataloader_num_workers))
        loader_kwargs: Dict[str, Any] = {
            "dataset": dataset,
            "batch_size": None,
            "num_workers": num_loader_workers,
            "pin_memory": (ctx.device.type == "cuda"),
            "collate_fn": collator,
        }
        if num_loader_workers > 0:
            loader_kwargs["prefetch_factor"] = max(1, int(cfg.prefetch_factor))
            loader_kwargs["persistent_workers"] = bool(cfg.persistent_workers)

        loader = DataLoader(**loader_kwargs)
        validation_loader = None
        validation_type_loaders: Dict[str, DataLoader] = {}
        if validation_dataset is not None:
            validation_loader_kwargs: Dict[str, Any] = {
                "dataset": validation_dataset,
                "batch_size": max(1, cfg.per_device_train_batch_size),
                "num_workers": num_loader_workers,
                "pin_memory": (ctx.device.type == "cuda"),
                "collate_fn": collator,
            }
            if num_loader_workers > 0:
                validation_loader_kwargs["prefetch_factor"] = max(1, int(cfg.prefetch_factor))
                validation_loader_kwargs["persistent_workers"] = bool(cfg.persistent_workers)
            validation_loader = DataLoader(**validation_loader_kwargs)
            for sample_type, type_dataset in validation_type_datasets.items():
                validation_type_loaders[sample_type] = DataLoader(
                    dataset=type_dataset,
                    batch_size=max(1, cfg.per_device_train_batch_size),
                    num_workers=0,
                    pin_memory=(ctx.device.type == "cuda"),
                    collate_fn=collator,
                )

        optimizer = build_optimizer(model, cfg)
        scheduler = FixedLRScheduler(
            optimizer=optimizer,
            base_lr=cfg.learning_rate,
            schedule=cfg.lr_schedule,
            total_steps=initial_total_steps,
            warmup_steps=effective_warmup_steps,
            min_lr_ratio=cfg.min_lr_ratio,
            lr_decay_factor=cfg.lr_decay_factor,
        )
        scaler = make_scaler(fp16=fp16, device=ctx.device)

        total_steps_ref = {"value": scheduler.total_steps}
        resume_epoch = 0
        resume_global_step = 0
        resume_state: Dict[str, Any] = {}
        if cfg.resume:
            resume_epoch, resume_global_step, resume_state = load_training_state(
                Path(cfg.resume).expanduser().resolve(), optimizer, scheduler, scaler, ctx.device, ctx.rank
            )
            scheduler.total_steps = max(
                1, initial_total_steps, resume_global_step,
            )
            total_steps_ref["value"] = scheduler.total_steps
            LOGGER.info(
                "Training fortgesetzt | checkpoint=%s start_epoch=%s global_step=%s",
                cfg.resume, resume_epoch, resume_global_step,
            )
            previous_best = Path(cfg.resume).expanduser().resolve().parent.parent / "best_model"
            if ctx.is_main and previous_best.is_dir() and not (outdir / "best_model").exists():
                shutil.copytree(previous_best, outdir / "best_model")
                LOGGER.info("Bisheriges best_model in den neuen Lauf übernommen: %s", previous_best)
            barrier(ctx)

        tokens_seen_ref = {
            "value": int(
                resume_state.get(
                    "total_tokens_seen",
                    resume_global_step
                    * batch_plan_stats["max_tokens_per_batch"]
                    * cfg.gradient_accumulation_steps
                    * max(1, ctx.world_size),
                )
            )
        }

        if ctx.is_main:
            LOGGER.info(
                "Fester Scheduler initialisiert | schedule=%s total_steps=%s warmup_steps=%s min_lr_ratio=%s lr_decay_factor=%s source=exact_batch_plan",
                cfg.lr_schedule,
                scheduler.total_steps,
                effective_warmup_steps,
                cfg.min_lr_ratio,
                cfg.lr_decay_factor,
            )
            LOGGER.info(
                "Dataset vollständig | csv_estimate=%s real_samples=%s total_steps=%s",
                total_samples_est,
                (read_json_if_exists(dataset_meta_path(cache_dir)) or {}).get("total_samples"),
                scheduler.total_steps,
            )
            LOGGER.info(
                "Datenpipeline | phase=%s max_history_turns=%s text_chunking=%s overlap=%s eos=%s packing=%s",
                cfg.training_phase,
                cfg.max_history_turns,
                cfg.chunk_long_texts,
                cfg.text_chunk_overlap,
                cfg.append_eos_to_text,
                cfg.pack_short_texts,
            )
            LOGGER.info(
                "sort_by_length=%s",
                cfg.sort_by_length,
            )
            LOGGER.info(
                "CUDA memory diagnostics | enabled=%s interval_steps=%s empty_cache_interval_steps=%s",
                cfg.log_cuda_memory,
                cfg.cuda_memory_log_interval_steps,
                cfg.cuda_empty_cache_interval_steps,
            )
            LOGGER.info(ngram_summary_text(ngram_state))

            initial_cuda_mem = cuda_memory_snapshot(ctx.device)
            if initial_cuda_mem:
                LOGGER.info(
                    "initial CUDA memory | dev=%s allocated=%s MB reserved=%s MB max_allocated=%s MB max_reserved=%s MB",
                    initial_cuda_mem["device_index"],
                    initial_cuda_mem["allocated_mb"],
                    initial_cuda_mem["reserved_mb"],
                    initial_cuda_mem["max_allocated_mb"],
                    initial_cuda_mem["max_reserved_mb"],
                )

            payload = {
                "running": True,
                "step": resume_global_step,
                "micro_step": 0,
                "loss": None,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "eta": "",
                "tokens_per_step": int(
                    batch_plan_stats["max_tokens_per_batch"]
                    * cfg.gradient_accumulation_steps
                    * max(1, ctx.world_size)
                ),
                "total_tokens": int(tokens_seen_ref["value"]),
                "epoch": resume_epoch,
                "total_steps": int(total_steps_ref["value"]),
                "scheduler_total_steps": int(total_steps_ref["value"]),
                "warmup_steps": int(effective_warmup_steps),
                "cuda_memory": initial_cuda_mem,
                "log_cuda_memory": bool(cfg.log_cuda_memory),
                "cuda_memory_log_interval_steps": int(cfg.cuda_memory_log_interval_steps),
                "cuda_empty_cache_interval_steps": int(cfg.cuda_empty_cache_interval_steps),
                "scheduler_mode": f"fixed_{cfg.lr_schedule}",
                "batch_plan": batch_plan_stats,
                "dataloader_num_workers_effective": num_loader_workers,
                "dataset_audit": dataset_audit_report,
                "hardware_profile": hardware_profile,
            }
            payload.update(
                _build_dataset_runtime_fields(
                    scheduler=scheduler,
                    cache_dir=cache_dir,
                    csv_total_samples_est=total_samples_est,
                )
            )
            status_writer.write(payload)

        model = wrap_ddp(model, cfg, ctx)
        barrier(ctx)

        global_step = resume_global_step
        micro_step = 0
        last_loss = None
        last_val_loss: Optional[float] = None
        last_perplexity: Optional[float] = None
        last_validation_by_type: Dict[str, Dict[str, Any]] = {}
        resume_best_val = resume_state.get("best_val_loss")
        best_val_loss = float(resume_best_val) if resume_best_val is not None else float("inf")
        epochs_without_improvement = int(resume_state.get("epochs_without_improvement", 0))
        early_stopped = False
        train_start_time = time.time()

        epochs = max(1, int(math.ceil(cfg.num_train_epochs)))
        for epoch in range(resume_epoch, epochs):
            if cfg.max_steps is not None and global_step >= int(cfg.max_steps):
                break
            if global_step >= int(total_steps_ref["value"]):
                break

            dataset.set_epoch(epoch)

            avg_loss, global_step, micro_step, reached_max_steps = train_epoch(
                model=model,
                loader=loader,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                cfg=cfg,
                ctx=ctx,
                epoch=epoch,
                global_step=global_step,
                total_steps_ref=total_steps_ref,
                micro_step=micro_step,
                csv_total_samples_est=total_samples_est,
                cache_dir=cache_dir,
                train_start_time=train_start_time,
                status_writer=status_writer,
                preview_writer=preview_writer,
                tokenizer=tokenizer,
                tokens_seen_ref=tokens_seen_ref,
            )
            last_loss = avg_loss

            if ctx.is_main:
                LOGGER.info("Epoche %d abgeschlossen | avg_loss=%.6f", epoch, avg_loss)

            if validation_loader is not None:
                validation_dataset.set_epoch(epoch)
                val_loss, perplexity, val_samples = evaluate_model(model, validation_loader, cfg, ctx)
                last_val_loss, last_perplexity = val_loss, perplexity
                validation_by_type: Dict[str, Dict[str, Any]] = {}
                for sample_type, type_loader in validation_type_loaders.items():
                    validation_type_datasets[sample_type].set_epoch(epoch)
                    type_loss, type_perplexity, type_samples = evaluate_model(
                        model, type_loader, cfg, ctx,
                    )
                    validation_by_type[sample_type] = {
                        "loss": type_loss,
                        "perplexity": type_perplexity,
                        "samples": type_samples,
                    }
                last_validation_by_type = validation_by_type
                if ctx.is_main:
                    LOGGER.info(
                        "Validation nach Typ | epoch=%s metrics=%s",
                        epoch,
                        json.dumps(validation_by_type, ensure_ascii=False, sort_keys=True),
                    )
                if val_loss is None:
                    if ctx.is_main:
                        LOGGER.warning("Validation-Split enthält keine nutzbaren Ziel-Tokens.")
                else:
                    improved = val_loss < (best_val_loss - cfg.early_stopping_min_delta)
                    if improved:
                        best_val_loss = val_loss
                        epochs_without_improvement = 0
                        save_best_model(model, tokenizer, outdir, ctx)
                    else:
                        epochs_without_improvement += 1
                    if ctx.is_main:
                        LOGGER.info(
                            "Validation | epoch=%s loss=%.6f perplexity=%.4f samples=%s "
                            "best=%.6f no_improvement=%s/%s",
                            epoch, val_loss, perplexity, val_samples, best_val_loss,
                            epochs_without_improvement, cfg.early_stopping_patience,
                        )
                        status_payload = {
                            "running": True,
                            "step": global_step,
                            "epoch": epoch,
                            "loss": avg_loss,
                            "val_loss": val_loss,
                            "perplexity": perplexity,
                            "best_val_loss": best_val_loss,
                            "validation_samples": val_samples,
                            "epochs_without_improvement": epochs_without_improvement,
                            "early_stopping_patience": cfg.early_stopping_patience,
                            "validation_by_type": validation_by_type,
                        }
                        status_payload.update(_build_dataset_runtime_fields(
                            scheduler=scheduler, cache_dir=cache_dir,
                            csv_total_samples_est=total_samples_est,
                        ))
                        status_writer.write(status_payload)
                    if cfg.early_stopping_patience > 0 and epochs_without_improvement >= cfg.early_stopping_patience:
                        early_stopped = True

            if cfg.save_every_epoch and not SHUTDOWN.stop:
                save_training_checkpoint(
                    model=model,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    outdir=outdir,
                    epoch=epoch,
                    global_step=global_step,
                    cfg=cfg,
                    ctx=ctx,
                    best_val_loss=(best_val_loss if math.isfinite(best_val_loss) else None),
                    epochs_without_improvement=epochs_without_improvement,
                    total_tokens_seen=int(tokens_seen_ref["value"]),
                )
                barrier(ctx)

            if early_stopped:
                if ctx.is_main:
                    LOGGER.info("Early Stopping nach Epoche %s aktiviert.", epoch)
                break
            if reached_max_steps or SHUTDOWN.stop:
                break

        if SHUTDOWN.stop:
            # Bei Stop trotzdem einen nutzbaren Zwischenstand speichern.
            # Wichtig: Nur rank 0 schreibt Dateien; danach warten alle Ranks am Barrier.
            if ctx.is_main:
                try:
                    final_meta = read_json_if_exists(dataset_meta_path(cache_dir)) or {}
                    final_progress = read_json_if_exists(dataset_progress_path(cache_dir)) or {}
                    final_skipped_samples = int(
                        final_meta.get("skipped_samples", final_progress.get("skipped_samples", 0))
                    )
                    final_cuda_mem = cuda_memory_snapshot(ctx.device)

                    LOGGER.info("Stop-Signal erkannt. Speichere Zwischenstand nach: %s", outdir)
                    save_model_artifacts(
                        model=model,
                        tokenizer=tokenizer,
                        outdir=outdir,
                        cfg=cfg,
                        ctx=ctx,
                        ngram_state=ngram_state,
                        skipped_samples=final_skipped_samples,
                    )

                    status_payload = {
                        "running": False,
                        "step": global_step,
                        "micro_step": micro_step,
                        "loss": last_loss,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "eta": "stopped",
                        "status": "stopped_saved",
                        "done": False,
                        "stopped": True,
                        "saved": True,
                        "save_dir": str(outdir),
                        "cuda_memory": final_cuda_mem,
                        "csv_total_samples_est": int(total_samples_est),
                        "skipped_samples": int(final_skipped_samples),
                        "dataset_progress": final_progress,
                        "dataset_meta": final_meta,
                        "dataset_ready": True,
                        "scheduler_state": scheduler.state_dict(),
                        "scheduler_total_steps": int(total_steps_ref["value"]),
                        "total_steps": int(total_steps_ref["value"]),
                        "warmup_steps": int(effective_warmup_steps),
                        "train_mode": cfg.train_mode,
                        "template_mode": cfg.template_mode,
                        "force_template": cfg.force_template,
                        "max_history_turns": cfg.max_history_turns,
                        "strict_whole_turns": True,
                        "sort_by_length": bool(cfg.sort_by_length),
                        "use_ngrams": bool(cfg.use_ngrams),
                        "ngram_summary": ngram_summary_text(ngram_state),
                        "val_loss": last_val_loss,
                        "perplexity": last_perplexity,
                        "validation_by_type": last_validation_by_type,
                        "best_val_loss": (best_val_loss if math.isfinite(best_val_loss) else None),
                        "epochs_without_improvement": epochs_without_improvement,
                        "early_stopped": False,
                        "scheduler_mode": f"fixed_{cfg.lr_schedule}",
                        "batch_plan": batch_plan_stats,
                        "dataloader_num_workers_effective": num_loader_workers,
                        "dataset_audit": dataset_audit_report,
                        "hardware_profile": hardware_profile,
                        "total_tokens": int(tokens_seen_ref["value"]),
                    }
                    status_payload.update(
                        _build_dataset_runtime_fields(
                            scheduler=scheduler,
                            cache_dir=cache_dir,
                            csv_total_samples_est=total_samples_est,
                        )
                    )
                    status_writer.write(status_payload)
                    LOGGER.info("Training sauber durch Stop-Signal beendet. Modell gespeichert nach: %s", outdir)
                except Exception as save_exc:
                    LOGGER.error("Speichern beim Stop fehlgeschlagen: %s", save_exc)
                    LOGGER.error(traceback.format_exc())
                    try:
                        status_writer.write(
                            {
                                "running": False,
                                "step": global_step,
                                "loss": last_loss,
                                "eta": "stopped",
                                "status": "stop_save_error",
                                "done": False,
                                "stopped": True,
                                "saved": False,
                                "error": f"{save_exc.__class__.__name__}: {save_exc}",
                                "save_dir": str(outdir),
                            }
                        )
                    except Exception:
                        pass
                    return 1

            release_training_memory(
                ctx=ctx,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                loader=loader,
                dataset=dataset,
                tokenizer=tokenizer,
            )
            model = None
            tokenizer = None
            optimizer = None
            scheduler = None
            scaler = None
            loader = None
            dataset = None
            gc.collect()

            barrier(ctx)
            return 0

        barrier(ctx)

        final_meta = read_json_if_exists(dataset_meta_path(cache_dir)) or {}
        final_progress = read_json_if_exists(dataset_progress_path(cache_dir)) or {}
        final_total_samples = int(final_meta.get("total_samples", final_progress.get("tokenized_samples", total_samples_est)))
        final_skipped_samples = int(final_meta.get("skipped_samples", final_progress.get("skipped_samples", 0)))
        final_cuda_mem = cuda_memory_snapshot(ctx.device)

        if ctx.is_main:
            save_model_artifacts(
                model=model,
                tokenizer=tokenizer,
                outdir=outdir,
                cfg=cfg,
                ctx=ctx,
                ngram_state=ngram_state,
                skipped_samples=final_skipped_samples,
            )

            final_payload = {
                "running": False,
                "step": global_step,
                "micro_step": micro_step,
                "loss": last_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "eta": "",
                "tokens_per_step": int(
                    batch_plan_stats["max_tokens_per_batch"]
                    * cfg.gradient_accumulation_steps
                    * max(1, ctx.world_size)
                ),
                "total_tokens": int(tokens_seen_ref["value"]),
                "done": True,
                "template_mode": cfg.template_mode,
                "force_template": cfg.force_template,
                "deterministic": cfg.deterministic,
                "allow_tf32": cfg.allow_tf32,
                "use_dataset_cache": cfg.use_dataset_cache,
                "cache_dir": str(cache_dir),
                "lr_decay_factor": cfg.lr_decay_factor,
                "scheduler_total_steps": int(total_steps_ref["value"]),
                "total_steps": int(total_steps_ref["value"]),
                "warmup_steps": effective_warmup_steps,
                "min_lr_ratio": cfg.min_lr_ratio,
                "train_mode": cfg.train_mode,
                "lora_r": cfg.lora_r,
                "lora_alpha": cfg.lora_alpha,
                "max_history_turns": cfg.max_history_turns,
                "strict_whole_turns": True,
                "sort_by_length": bool(cfg.sort_by_length),
                "use_ngrams": bool(cfg.use_ngrams),
                "ngram_summary": ngram_summary_text(ngram_state),
                "scheduler_state": scheduler.state_dict(),
                "cuda_memory": final_cuda_mem,
                "log_cuda_memory": bool(cfg.log_cuda_memory),
                "cuda_memory_log_interval_steps": int(cfg.cuda_memory_log_interval_steps),
                "cuda_empty_cache_interval_steps": int(cfg.cuda_empty_cache_interval_steps),
                "csv_total_samples_est": int(total_samples_est),
                "total_samples_real": int(final_total_samples),
                "skipped_samples": int(final_skipped_samples),
                "dataset_progress": final_progress,
                "dataset_meta": final_meta,
                "dataset_ready": bool(final_progress.get("done") or final_meta.get("done")),
                "scheduler_mode": f"fixed_{cfg.lr_schedule}",
                "val_split": cfg.val_split,
                "val_loss": last_val_loss,
                "perplexity": last_perplexity,
                "validation_by_type": last_validation_by_type,
                "best_val_loss": (best_val_loss if math.isfinite(best_val_loss) else None),
                "epochs_without_improvement": epochs_without_improvement,
                "early_stopped": early_stopped,
                "early_stopping_patience": cfg.early_stopping_patience,
                "best_model_dir": str(outdir / "best_model") if (outdir / "best_model").exists() else None,
                "batch_plan": batch_plan_stats,
                "dataloader_num_workers_effective": num_loader_workers,
                "dataset_audit": dataset_audit_report,
                "hardware_profile": hardware_profile,
            }
            status_writer.write(final_payload)
            LOGGER.info("Training abgeschlossen. Modell gespeichert nach: %s", outdir)

        release_training_memory(
            ctx=ctx,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            loader=loader,
            dataset=dataset,
            tokenizer=tokenizer,
        )
        model = None
        tokenizer = None
        optimizer = None
        scheduler = None
        scaler = None
        loader = None
        dataset = None
        gc.collect()

        barrier(ctx)
        return 0

    except Exception as e:
        LOGGER.error("Fataler Worker-Fehler: %s", e)
        LOGGER.error(traceback.format_exc())
        try:
            if ctx.device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception:
            pass
        if ctx.is_main:
            try:
                status_writer.write(
                    {
                        "running": False,
                        "error": f"{e.__class__.__name__}: {e}",
                    }
                )
            except Exception:
                pass
        return 1
    finally:
        cleanup_dist()


if __name__ == "__main__":
    raise SystemExit(main())
