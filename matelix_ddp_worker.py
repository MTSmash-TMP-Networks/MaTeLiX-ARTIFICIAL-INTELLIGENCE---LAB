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
import shutil
import signal
import sys
import time
import traceback
from contextlib import nullcontext
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

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

    learning_rate: float = 2e-4
    lr_schedule: str = "cosine"
    lr_decay_factor: float = 1.0
    warmup_steps: int = 0
    warmup_ratio: float = 0.0
    min_lr_ratio: float = 0.0
    lr_adjust_interval_steps: int = 25
    lr_adjust_min_change: float = 0.05

    adaptive_scheduler: bool = True
    adaptive_scheduler_freeze_on_producer_done: bool = True
    adaptive_scheduler_never_increase_lr: bool = True
    adaptive_scheduler_only_extend_steps: bool = True

    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    num_train_epochs: float = 3.0
    max_steps: Optional[int] = None
    max_seq_length: int = 1024
    chunk_size: Optional[int] = None
    max_history_turns: Optional[int] = None

    shuffle: bool = False
    sort_by_length: bool = True
    dataloader_num_workers: int = 0
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01

    precision_mode: str = "auto"
    gradient_checkpointing: bool = False

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
    merge_lora_on_save: bool = True

    ddp_find_unused_parameters: bool = True
    ddp_static_graph: bool = False
    ddp_broadcast_buffers: bool = False
    ddp_timeout_minutes: int = 30

    seed: int = 42
    deterministic: bool = False
    allow_tf32: bool = True
    use_ngrams: bool = False
    force_template: bool = True
    prefetch_factor: int = 2
    persistent_workers: bool = True

    use_dataset_cache: bool = True
    rebuild_dataset_cache: bool = False
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
        self.lr_adjust_interval_steps = max(1, int(self.lr_adjust_interval_steps))
        self.lr_adjust_min_change = max(0.0, float(self.lr_adjust_min_change))
        self.adaptive_scheduler = bool(self.adaptive_scheduler)
        self.adaptive_scheduler_freeze_on_producer_done = bool(self.adaptive_scheduler_freeze_on_producer_done)
        self.adaptive_scheduler_never_increase_lr = bool(self.adaptive_scheduler_never_increase_lr)
        self.adaptive_scheduler_only_extend_steps = bool(self.adaptive_scheduler_only_extend_steps)
        self.seed = int(self.seed)
        self.dataloader_num_workers = int(self.dataloader_num_workers)
        self.ddp_timeout_minutes = int(self.ddp_timeout_minutes)
        self.ddp_find_unused_parameters = bool(self.ddp_find_unused_parameters)
        self.ddp_static_graph = bool(self.ddp_static_graph)
        self.ddp_broadcast_buffers = bool(self.ddp_broadcast_buffers)
        self.force_template = bool(self.force_template)
        self.train_from_scratch = bool(self.train_from_scratch)
        self.include_prompt_loss = bool(self.include_prompt_loss)
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
        "resume", "save_every_epoch", "monitor_metric", "monitor_mode",
        "use_tensorboard", "val_csv", "val_split", "split_seed",
        "keep_last_k_checkpoints", "validate_every_epoch",
        "early_stopping_patience", "early_stopping_min_delta",
        "log_every_steps", "compile_model", "compile_mode",
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


def _iter_candidate_chains(csv_path: str, shuffle_threads: bool = False) -> Iterator[List[Dict[str, Any]]]:
    rows, id2row = _load_thread_rows(csv_path)
    candidates = [r for r in rows if (r.get("Assistentin") or "").strip() and r.get("id")]
    if not candidates:
        return

    from functools import lru_cache

    @lru_cache(maxsize=None)
    def root_and_depth(rid: str) -> Tuple[str, int]:
        depth = 0
        cur = id2row.get(rid)
        if not cur:
            return ("", 0)
        while True:
            pid = cur.get("parent_id", "")
            if not pid or pid not in id2row:
                return (cur["id"], depth)
            cur = id2row[pid]
            depth += 1

    threads: Dict[str, List[Tuple[int, int, Dict[str, Any]]]] = {}
    for r in candidates:
        root_id, depth = root_and_depth(r.get("id", ""))
        threads.setdefault(root_id, []).append((depth, int(r["_rowidx"]), r))

    order = list(threads.keys())
    if shuffle_threads:
        random.shuffle(order)

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
                yield chain


def chat_structured_iter(csv_path: str, shuffle_threads: bool = False) -> Iterator[StructuredChatSample]:
    for chain in _iter_candidate_chains(csv_path, shuffle_threads=shuffle_threads):
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

        yield StructuredChatSample(system=system_text, turns=turns, target_answer=answer)


def dialogplus_structured_iter(csv_path: str, shuffle_threads: bool = False) -> Iterator[StructuredChatSample]:
    for item in chat_structured_iter(csv_path, shuffle_threads=shuffle_threads):
        yield item


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


def build_examples_stream(cfg: TrainConfig) -> Iterator[Any]:
    if cfg.template_mode == "chat":
        return chat_structured_iter(cfg.csv_path, shuffle_threads=bool(cfg.shuffle))
    if cfg.template_mode == "dialogplus":
        return dialogplus_structured_iter(cfg.csv_path, shuffle_threads=bool(cfg.shuffle))
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


def tokenize_example(
    item: StructuredChatSample | str,
    tokenizer,
    max_seq_length: int,
    template_mode: str,
    max_history_turns: Optional[int],
    include_prompt_loss: bool = False,
) -> Optional[Dict[str, Any]]:
    if isinstance(item, str):
        ids = tokenizer(item, add_special_tokens=False)["input_ids"]
        if len(ids) > max_seq_length:
            return None
        labels = ids.copy()
        if len(ids) < 2:
            eos_or_pad = tokenizer.eos_token_id or tokenizer.pad_token_id or 0
            ids = ids + [eos_or_pad]
            labels = labels + [eos_or_pad]
        return {
            "input_ids": ids,
            "attention_mask": [1] * len(ids),
            "labels": labels,
            "seq_len": len(ids),
        }

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
    }


def count_examples_fast(cfg: TrainConfig) -> int:
    count = 0
    with open(cfg.csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if cfg.template_mode in {"chat", "dialogplus"}:
            for row in reader:
                rid = normalize_id(row.get("id", ""))
                asst = (row.get("Assistentin") or "").strip()
                if rid and asst:
                    count += 1
        else:
            for row in reader:
                txt = (row.get(cfg.column_name) or "").strip()
                if txt:
                    count += 1
    return count


def get_first_raw_example_preview(cfg: TrainConfig) -> Tuple[str, str]:
    try:
        examples_iter = build_examples_stream(cfg)
        first_item = next(examples_iter)
    except StopIteration:
        return "", ""
    except Exception:
        return "", ""

    try:
        if isinstance(first_item, StructuredChatSample):
            parts = []
            if first_item.system:
                parts.append(f"[SYSTEM]\n{first_item.system}")
            for t in first_item.turns:
                parts.append(f"[{t.role.upper()}]\n{t.content}")
            parts.append(f"[TARGET_ASSISTANT]\n{first_item.target_answer}")
            preview = "\n\n".join(parts)
        else:
            preview = str(first_item)
    except Exception:
        preview = ""

    return preview[:4000], preview[:20000]


def compute_shard_cache_dir(cfg: TrainConfig) -> Path:
    csv_path = Path(cfg.csv_path).expanduser().resolve()
    csv_stat = csv_path.stat()

    payload = {
        "csv_path": str(csv_path),
        "csv_mtime": int(csv_stat.st_mtime),
        "csv_size": int(csv_stat.st_size),
        "model_dir": str(Path(cfg.model_dir).expanduser().resolve()),
        "template_mode": cfg.template_mode,
        "column_name": cfg.column_name,
        "max_seq_length": int(cfg.max_seq_length),
        "sort_by_length": bool(cfg.sort_by_length),
        "max_history_turns": cfg.max_history_turns,
        "strict_whole_turns": True,
        "bucketed_shuffle": True,
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
    cache_root = Path(cfg.output_dir or cfg.save_dir or "./training_outputs/worker_run").expanduser().resolve() / "dataset_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    return cache_root / f"shards_{key}"


def shard_file_path(cache_dir: Path, shard_idx: int) -> Path:
    return cache_dir / f"shard_{shard_idx:06d}.pkl"


def producer_done_path(cache_dir: Path) -> Path:
    return cache_dir / "_producer_done.json"


def producer_error_path(cache_dir: Path) -> Path:
    return cache_dir / "_producer_error.txt"


def producer_meta_path(cache_dir: Path) -> Path:
    return cache_dir / "_producer_meta.json"


def producer_progress_path(cache_dir: Path) -> Path:
    return cache_dir / "_producer_progress.json"


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
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int, int]:
    if sort_by_length and pending_samples:
        pending_samples.sort(key=lambda s: int(s.get("seq_len") or len(s["input_ids"])))

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


def _write_producer_progress(
    cache_dir: Path,
    *,
    seen_samples: int,
    tokenized_samples: int,
    skipped_samples: int,
    shard_idx: int,
    done: bool,
) -> None:
    payload = {
        "seen_samples": int(seen_samples),
        "tokenized_samples": int(tokenized_samples),
        "skipped_samples": int(skipped_samples),
        "num_shards_written": int(shard_idx),
        "done": bool(done),
        "updated_at": time.time(),
    }
    _atomic_write_json(producer_progress_path(cache_dir), payload)


def shard_producer_process_main(cfg_dict: Dict[str, Any], cache_dir_str: str) -> None:
    try:
        cfg = TrainConfig(**cfg_dict)
        cfg.normalize()
        cache_dir = Path(cache_dir_str)
        cache_dir.mkdir(parents=True, exist_ok=True)

        tokenizer = AutoTokenizer.from_pretrained(cfg.model_dir, trust_remote_code=False, use_fast=True)
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
            csv_path=cfg.csv_path,
        )

        ngram_state = build_or_load_ngram_state(
            tokenizer=tokenizer,
            cfg=ngram_cfg,
            outdir=cache_dir,
            rebuild=bool(cfg.rebuild_dataset_cache),
        )

        shard_size = int(cfg.tokenized_shard_size)
        shard_idx = 0
        global_start = 0
        total_samples = 0
        skipped_samples = 0
        seen_samples = 0

        current_samples: List[Dict[str, Any]] = []
        pending_samples: List[Dict[str, Any]] = []

        if cfg.sort_by_length:
            sort_buffer_size = max(512, min(shard_size * 2, 20000))
        else:
            sort_buffer_size = shard_size

        examples_iter = build_examples_stream(cfg)

        _write_producer_progress(
            cache_dir,
            seen_samples=0,
            tokenized_samples=0,
            skipped_samples=0,
            shard_idx=0,
            done=False,
        )

        for item in examples_iter:
            seen_samples += 1

            sample = tokenize_example(
                item=item,
                tokenizer=tokenizer,
                max_seq_length=int(cfg.max_seq_length),
                template_mode=cfg.template_mode,
                max_history_turns=cfg.max_history_turns,
                include_prompt_loss=bool(cfg.include_prompt_loss),
            )

            if sample is None:
                skipped_samples += 1
                if seen_samples % 100 == 0:
                    _write_producer_progress(
                        cache_dir,
                        seen_samples=seen_samples,
                        tokenized_samples=total_samples,
                        skipped_samples=skipped_samples,
                        shard_idx=shard_idx,
                        done=False,
                    )
                continue

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
                )
                _write_producer_progress(
                    cache_dir,
                    seen_samples=seen_samples,
                    tokenized_samples=total_samples,
                    skipped_samples=skipped_samples,
                    shard_idx=shard_idx,
                    done=False,
                )

        if pending_samples:
            pending_samples, current_samples, shard_idx, global_start = _flush_pending_samples(
                cache_dir=cache_dir,
                pending_samples=pending_samples,
                current_samples=current_samples,
                shard_idx=shard_idx,
                global_start=global_start,
                shard_size=shard_size,
                sort_by_length=bool(cfg.sort_by_length),
            )

        if current_samples:
            _write_shard(cache_dir, shard_idx, global_start, current_samples)
            shard_idx += 1

        _write_producer_progress(
            cache_dir,
            seen_samples=seen_samples,
            tokenized_samples=total_samples,
            skipped_samples=skipped_samples,
            shard_idx=shard_idx,
            done=True,
        )

        meta = {
            "done": True,
            "num_shards": shard_idx,
            "total_samples": total_samples,
            "seen_samples": seen_samples,
            "skipped_samples": skipped_samples,
            "template_mode": cfg.template_mode,
            "max_seq_length": cfg.max_seq_length,
            "max_history_turns": cfg.max_history_turns,
            "strict_whole_turns": True,
            "sort_by_length": bool(cfg.sort_by_length),
            "sort_buffer_size": sort_buffer_size,
            "bucketed_shuffle": True,
            "use_ngrams": bool(cfg.use_ngrams),
            "ngram_summary": ngram_summary_text(ngram_state),
            "ngram_selected_count": int((ngram_state.stats or {}).get("selected_count", 0)) if ngram_state else 0,
        }
        _atomic_write_json(producer_meta_path(cache_dir), meta)
        _atomic_write_json(producer_done_path(cache_dir), {"done": True})

    except Exception as e:
        producer_error_path(Path(cache_dir_str)).write_text(
            f"{e.__class__.__name__}: {e}\n\n{traceback.format_exc()}",
            encoding="utf-8",
        )
        raise


def start_shard_producer(cfg: TrainConfig, cache_dir: Path, ctx: DistContext) -> Optional[mp.Process]:
    if not ctx.is_main:
        return None

    if cfg.rebuild_dataset_cache and cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)

    cache_dir.mkdir(parents=True, exist_ok=True)

    done_file = producer_done_path(cache_dir)
    error_file = producer_error_path(cache_dir)
    progress_file = producer_progress_path(cache_dir)

    if error_file.exists():
        error_file.unlink(missing_ok=True)

    if cfg.use_dataset_cache and done_file.exists() and not cfg.rebuild_dataset_cache:
        LOGGER.info("Fertiger Shard-Cache bereits vorhanden: %s", cache_dir)
        return None

    for p in cache_dir.glob("shard_*.pkl"):
        p.unlink(missing_ok=True)
    done_file.unlink(missing_ok=True)
    producer_meta_path(cache_dir).unlink(missing_ok=True)
    progress_file.unlink(missing_ok=True)

    proc = mp.Process(
        target=shard_producer_process_main,
        args=(cfg.__dict__.copy(), str(cache_dir)),
        daemon=True,
    )
    proc.start()
    LOGGER.info("Shard-Producer gestartet (pid=%s): %s", proc.pid, cache_dir)
    return proc


def wait_for_first_shard(cache_dir: Path, poll_sec: float = 1.0) -> None:
    while True:
        if producer_error_path(cache_dir).exists():
            raise RuntimeError(producer_error_path(cache_dir).read_text(encoding="utf-8"))
        if shard_file_path(cache_dir, 0).exists():
            return
        if producer_done_path(cache_dir).exists():
            meta_p = producer_meta_path(cache_dir)
            if meta_p.exists():
                meta = json.loads(meta_p.read_text(encoding="utf-8"))
                if int(meta.get("num_shards", 0)) == 0:
                    raise RuntimeError(
                        f"Shard-Producer fertig, aber kein Shard erzeugt. "
                        f"skipped_samples={meta.get('skipped_samples', 0)}"
                    )
            raise RuntimeError("Shard-Producer beendet, aber erster Shard fehlt.")
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


class TokenizedShardIterableDataset(IterableDataset):
    def __init__(
        self,
        cache_dir: Path,
        rank: int,
        world_size: int,
        shuffle: bool = False,
        sort_by_length: bool = True,
        epoch: int = 0,
    ):
        super().__init__()
        self.cache_dir = cache_dir
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.sort_by_length = sort_by_length
        self.epoch = epoch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = int(worker_info.id) if worker_info is not None else 0
        num_workers = int(worker_info.num_workers) if worker_info is not None else 1
        combined_world_size = max(1, int(self.world_size) * max(1, num_workers))
        combined_rank = int(self.rank) + worker_id * max(1, int(self.world_size))

        shard_idx = 0
        while True:
            path = shard_file_path(self.cache_dir, shard_idx)

            while True:
                if producer_error_path(self.cache_dir).exists():
                    raise RuntimeError(producer_error_path(self.cache_dir).read_text(encoding="utf-8"))
                if path.exists():
                    break
                if producer_done_path(self.cache_dir).exists():
                    return
                time.sleep(0.5)

            with open(path, "rb") as f:
                payload = pickle.load(f)

            samples = payload["samples"]
            global_start = int(payload["global_start"])

            if self.shuffle:
                rng = random.Random((self.epoch + 1) * 1000003 + shard_idx)
                if self.sort_by_length:
                    order = _make_bucketed_shuffle_order(samples, rng)
                else:
                    order = list(range(len(samples)))
                    rng.shuffle(order)
            else:
                order = list(range(len(samples)))

            for local_idx in order:
                global_idx = global_start + local_idx
                if (global_idx % combined_world_size) != combined_rank:
                    continue
                item = samples[local_idx]
                yield {
                    "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
                    "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
                    "labels": torch.tensor(item["labels"], dtype=torch.long),
                }

            shard_idx += 1


def _make_bucketed_shuffle_order(samples: List[Dict[str, Any]], rng: random.Random) -> List[int]:
    order = list(range(len(samples)))
    if not order:
        return order

    if not all(("seq_len" in s) for s in samples):
        rng.shuffle(order)
        return order

    order.sort(key=lambda i: int(samples[i].get("seq_len") or len(samples[i]["input_ids"])))

    bucket_size = max(8, min(64, len(order)))
    buckets = [order[i:i + bucket_size] for i in range(0, len(order), bucket_size)]

    for bucket in buckets:
        rng.shuffle(bucket)
    rng.shuffle(buckets)
    return [idx for bucket in buckets for idx in bucket]


class DataCollator:
    def __init__(self, pad_token_id: int, pad_to_multiple_of: int = 8):
        self.pad_token_id = int(pad_token_id)
        self.pad_to_multiple_of = int(pad_to_multiple_of)

    def __call__(self, features: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        max_len = max(int(x["input_ids"].numel()) for x in features)
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

    target_modules = []
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
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )
    model = get_peft_model(model, peft_cfg)
    try:
        model.print_trainable_parameters()
    except Exception:
        pass
    LOGGER.info("LoRA aktiv | r=%s alpha=%s targets=%s", cfg.lora_r, cfg.lora_alpha, ",".join(target_modules))
    return model


def build_model_and_tokenizer(cfg: TrainConfig, ctx: DistContext):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_dir, trust_remote_code=False, use_fast=True)
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
        csv_path=cfg.csv_path,
    )

    ngram_state = build_or_load_ngram_state(
        tokenizer=tokenizer,
        cfg=ngram_cfg,
        outdir=Path(cfg.output_dir or cfg.save_dir or "./training_outputs/worker_run"),
        rebuild=bool(cfg.rebuild_dataset_cache),
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
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_dir,
            trust_remote_code=False,
            torch_dtype=load_dtype,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        )
        LOGGER.info("Model init: pretrained weights | source=%s", cfg.model_dir)

    if need_resize or model.get_input_embeddings().weight.shape[0] != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))

    if ctx.device.type in {"cpu", "mps"} or (cfg.precision_mode or "").lower() == "fp32":
        model = model.to(torch.float32)

    if hasattr(model, "config"):
        model.config.use_cache = False

    model = apply_training_mode(model, cfg)

    if cfg.gradient_checkpointing:
        _enable_gradient_checkpointing(model)

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


class AdaptiveLRScheduler:
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
        adaptive_enabled: bool = True,
        never_increase_lr: bool = True,
        only_extend_steps: bool = True,
    ) -> None:
        self.optimizer = optimizer
        self.base_lr = float(base_lr)
        self.schedule = (schedule or "cosine").lower().strip()
        self.total_steps = max(1, int(total_steps))
        self.warmup_steps = max(0, int(warmup_steps))
        self.min_lr_ratio = min(1.0, max(0.0, float(min_lr_ratio)))
        self.lr_decay_factor = max(0.01, float(lr_decay_factor))
        self.adaptive_enabled = bool(adaptive_enabled)
        self.never_increase_lr = bool(never_increase_lr)
        self.only_extend_steps = bool(only_extend_steps)
        self.frozen = False
        self.last_lr = self.base_lr
        self.max_total_steps_seen = self.total_steps
        self.last_effective_total_samples = None
        self._apply_lr(self.get_lr_for_step(0), global_step=0)

    def _apply_lr(self, lr: float, global_step: int) -> float:
        lr = float(max(0.0, lr))
        if self.never_increase_lr and global_step > 0:
            lr = min(lr, float(self.last_lr))
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.last_lr = lr
        return lr

    def freeze(self) -> None:
        self.frozen = True

    def update_total_steps(self, total_steps: int) -> bool:
        total_steps = max(1, int(total_steps))
        if self.only_extend_steps:
            total_steps = max(self.total_steps, total_steps)
        changed = total_steps != self.total_steps
        self.total_steps = total_steps
        self.max_total_steps_seen = max(self.max_total_steps_seen, self.total_steps)
        return changed

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
        return self._apply_lr(self.get_lr_for_step(global_step), global_step=global_step)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "base_lr": self.base_lr,
            "schedule": self.schedule,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "min_lr_ratio": self.min_lr_ratio,
            "lr_decay_factor": self.lr_decay_factor,
            "last_lr": self.last_lr,
            "adaptive_enabled": self.adaptive_enabled,
            "never_increase_lr": self.never_increase_lr,
            "only_extend_steps": self.only_extend_steps,
            "frozen": self.frozen,
            "max_total_steps_seen": self.max_total_steps_seen,
            "last_effective_total_samples": self.last_effective_total_samples,
        }


def estimate_total_steps_from_samples(total_samples: int, cfg: TrainConfig, ctx: DistContext) -> int:
    total_samples = max(1, int(total_samples))
    local_samples = int(math.ceil(total_samples / max(1, ctx.world_size)))
    batches_per_epoch = max(1, int(math.ceil(local_samples / max(1, cfg.per_device_train_batch_size))))
    updates_per_epoch = max(1, int(math.ceil(batches_per_epoch / max(1, cfg.gradient_accumulation_steps))))

    if cfg.max_steps is not None:
        return max(1, int(cfg.max_steps))

    return max(1, int(math.ceil(cfg.num_train_epochs * updates_per_epoch)))


def maybe_adjust_scheduler_from_progress(
    scheduler: AdaptiveLRScheduler,
    cfg: TrainConfig,
    ctx: DistContext,
    cache_dir: Path,
    csv_total_samples_est: int,
    global_step: int,
    force: bool = False,
) -> Tuple[bool, int, Dict[str, Any]]:
    progress = read_json_if_exists(producer_progress_path(cache_dir)) or {}
    meta = read_json_if_exists(producer_meta_path(cache_dir)) or {}
    producer_done = bool(progress.get("done") or meta.get("done"))

    effective_total_samples = max(1, int(csv_total_samples_est))
    source = "csv_estimate"

    if meta:
        total_samples_real = int(meta.get("total_samples", 0))
        if total_samples_real > 0:
            effective_total_samples = total_samples_real
            source = "producer_meta_final"
    elif progress:
        seen_samples = int(progress.get("seen_samples", 0))
        tokenized_samples = int(progress.get("tokenized_samples", 0))
        if seen_samples > 0 and tokenized_samples > 0 and csv_total_samples_est > 0:
            projected = int(round(csv_total_samples_est * (tokenized_samples / float(seen_samples))))
            if projected > 0:
                effective_total_samples = projected
                source = "producer_progress_projected"

    proposed_total_steps = estimate_total_steps_from_samples(effective_total_samples, cfg, ctx)
    current_total_steps = max(1, int(scheduler.total_steps))
    rel_change = abs(proposed_total_steps - current_total_steps) / float(max(1, current_total_steps))

    info = {
        "source": ("static" if not cfg.adaptive_scheduler else source),
        "effective_total_samples": int(effective_total_samples),
        "proposed_total_steps": int(proposed_total_steps),
        "current_total_steps": int(current_total_steps),
        "relative_change": float(rel_change),
        "progress": progress,
        "meta": meta,
        "producer_done": producer_done,
        "adaptive_active": bool(cfg.adaptive_scheduler and not scheduler.frozen),
    }

    if not cfg.adaptive_scheduler:
        scheduler.last_effective_total_samples = int(effective_total_samples)
        return False, current_total_steps, info

    if scheduler.frozen and not force:
        scheduler.last_effective_total_samples = int(effective_total_samples)
        return False, current_total_steps, info

    should_update = force or (proposed_total_steps != current_total_steps and rel_change >= cfg.lr_adjust_min_change)
    changed = False

    if should_update:
        changed = scheduler.update_total_steps(proposed_total_steps)
        scheduler.last_effective_total_samples = int(effective_total_samples)
        if changed:
            scheduler.step(global_step)

    if producer_done and cfg.adaptive_scheduler_freeze_on_producer_done:
        scheduler.freeze()

    return changed, scheduler.total_steps, info


def _build_live_runtime_fields(
    *,
    scheduler: AdaptiveLRScheduler,
    cache_dir: Path,
    csv_total_samples_est: int,
    cfg: TrainConfig,
) -> Dict[str, Any]:
    progress = read_json_if_exists(producer_progress_path(cache_dir)) or {}
    meta = read_json_if_exists(producer_meta_path(cache_dir)) or {}

    source = "csv_estimate"
    effective_total_samples = int(max(1, csv_total_samples_est))

    if meta:
        total_samples_real = int(meta.get("total_samples", 0))
        if total_samples_real > 0:
            effective_total_samples = total_samples_real
            source = "producer_meta_final"
    elif progress:
        seen_samples = int(progress.get("seen_samples", 0))
        tokenized_samples = int(progress.get("tokenized_samples", 0))
        if seen_samples > 0 and tokenized_samples > 0 and csv_total_samples_est > 0:
            projected = int(round(csv_total_samples_est * (tokenized_samples / float(seen_samples))))
            if projected > 0:
                effective_total_samples = projected
                source = "producer_progress_projected"

    current_total_steps = max(1, int(scheduler.total_steps))
    rel_change = 0.0

    total_samples_real = None
    skipped_samples = None

    if meta:
        if meta.get("total_samples") is not None:
            total_samples_real = int(meta["total_samples"])
        if meta.get("skipped_samples") is not None:
            skipped_samples = int(meta["skipped_samples"])
    elif progress:
        if progress.get("tokenized_samples") is not None:
            total_samples_real = int(progress["tokenized_samples"])
        if progress.get("skipped_samples") is not None:
            skipped_samples = int(progress["skipped_samples"])

    return {
        "csv_total_samples_est": int(csv_total_samples_est),
        "total_samples_real": total_samples_real,
        "skipped_samples": skipped_samples,
        "producer_progress": progress,
        "producer_meta": meta,
        "scheduler_state": scheduler.state_dict(),
        "scheduler_source": source,
        "projected_samples": int(effective_total_samples),
        "scheduler_rel_change": float(rel_change),
        "producer_done": bool(progress.get("done") or meta.get("done")),
        "adaptive_scheduler": bool(cfg.adaptive_scheduler),
        "adaptive_scheduler_frozen": bool(scheduler.frozen),
        "adaptive_scheduler_never_increase_lr": bool(cfg.adaptive_scheduler_never_increase_lr),
        "adaptive_scheduler_only_extend_steps": bool(cfg.adaptive_scheduler_only_extend_steps),
        "scheduler_mode": (f"adaptive_{cfg.lr_schedule}" if cfg.adaptive_scheduler else str(cfg.lr_schedule)),
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


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: AdaptiveLRScheduler,
    scaler: Any,
    cfg: TrainConfig,
    ctx: DistContext,
    epoch: int,
    global_step: int,
    total_steps_ref: Dict[str, int],
    csv_total_samples_est: int,
    cache_dir: Path,
    train_start_time: float,
    status_writer: JsonStatusWriter,
    preview_writer: Optional[JsonPreviewWriter],
    tokenizer,
) -> Tuple[float, int, bool]:
    model.train()

    _, fp16, bf16 = pick_precision(cfg, ctx.device)
    amp_dtype = torch.float16 if fp16 else (torch.bfloat16 if bf16 else None)
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=amp_dtype)
        if (ctx.device.type == "cuda" and amp_dtype is not None)
        else nullcontext()
    )

    optimizer.zero_grad(set_to_none=True)

    running_loss = 0.0
    running_updates = 0
    reached_max_steps = False
    accum_counter = 0
    last_micro_loss_value: Optional[float] = None

    join_ctx = model.join() if (ctx.is_distributed and isinstance(model, DDP)) else nullcontext()

    with join_ctx:
        for batch in loader:
            # DDP-sicherer Stop:
            # Kein sync_stop()/dist.all_reduce() im Microbatch-Loop verwenden,
            # weil das mit DDP-Gradient-Allreduces kollidieren kann.
            # Der Launcher sendet SIGTERM an alle Worker; lokales Flag reicht.
            if SHUTDOWN.stop:
                reached_max_steps = True
                break

            batch = move_batch(batch, ctx.device)

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
                micro_step = accum_counter + 1
                should_step = micro_step >= cfg.gradient_accumulation_steps
                backward_sync_ctx = (
                    model.no_sync()
                    if (ctx.is_distributed and isinstance(model, DDP) and not should_step)
                    else nullcontext()
                )

                with backward_sync_ctx:
                    with autocast_ctx:
                        outputs = model(**batch)
                        loss = outputs.loss

                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Nicht-finite Loss erkannt: {float(loss.detach().item())}")

                    loss_value = float(loss.detach().item())
                    last_micro_loss_value = loss_value
                    loss_to_backprop = loss / cfg.gradient_accumulation_steps

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
                raise RuntimeError(f"CUDA OOM im Trainingsschritt. Empfehlung: kleinere max_seq_length oder Batch. Original: {oom}")

            accum_counter += 1
            should_step = accum_counter >= cfg.gradient_accumulation_steps

            if should_step:
                if scaler is not None and getattr(scaler, "is_enabled", lambda: False)():
                    scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)

                if scaler is not None and getattr(scaler, "is_enabled", lambda: False)():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                accum_counter = 0

                if (global_step % cfg.lr_adjust_interval_steps) == 0:
                    changed, proposed_total_steps, info = maybe_adjust_scheduler_from_progress(
                        scheduler=scheduler,
                        cfg=cfg,
                        ctx=ctx,
                        cache_dir=cache_dir,
                        csv_total_samples_est=csv_total_samples_est,
                        global_step=global_step,
                        force=False,
                    )
                    total_steps_ref["value"] = scheduler.total_steps
                    if ctx.is_main and changed:
                        LOGGER.info(
                            "Scheduler angepasst | source=%s total_steps=%s -> %s rel_change=%.4f samples=%s frozen=%s",
                            info["source"],
                            info["current_total_steps"],
                            proposed_total_steps,
                            info["relative_change"],
                            info["effective_total_samples"],
                            scheduler.frozen,
                        )

                lr = scheduler.step(global_step)
                total_steps_ref["value"] = scheduler.total_steps

                maybe_empty_cuda_cache(cfg, ctx, global_step)
                cuda_mem = maybe_log_cuda_memory(
                    cfg=cfg,
                    ctx=ctx,
                    global_step=global_step,
                    prefix=f"step={global_step}",
                )

                running_updates += 1
                reduced_loss = (
                    all_reduce_mean(float(loss.detach().item()), ctx)
                    if ctx.is_distributed
                    else float(loss.detach().item())
                )
                running_loss += reduced_loss

                if ctx.is_main:
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
                        "loss": reduced_loss,
                        "learning_rate": lr,
                        "eta": eta,
                        "tokens_per_step": int(
                            cfg.max_seq_length
                            * cfg.per_device_train_batch_size
                            * cfg.gradient_accumulation_steps
                            * max(1, ctx.world_size)
                        ),
                        "total_tokens": int(
                            global_step
                            * cfg.max_seq_length
                            * cfg.per_device_train_batch_size
                            * cfg.gradient_accumulation_steps
                            * max(1, ctx.world_size)
                        ),
                        "epoch": epoch,
                        "total_steps": int(total_steps_ref["value"]),
                        "scheduler_total_steps": int(total_steps_ref["value"]),
                        "cuda_memory": cuda_mem,
                    }
                    payload.update(
                        _build_live_runtime_fields(
                            scheduler=scheduler,
                            cache_dir=cache_dir,
                            csv_total_samples_est=csv_total_samples_est,
                            cfg=cfg,
                        )
                    )
                    status_writer.write(payload)

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

        if scaler is not None and getattr(scaler, "is_enabled", lambda: False)():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)
        global_step += 1

        changed, proposed_total_steps, info = maybe_adjust_scheduler_from_progress(
            scheduler=scheduler,
            cfg=cfg,
            ctx=ctx,
            cache_dir=cache_dir,
            csv_total_samples_est=csv_total_samples_est,
            global_step=global_step,
            force=False,
        )
        total_steps_ref["value"] = scheduler.total_steps
        if ctx.is_main and changed:
            LOGGER.info(
                "Scheduler angepasst | source=%s total_steps=%s -> %s rel_change=%.4f samples=%s frozen=%s",
                info["source"],
                info["current_total_steps"],
                proposed_total_steps,
                info["relative_change"],
                info["effective_total_samples"],
                scheduler.frozen,
            )

        lr = scheduler.step(global_step)
        total_steps_ref["value"] = scheduler.total_steps

        maybe_empty_cuda_cache(cfg, ctx, global_step)
        cuda_mem = maybe_log_cuda_memory(
            cfg=cfg,
            ctx=ctx,
            global_step=global_step,
            prefix=f"step={global_step}",
        )

        running_updates += 1
        reduced_loss = float(last_micro_loss_value or 0.0) if ctx.is_distributed else all_reduce_mean(float(last_micro_loss_value or 0.0), ctx)
        running_loss += reduced_loss

        if ctx.is_main:
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
                "loss": reduced_loss,
                "learning_rate": lr,
                "eta": eta,
                "tokens_per_step": int(
                    cfg.max_seq_length
                    * cfg.per_device_train_batch_size
                    * cfg.gradient_accumulation_steps
                    * max(1, ctx.world_size)
                ),
                "total_tokens": int(
                    global_step
                    * cfg.max_seq_length
                    * cfg.per_device_train_batch_size
                    * cfg.gradient_accumulation_steps
                    * max(1, ctx.world_size)
                ),
                "epoch": epoch,
                "total_steps": int(total_steps_ref["value"]),
                "scheduler_total_steps": int(total_steps_ref["value"]),
                "cuda_memory": cuda_mem,
            }
            payload.update(
                _build_live_runtime_fields(
                    scheduler=scheduler,
                    cache_dir=cache_dir,
                    csv_total_samples_est=csv_total_samples_est,
                    cfg=cfg,
                )
            )
            status_writer.write(payload)

    avg_loss = running_loss / max(1, running_updates)
    return avg_loss, global_step, reached_max_steps



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
        "bucketed_shuffle": True,
        "use_ngrams": bool(cfg.use_ngrams),
        "ngram_summary": ngram_summary_text(ngram_state),
    }
    if skipped_samples is not None:
        template_info["skipped_samples"] = int(skipped_samples)


    (outdir / "template_info.json").write_text(
        json.dumps(template_info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


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
    producer_proc: Optional[mp.Process] = None,
) -> None:
    """Versucht Trainings-RAM/VRAM vor Prozessende explizit freizugeben.

    Hinweis: Die Funktion leert interne Strukturen und verschiebt das Modell auf CPU.
    Die aufrufende Funktion setzt danach ihre lokalen Variablen auf None, damit die
    letzten starken Referenzen verschwinden und gc/empty_cache wirklich greifen.
    """
    try:
        if producer_proc is not None and producer_proc.is_alive():
            producer_proc.terminate()
            producer_proc.join(timeout=5)
    except Exception:
        pass

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

    producer_proc: Optional[mp.Process] = None
    ngram_state = None

    try:
        set_seed(cfg.seed + ctx.rank, deterministic=cfg.deterministic)

        LOGGER.info(
            "Worker gestartet | rank=%s local_rank=%s world_size=%s device=%s",
            ctx.rank, ctx.local_rank, ctx.world_size, ctx.device
        )
        LOGGER.info("Train mode: %s | train_from_scratch=%s | include_prompt_loss=%s", cfg.train_mode, cfg.train_from_scratch, cfg.include_prompt_loss)
        LOGGER.info("Config: %s", json.dumps(cfg.__dict__, ensure_ascii=False))

        model, tokenizer, fp16, bf16, ngram_state = build_model_and_tokenizer(cfg, ctx)

        total_samples_est = count_examples_fast(cfg)
        if total_samples_est <= 0:
            raise RuntimeError("Kein Trainingssample gefunden.")

        if ctx.is_main:
            try:
                preview, preview_full = get_first_raw_example_preview(cfg)
                if preview_writer is not None and (preview or preview_full):
                    preview_writer.write(preview, preview_full)
            except Exception:
                pass

        cache_dir = compute_shard_cache_dir(cfg)
        producer_proc = start_shard_producer(cfg, cache_dir, ctx)

        if ctx.is_main:
            LOGGER.info("Warte auf ersten tokenisierten Shard ...")
        wait_for_first_shard(cache_dir)

        barrier(ctx)

        dataset = TokenizedShardIterableDataset(
            cache_dir=cache_dir,
            rank=ctx.rank,
            world_size=ctx.world_size,
            shuffle=bool(cfg.shuffle),
            sort_by_length=bool(cfg.sort_by_length),
            epoch=0,
        )
        collator = DataCollator(tokenizer.pad_token_id or tokenizer.eos_token_id or 0, pad_to_multiple_of=8)

        initial_total_steps = estimate_total_steps_from_samples(total_samples_est, cfg, ctx)

        effective_warmup_steps = cfg.warmup_steps
        if effective_warmup_steps <= 0 and cfg.warmup_ratio > 0.0:
            effective_warmup_steps = int(math.ceil(initial_total_steps * cfg.warmup_ratio))
        effective_warmup_steps = max(0, min(effective_warmup_steps, max(0, initial_total_steps - 1)))

        num_loader_workers = max(0, int(cfg.dataloader_num_workers))
        loader_kwargs: Dict[str, Any] = {
            "dataset": dataset,
            "batch_size": cfg.per_device_train_batch_size,
            "num_workers": num_loader_workers,
            "pin_memory": (ctx.device.type == "cuda"),
            "collate_fn": collator,
        }
        if num_loader_workers > 0:
            loader_kwargs["prefetch_factor"] = max(1, int(cfg.prefetch_factor))
            loader_kwargs["persistent_workers"] = bool(cfg.persistent_workers)

        loader = DataLoader(**loader_kwargs)

        optimizer = build_optimizer(model, cfg)
        scheduler = AdaptiveLRScheduler(
            optimizer=optimizer,
            base_lr=cfg.learning_rate,
            schedule=cfg.lr_schedule,
            total_steps=initial_total_steps,
            warmup_steps=effective_warmup_steps,
            min_lr_ratio=cfg.min_lr_ratio,
            lr_decay_factor=cfg.lr_decay_factor,
            adaptive_enabled=cfg.adaptive_scheduler,
            never_increase_lr=cfg.adaptive_scheduler_never_increase_lr,
            only_extend_steps=cfg.adaptive_scheduler_only_extend_steps,
        )
        scaler = make_scaler(fp16=fp16, device=ctx.device)

        changed, proposed_total_steps, info = maybe_adjust_scheduler_from_progress(
            scheduler=scheduler,
            cfg=cfg,
            ctx=ctx,
            cache_dir=cache_dir,
            csv_total_samples_est=total_samples_est,
            global_step=0,
            force=True,
        )
        total_steps_ref = {"value": scheduler.total_steps}

        if ctx.is_main:
            LOGGER.info(
                "Scheduler initialisiert | schedule=%s total_steps=%s warmup_steps=%s min_lr_ratio=%s lr_decay_factor=%s source=%s adaptive=%s freeze_on_done=%s never_increase_lr=%s only_extend_steps=%s",
                cfg.lr_schedule,
                scheduler.total_steps,
                effective_warmup_steps,
                cfg.min_lr_ratio,
                cfg.lr_decay_factor,
                info["source"],
                cfg.adaptive_scheduler,
                cfg.adaptive_scheduler_freeze_on_producer_done,
                cfg.adaptive_scheduler_never_increase_lr,
                cfg.adaptive_scheduler_only_extend_steps,
            )
            LOGGER.info(
                "Samples initial | csv_estimate=%s effective_samples=%s proposed_total_steps=%s",
                total_samples_est,
                info["effective_total_samples"],
                proposed_total_steps,
            )
            LOGGER.info(
                "Strict whole-turn packing aktiv | max_history_turns=%s | no partial turns | oversize samples skipped",
                cfg.max_history_turns,
            )
            LOGGER.info(
                "sort_by_length=%s | shuffle=%s | bucketed_shuffle=%s",
                cfg.sort_by_length,
                cfg.shuffle,
                bool(cfg.shuffle and cfg.sort_by_length),
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
                "step": 0,
                "loss": None,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "eta": "",
                "tokens_per_step": int(
                    cfg.max_seq_length
                    * cfg.per_device_train_batch_size
                    * cfg.gradient_accumulation_steps
                    * max(1, ctx.world_size)
                ),
                "total_tokens": 0,
                "epoch": 0,
                "total_steps": int(total_steps_ref["value"]),
                "scheduler_total_steps": int(total_steps_ref["value"]),
                "warmup_steps": int(effective_warmup_steps),
                "cuda_memory": initial_cuda_mem,
                "log_cuda_memory": bool(cfg.log_cuda_memory),
                "cuda_memory_log_interval_steps": int(cfg.cuda_memory_log_interval_steps),
                "cuda_empty_cache_interval_steps": int(cfg.cuda_empty_cache_interval_steps),
                "adaptive_scheduler": bool(cfg.adaptive_scheduler),
                "adaptive_scheduler_frozen": bool(scheduler.frozen),
                "adaptive_scheduler_never_increase_lr": bool(cfg.adaptive_scheduler_never_increase_lr),
                "adaptive_scheduler_only_extend_steps": bool(cfg.adaptive_scheduler_only_extend_steps),
                "scheduler_mode": (f"adaptive_{cfg.lr_schedule}" if cfg.adaptive_scheduler else str(cfg.lr_schedule)),
            }
            payload.update(
                _build_live_runtime_fields(
                    scheduler=scheduler,
                    cache_dir=cache_dir,
                    csv_total_samples_est=total_samples_est,
                    cfg=cfg,
                )
            )
            status_writer.write(payload)

        model = wrap_ddp(model, cfg, ctx)
        barrier(ctx)

        global_step = 0
        last_loss = None
        train_start_time = time.time()

        epochs = max(1, int(math.ceil(cfg.num_train_epochs)))
        for epoch in range(epochs):
            if cfg.max_steps is not None and global_step >= int(cfg.max_steps):
                break
            if global_step >= int(total_steps_ref["value"]):
                break

            dataset.set_epoch(epoch)

            avg_loss, global_step, reached_max_steps = train_epoch(
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
                csv_total_samples_est=total_samples_est,
                cache_dir=cache_dir,
                train_start_time=train_start_time,
                status_writer=status_writer,
                preview_writer=preview_writer,
                tokenizer=tokenizer,
            )
            last_loss = avg_loss

            if ctx.is_main:
                LOGGER.info("Epoche %d abgeschlossen | avg_loss=%.6f", epoch, avg_loss)

            if reached_max_steps or SHUTDOWN.stop:
                break

        if SHUTDOWN.stop:
            if producer_proc is not None and producer_proc.is_alive():
                try:
                    producer_proc.terminate()
                    producer_proc.join(timeout=5)
                except Exception:
                    pass

            # Bei Stop trotzdem einen nutzbaren Zwischenstand speichern.
            # Wichtig: Nur rank 0 schreibt Dateien; danach warten alle Ranks am Barrier.
            if ctx.is_main:
                try:
                    final_meta = read_json_if_exists(producer_meta_path(cache_dir)) or {}
                    final_progress = read_json_if_exists(producer_progress_path(cache_dir)) or {}
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
                        "producer_progress": final_progress,
                        "producer_meta": final_meta,
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
                        "bucketed_shuffle": True,
                        "use_ngrams": bool(cfg.use_ngrams),
                        "ngram_summary": ngram_summary_text(ngram_state),
                        "adaptive_scheduler": bool(cfg.adaptive_scheduler),
                        "adaptive_scheduler_frozen": bool(scheduler.frozen),
                        "adaptive_scheduler_never_increase_lr": bool(cfg.adaptive_scheduler_never_increase_lr),
                        "adaptive_scheduler_only_extend_steps": bool(cfg.adaptive_scheduler_only_extend_steps),
                        "scheduler_mode": (
                            f"adaptive_{cfg.lr_schedule}" if cfg.adaptive_scheduler else str(cfg.lr_schedule)
                        ),
                    }
                    status_payload.update(
                        _build_live_runtime_fields(
                            scheduler=scheduler,
                            cache_dir=cache_dir,
                            csv_total_samples_est=total_samples_est,
                            cfg=cfg,
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
                producer_proc=producer_proc,
            )
            model = None
            tokenizer = None
            optimizer = None
            scheduler = None
            scaler = None
            loader = None
            dataset = None
            producer_proc = None
            gc.collect()

            barrier(ctx)
            return 0

        barrier(ctx)

        changed, proposed_total_steps, info = maybe_adjust_scheduler_from_progress(
            scheduler=scheduler,
            cfg=cfg,
            ctx=ctx,
            cache_dir=cache_dir,
            csv_total_samples_est=total_samples_est,
            global_step=global_step,
            force=True,
        )
        total_steps_ref["value"] = scheduler.total_steps
        if ctx.is_main and changed:
            LOGGER.info(
                "Finale Scheduler-Anpassung | source=%s total_steps=%s frozen=%s",
                info["source"],
                total_steps_ref["value"],
                scheduler.frozen,
            )

        if producer_proc is not None and producer_proc.is_alive():
            producer_proc.join(timeout=5)

        final_meta = read_json_if_exists(producer_meta_path(cache_dir)) or {}
        final_progress = read_json_if_exists(producer_progress_path(cache_dir)) or {}
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
                "loss": last_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "eta": "",
                "tokens_per_step": int(
                    cfg.max_seq_length
                    * cfg.per_device_train_batch_size
                    * cfg.gradient_accumulation_steps
                    * max(1, ctx.world_size)
                ),
                "total_tokens": int(
                    global_step
                    * cfg.max_seq_length
                    * cfg.per_device_train_batch_size
                    * cfg.gradient_accumulation_steps
                    * max(1, ctx.world_size)
                ),
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
                "bucketed_shuffle": True,
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
                "producer_progress": final_progress,
                "producer_meta": final_meta,
                "scheduler_source": info["source"],
                "projected_samples": int(info["effective_total_samples"]),
                "scheduler_rel_change": float(info["relative_change"]),
                "producer_done": bool(final_progress.get("done") or final_meta.get("done")),
                "adaptive_scheduler": bool(cfg.adaptive_scheduler),
                "adaptive_scheduler_frozen": bool(scheduler.frozen),
                "adaptive_scheduler_never_increase_lr": bool(cfg.adaptive_scheduler_never_increase_lr),
                "adaptive_scheduler_only_extend_steps": bool(cfg.adaptive_scheduler_only_extend_steps),
                "scheduler_mode": (f"adaptive_{cfg.lr_schedule}" if cfg.adaptive_scheduler else str(cfg.lr_schedule)),
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
            producer_proc=producer_proc,
        )
        model = None
        tokenizer = None
        optimizer = None
        scheduler = None
        scaler = None
        loader = None
        dataset = None
        producer_proc = None
        gc.collect()

        barrier(ctx)
        return 0

    except Exception as e:
        LOGGER.error("Fataler Worker-Fehler: %s", e)
        LOGGER.error(traceback.format_exc())
        try:
            if producer_proc is not None and producer_proc.is_alive():
                producer_proc.terminate()
        except Exception:
            pass
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
