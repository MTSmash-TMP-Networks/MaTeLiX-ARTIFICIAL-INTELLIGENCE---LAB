#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
matelix_ddp_worker.py

Vollständiger, web-kompatibler DDP-Worker für MaTeLiX.
Korrigiert u. a.:
- DDP init_process_group: kein int mehr als device_id
- save_dir Alias-Kompatibilität
- unbekannte JSON-Felder werden ignoriert
- robustes Gradient Checkpointing unter DDP
- find_unused_parameters konfigurierbar
- rank-sicheres Logging / Status-Datei
- Trainingsdaten: text / chat / dialogplus
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
import signal
import sys
import time
import traceback
from contextlib import nullcontext
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from torch.amp import GradScaler
    _NEW_SCALER = True
except Exception:
    from torch.cuda.amp import GradScaler
    _NEW_SCALER = False

csv.field_size_limit(1024 * 1024 * 128)

LOGGER = logging.getLogger("matelix_ddp_worker")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

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
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    num_train_epochs: float = 3.0
    max_steps: Optional[int] = None
    max_seq_length: int = 1024
    chunk_size: Optional[int] = None

    shuffle: bool = False
    sort_by_length: bool = True
    dataloader_num_workers: int = 0
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01

    precision_mode: str = "auto"
    gradient_checkpointing: bool = False

    train_mode: str = "full"
    lora_r: int = 8
    lora_alpha: int = 16
    merge_lora_on_save: bool = True

    ddp_find_unused_parameters: bool = False
    ddp_static_graph: bool = False
    ddp_broadcast_buffers: bool = False
    ddp_timeout_minutes: int = 30

    seed: int = 42

    use_ngrams: bool = False

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
        self.seed = int(self.seed)
        self.dataloader_num_workers = int(self.dataloader_num_workers)
        self.ddp_timeout_minutes = int(self.ddp_timeout_minutes)
        self.ddp_find_unused_parameters = bool(self.ddp_find_unused_parameters)
        self.ddp_static_graph = bool(self.ddp_static_graph)
        self.ddp_broadcast_buffers = bool(self.ddp_broadcast_buffers)


def _coerce_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(payload or {})

    if "save_dir" in payload and "output_dir" not in payload:
        payload["output_dir"] = payload["save_dir"]

    # web/ddp launcher noise safely ignored
    ignore_keys = {
        "nproc_per_node", "nnodes", "node_rank", "master_addr", "master_port",
        "world_size", "local_rank", "rank", "run_name", "experiment_name",
        "resume", "save_every_epoch", "monitor_metric", "monitor_mode",
        "use_tensorboard", "val_csv", "val_split", "split_seed",
        "keep_last_k_checkpoints", "validate_every_epoch",
        "early_stopping_patience", "early_stopping_min_delta",
        "log_every_steps", "compile_model", "compile_mode", "tf32",
        "persistent_workers", "prefetch_factor", "pin_memory",
        "scheduler", "warmup_steps", "warmup_ratio", "min_lr_ratio",
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


# ---------------------------------------------------------------------------
# DDP / state
# ---------------------------------------------------------------------------

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


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
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

        # Wichtig: Bei dieser Torch-Version darf device_id kein int sein.
        # Entweder torch.device oder weglassen. Wir geben für CUDA korrekt torch.device mit.
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


# ---------------------------------------------------------------------------
# Logging / status files
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Tokenizer / dataset helpers
# ---------------------------------------------------------------------------

def normalize_id(val: Any) -> str:
    if val is None:
        return ""
    s = str(val).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


def prepare_tokenizer(tokenizer) -> bool:
    need_resize = False
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        need_resize = True

    added = tokenizer.add_tokens(["<|System|>", "<|Benutzer|>", "<|Assistentin|>"], special_tokens=False)
    if added > 0:
        need_resize = True

    tokenizer.padding_side = "left"
    if not getattr(tokenizer, "chat_template", None):
        tokenizer.chat_template = """{{ bos_token }}
{% for message in messages %}
{% if message.role == 'system' %}<|System|>
{{ message.content }}
{% elif message.role == 'user' %}<|Benutzer|>
{{ message.content }}
{% elif message.role == 'assistant' %}<|Assistentin|>
{{ message.content }}
{% endif %}
{% endfor %}
{% if add_generation_prompt %}<|Assistentin|>
{% else %}</s>{% endif %}"""
    return need_resize


def column_iter(csv_path: str, column_name: str) -> Iterator[str]:
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            txt = (row.get(column_name) or "").strip()
            if txt:
                yield txt


def chat_block_iter(csv_path: str, shuffle_threads: bool = False) -> Iterator[Tuple[str, str]]:
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows: List[Dict[str, Any]] = []
        for idx, row in enumerate(reader):
            row["_rowidx"] = idx
            row["id"] = normalize_id(row.get("id", ""))
            row["parent_id"] = normalize_id(row.get("parent_id", ""))
            rows.append(row)

    id2row = {r["id"]: r for r in rows if r.get("id")}
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
            if not chain:
                continue

            target_idx = len(chain) - 1
            answer = (chain[target_idx].get("Assistentin") or "").strip()
            if not answer:
                continue

            system_text = (chain[0].get("system") or "").strip()
            parts: List[str] = ["<s>\n"]
            if system_text:
                parts.extend(["<|System|>\n", system_text, "\n"])

            for j in range(target_idx + 1):
                turn = chain[j]
                user = (turn.get("Benutzer") or "").strip()
                ctx = (turn.get("Kontext") or "").strip()
                asst = (turn.get("Assistentin") or "").strip()

                if user:
                    parts.append("<|Benutzer|>\n")
                    parts.append(f"{ctx}\n{user}".strip() if ctx else user)
                    parts.append("\n")

                if j < target_idx and asst:
                    parts.append("<|Assistentin|>\n")
                    parts.append(asst)
                    parts.append("\n")
                elif j == target_idx:
                    parts.append("<|Assistentin|>\n")

            yield "".join(parts), answer + "\n</s>"


def dialogplus_block_iter(csv_path: str, shuffle_threads: bool = False) -> Iterator[Tuple[str, str]]:
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows: List[Dict[str, Any]] = []
        for idx, row in enumerate(reader):
            row["_rowidx"] = idx
            row["id"] = normalize_id(row.get("id", ""))
            row["parent_id"] = normalize_id(row.get("parent_id", ""))
            rows.append(row)

    id2row = {r["id"]: r for r in rows if r.get("id")}
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
            if not chain:
                continue

            target_idx = len(chain) - 1
            answer = (chain[target_idx].get("Assistentin") or "").strip()
            if not answer:
                continue

            system_text = (chain[0].get("system") or "").strip()
            parts: List[str] = []
            if system_text:
                parts.extend(["<|System|>\n", system_text, "\n</s>"])

            for j in range(target_idx + 1):
                turn = chain[j]
                user = (turn.get("Benutzer") or "").strip()
                ctx = (turn.get("Kontext") or "").strip()
                asst = (turn.get("Assistentin") or "").strip()

                if user:
                    parts.append("\n<|Benutzer|>\n")
                    parts.append(f"{ctx}\n{user}".strip() if ctx else user)
                    parts.append("\n</s>")

                if j < target_idx and asst:
                    parts.append("\n<|Assistentin|>\n")
                    parts.append(asst)
                    parts.append("\n</s>")
                elif j == target_idx:
                    parts.append("\n<|Assistentin|>\n")

            yield "".join(parts), answer + "\n</s>"



def _find_last_subsequence(haystack: List[int], needle: List[int]) -> int:
    if not haystack or not needle or len(needle) > len(haystack):
        return -1
    n = len(needle)
    for i in range(len(haystack) - n, -1, -1):
        if haystack[i:i+n] == needle:
            return i
    return -1


def _find_first_subsequence(haystack: List[int], needle: List[int]) -> int:
    if not haystack or not needle or len(needle) > len(haystack):
        return -1
    n = len(needle)
    for i in range(0, len(haystack) - n + 1):
        if haystack[i:i+n] == needle:
            return i
    return -1


def _trim_window_to_turn_boundary(
    token_ids: List[int],
    max_len: int,
    turn_markers: List[List[int]],
) -> List[int]:
    """
    Nimmt immer das rechte Ende des Fensters und versucht den linken Rand
    auf eine Turn-Grenze (<|System|>, <|Benutzer|>, <|Assistentin|>) zu legen.
    Wenn keine Grenze im Fenster gefunden wird, bleibt das rechte Fenster erhalten.
    """
    if len(token_ids) <= max_len:
        return token_ids

    window = token_ids[-max_len:]
    best_start = 0
    for marker_ids in turn_markers:
        pos = _find_first_subsequence(window, marker_ids)
        if pos != -1:
            best_start = max(best_start, pos)
    trimmed = window[best_start:]
    return trimmed if trimmed else window


class CausalLMDataset(Dataset):
    def __init__(self, examples: Sequence[Tuple[str, str] | str], tokenizer, max_seq_length: int, template_mode: str, sort_by_length: bool):
        self.tokenizer = tokenizer
        self.max_seq_length = int(max_seq_length)
        self.template_mode = template_mode
        self.samples: List[Any] = list(examples)

        # Turn-Marker für sauberes Ausrichten auf Dialoggrenzen
        self.system_marker_ids = tokenizer("<|System|>\n", add_special_tokens=False)["input_ids"]
        self.user_marker_ids = tokenizer("<|Benutzer|>\n", add_special_tokens=False)["input_ids"]
        self.assistant_marker_ids = tokenizer("<|Assistentin|>\n", add_special_tokens=False)["input_ids"]
        self.turn_markers = [
            self.system_marker_ids,
            self.user_marker_ids,
            self.assistant_marker_ids,
        ]

        if sort_by_length:
            def _len_fn(x: Any) -> int:
                if isinstance(x, tuple):
                    p, a = x
                    return len(tokenizer(p + a, add_special_tokens=False)["input_ids"])
                return len(tokenizer(x, add_special_tokens=False)["input_ids"])
            self.samples.sort(key=_len_fn)

    def __len__(self) -> int:
        return len(self.samples)

    def _truncate_dialog_sample(self, prompt_ids: List[int], answer_ids: List[int]) -> Tuple[List[int], List[int]]:
        """
        Zielregeln:
        - Links darf abgeschnitten werden.
        - Das Ende der Zielantwort soll immer erhalten bleiben.
        - Wenn die komplette Antwort nicht in max_seq_length passt, wird sie als
          eigener Assistentinnen-Turn behandelt und am linken Rand möglichst auf
          eine Turn-Grenze ausgerichtet.
        """
        total = len(prompt_ids) + len(answer_ids)

        # Fall 1: passt komplett
        if total <= self.max_seq_length:
            input_ids = prompt_ids + answer_ids
            labels = ([-100] * len(prompt_ids)) + answer_ids
            return input_ids, labels

        # Fall 2: komplette Antwort passt, also nur Prompt links kürzen
        if len(answer_ids) <= self.max_seq_length:
            keep_prompt = max(0, self.max_seq_length - len(answer_ids))
            trimmed_prompt = prompt_ids[-keep_prompt:] if keep_prompt > 0 else []
            trimmed_prompt = _trim_window_to_turn_boundary(trimmed_prompt, len(trimmed_prompt), self.turn_markers)
            input_ids = trimmed_prompt + answer_ids
            labels = ([-100] * len(trimmed_prompt)) + answer_ids
            return input_ids[-self.max_seq_length:], labels[-self.max_seq_length:]

        # Fall 3: selbst die Antwort ist länger als max_seq_length.
        # Dann NICHT mitten im gesamten Dialog schneiden, sondern die Antwort als
        # eigenen Turn behandeln: <|Assistentin|>\n + Antworttail.
        assistant_turn = list(self.assistant_marker_ids) + list(answer_ids)
        assistant_turn = _trim_window_to_turn_boundary(assistant_turn, self.max_seq_length, self.turn_markers)

        # Marker innerhalb des finalen Fensters suchen, damit der Turn sauber beginnt
        marker_pos = _find_first_subsequence(assistant_turn, self.assistant_marker_ids)
        if marker_pos == -1:
            # Falls der Marker nicht mehr vollständig im Fenster liegt, erzwingen wir
            # einen minimal sauberen Turn-Start.
            tail_budget = max(0, self.max_seq_length - len(self.assistant_marker_ids))
            answer_tail = answer_ids[-tail_budget:] if tail_budget > 0 else []
            assistant_turn = list(self.assistant_marker_ids) + answer_tail
            marker_pos = 0

        answer_start = marker_pos + len(self.assistant_marker_ids)
        input_ids = assistant_turn[-self.max_seq_length:]
        labels = ([-100] * answer_start) + input_ids[answer_start:]
        labels = labels[-len(input_ids):]
        return input_ids, labels

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.samples[idx]

        if isinstance(item, tuple):
            prompt, answer = item
            prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
            answer_ids = self.tokenizer(answer, add_special_tokens=False)["input_ids"]
            input_ids, labels = self._truncate_dialog_sample(prompt_ids, answer_ids)
        else:
            ids = self.tokenizer(item, add_special_tokens=False)["input_ids"]
            ids = ids[-self.max_seq_length:]
            ids = _trim_window_to_turn_boundary(ids, self.max_seq_length, self.turn_markers)
            input_ids = ids
            labels = ids.copy()

        if len(input_ids) < 2:
            eos_or_pad = self.tokenizer.eos_token_id or self.tokenizer.pad_token_id or 0
            input_ids = input_ids + [eos_or_pad]
            labels = labels + [eos_or_pad]

        attention_mask = [1] * len(input_ids)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


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


def build_examples(cfg: TrainConfig) -> List[Any]:
    if cfg.template_mode == "chat":
        return list(chat_block_iter(cfg.csv_path, shuffle_threads=bool(cfg.shuffle)))
    if cfg.template_mode == "dialogplus":
        return list(dialogplus_block_iter(cfg.csv_path, shuffle_threads=bool(cfg.shuffle)))
    return list(column_iter(cfg.csv_path, cfg.column_name))


# ---------------------------------------------------------------------------
# Model / optimizer / scheduler
# ---------------------------------------------------------------------------

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


def build_model_and_tokenizer(cfg: TrainConfig, ctx: DistContext):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_dir, trust_remote_code=False, use_fast=True)
    need_resize = prepare_tokenizer(tokenizer)

    load_dtype, fp16, bf16 = pick_precision(cfg, ctx.device)
    LOGGER.info("Precision: load_dtype=%s fp16=%s bf16=%s", load_dtype, fp16, bf16)

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_dir,
        trust_remote_code=False,
        torch_dtype=load_dtype,
        low_cpu_mem_usage=True,
    )

    if need_resize or model.get_input_embeddings().weight.shape[0] != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))

    # Wichtig: für MPS/CPU stabil auf float32
    if ctx.device.type in {"cpu", "mps"} or (cfg.precision_mode or "").lower() == "fp32":
        model = model.to(torch.float32)

    if hasattr(model, "config"):
        model.config.use_cache = False

    if cfg.gradient_checkpointing:
        _enable_gradient_checkpointing(model, cfg, ctx)

    model.to(ctx.device)
    return model, tokenizer, fp16, bf16


def _enable_gradient_checkpointing(model: nn.Module, cfg: TrainConfig, ctx: DistContext) -> None:
    if not hasattr(model, "gradient_checkpointing_enable"):
        LOGGER.info("Gradient Checkpointing nicht verfügbar.")
        return

    # h2o-artiger kompatibler Pfad: zuerst non-reentrant versuchen
    enabled = False
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        enabled = True
        LOGGER.info("Gradient Checkpointing: ON (non-reentrant)")
    except TypeError:
        try:
            model.gradient_checkpointing_enable()
            enabled = True
            LOGGER.info("Gradient Checkpointing: ON (legacy API)")
        except Exception as e:
            LOGGER.warning("Gradient Checkpointing konnte nicht aktiviert werden: %s", e)

    if enabled and hasattr(model, "enable_input_require_grads"):
        try:
            model.enable_input_require_grads()
        except Exception:
            pass


def build_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    no_decay_terms = ("bias", "LayerNorm.weight", "layernorm.weight", "norm.weight", "ln_f.weight")
    named_params = list(unwrap_model(model).named_parameters())
    decay = [p for n, p in named_params if p.requires_grad and not any(x in n for x in no_decay_terms)]
    no_decay = [p for n, p in named_params if p.requires_grad and any(x in n for x in no_decay_terms)]

    return AdamW(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.learning_rate,
    )


def build_scheduler(optimizer: torch.optim.Optimizer, total_steps: int, schedule: str) -> LambdaLR:
    schedule = (schedule or "cosine").lower().strip()

    def _lambda(step: int) -> float:
        if total_steps <= 0:
            return 1.0
        progress = min(1.0, max(0.0, step / max(1, total_steps)))
        if schedule == "linear":
            return 1.0 - progress
        if schedule == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return 1.0

    return LambdaLR(optimizer, _lambda)


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


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

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
    scheduler: Optional[LambdaLR],
    scaler: Any,
    cfg: TrainConfig,
    ctx: DistContext,
    epoch: int,
    global_step: int,
    total_steps: int,
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

    for batch_idx, batch in enumerate(loader):
        if sync_stop(SHUTDOWN.stop, ctx):
            break

        batch = move_batch(batch, ctx.device)

        if ctx.is_main and preview_writer is not None:
            try:
                input_ids_cpu = batch["input_ids"].detach().to("cpu")
                texts = []
                for ids in input_ids_cpu:
                    txt = tokenizer.decode(ids.tolist(), skip_special_tokens=False)
                    texts.append(txt)
                preview_text = "\n\n---\n\n".join(texts)
                preview_writer.write(preview_text[:4000], preview_text[:20000])
            except Exception:
                pass

        try:
            with autocast_ctx:
                outputs = model(**batch)
                loss = outputs.loss

            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nicht-finite Loss erkannt: {float(loss.detach().item())}")

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
            raise RuntimeError(
                f"CUDA OOM im Trainingsschritt. Das ist meist kein Leak, sondern ein Batch/Sequenz-Peak. "
                f"Empfehlung: sort_by_length=False, kleinere max_seq_length oder kleinere per_device_train_batch_size. Original: {oom}"
            )

        should_step = (
            ((batch_idx + 1) % cfg.gradient_accumulation_steps == 0)
            or (batch_idx + 1 == len(loader))
        )

        if should_step:
            if scaler is not None and getattr(scaler, "is_enabled", lambda: False)():
                scaler.unscale_(optimizer)

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)

            if scaler is not None and getattr(scaler, "is_enabled", lambda: False)():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

            global_step += 1
            running_updates += 1
            reduced_loss = all_reduce_mean(float(loss.detach().item()), ctx)
            running_loss += reduced_loss

            if ctx.is_main:
                lr = optimizer.param_groups[0]["lr"]
                elapsed = max(1e-6, time.time() - train_start_time)
                steps_done = max(1, global_step)
                steps_left = max(0, int(total_steps) - int(global_step))
                sec_per_step = elapsed / steps_done
                eta = format_eta(sec_per_step * steps_left)
                LOGGER.info("Step %d | Loss: %.6f | LR: %s", global_step, reduced_loss, lr)
                status_writer.write(
                    {
                        "running": True,
                        "step": global_step,
                        "loss": reduced_loss,
                        "learning_rate": lr,
                        "eta": eta,
                        "tokens_per_step": int(cfg.max_seq_length * cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps * max(1, ctx.world_size)),
                        "total_tokens": int(global_step * cfg.max_seq_length * cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps * max(1, ctx.world_size)),
                        "epoch": epoch,
                        "total_steps": int(total_steps),
                    }
                )

            if cfg.max_steps is not None and global_step >= int(cfg.max_steps):
                reached_max_steps = True
                break

    avg_loss = running_loss / max(1, running_updates)
    return avg_loss, global_step, reached_max_steps


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    args = parser.parse_args()

    register_signal_handlers()
    cfg = load_cfg(args.config)
    ctx = init_dist(cfg)

    # Wichtige OOM-Schutzmaßnahme:
    # sort_by_length in strikt aufsteigender Reihenfolge schiebt die größten Samples ans Ende.
    # Dadurch sieht es wie ein Memory-Leak aus, obwohl nur später die längsten Sequenzen kommen.
    # Unter Multi-GPU deaktivieren wir das standardmäßig für stabileren VRAM-Verlauf.
    if ctx.is_distributed and cfg.sort_by_length:
        cfg.sort_by_length = False

    outdir = Path(cfg.output_dir or cfg.save_dir or "./training_outputs/worker_run").expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    log_path = outdir / "training.log"
    status_path = outdir / "status.json"
    setup_logging(log_path, ctx)
    status_writer = JsonStatusWriter(status_path, ctx)
    preview_path = outdir / "livepreview.json"
    preview_writer = JsonPreviewWriter(preview_path, ctx)

    try:
        set_seed(cfg.seed + ctx.rank)

        LOGGER.info("Worker gestartet | rank=%s local_rank=%s world_size=%s device=%s", ctx.rank, ctx.local_rank, ctx.world_size, ctx.device)
        LOGGER.info("Config: %s", json.dumps(cfg.__dict__, ensure_ascii=False))

        model, tokenizer, fp16, bf16 = build_model_and_tokenizer(cfg, ctx)

        examples = build_examples(cfg)
        if not examples:
            raise RuntimeError("Kein Trainingssample gefunden.")

        if ctx.is_main:
            if isinstance(examples[0], tuple):
                preview = (examples[0][0] + examples[0][1])
            else:
                preview = str(examples[0])
            preview_writer.write(preview[:4000], preview[:20000])

        dataset = CausalLMDataset(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=cfg.max_seq_length,
            template_mode=cfg.template_mode,
            sort_by_length=bool(cfg.sort_by_length),
        )
        collator = DataCollator(tokenizer.pad_token_id or tokenizer.eos_token_id or 0, pad_to_multiple_of=8)

        if ctx.is_distributed:
            sampler = DistributedSampler(
                dataset,
                num_replicas=ctx.world_size,
                rank=ctx.rank,
                shuffle=bool(cfg.shuffle),
                drop_last=False,
            )
        else:
            sampler = None

        loader = DataLoader(
            dataset,
            batch_size=cfg.per_device_train_batch_size,
            shuffle=(sampler is None and bool(cfg.shuffle)),
            sampler=sampler,
            num_workers=max(0, int(cfg.dataloader_num_workers)),
            pin_memory=(ctx.device.type == "cuda"),
            collate_fn=collator,
            persistent_workers=False,
        )

        optimizer = build_optimizer(model, cfg)

        updates_per_epoch = max(1, math.ceil(len(loader) / max(1, cfg.gradient_accumulation_steps)))
        total_steps = int(cfg.max_steps) if cfg.max_steps is not None else max(1, int(math.ceil(cfg.num_train_epochs * updates_per_epoch)))
        scheduler = build_scheduler(optimizer, total_steps, cfg.lr_schedule)
        scaler = make_scaler(fp16=fp16, device=ctx.device)

        model = wrap_ddp(model, cfg, ctx)
        barrier(ctx)

        global_step = 0
        last_loss = None
        train_start_time = time.time()

        epochs = max(1, int(math.ceil(cfg.num_train_epochs)))
        for epoch in range(epochs):
            if sampler is not None:
                sampler.set_epoch(epoch)

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
                total_steps=total_steps,
                train_start_time=train_start_time,
                status_writer=status_writer,
                preview_writer=preview_writer,
                tokenizer=tokenizer,
            )
            last_loss = avg_loss

            if ctx.is_main:
                LOGGER.info("Epoche %d abgeschlossen | avg_loss=%.6f", epoch, avg_loss)

            if reached_max_steps or sync_stop(SHUTDOWN.stop, ctx):
                break

        barrier(ctx)

        if ctx.is_main:
            save_target = unwrap_model(model)
            save_target.save_pretrained(outdir)
            tokenizer.save_pretrained(outdir)
            status_writer.write(
                {
                    "running": False,
                    "step": global_step,
                    "loss": last_loss,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "eta": "",
                    "tokens_per_step": int(cfg.max_seq_length * cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps * max(1, ctx.world_size)),
                    "total_tokens": int(global_step * cfg.max_seq_length * cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps * max(1, ctx.world_size)),
                    "done": True,
                }
            )
            LOGGER.info("Training abgeschlossen. Modell gespeichert nach: %s", outdir)

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
