#!/usr/bin/env python3
# matelix_lab_server_web_ddp.py
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

import asyncio
import csv
import gc
import hashlib
import json
import os
import platform
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import psutil
import torch
from fastapi import (
    Body,
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

try:
    from pydantic import BaseModel, Field, ConfigDict  # type: ignore
    PYDANTIC_V2 = True
except Exception:
    from pydantic import BaseModel, Field  # type: ignore
    ConfigDict = None  # type: ignore
    PYDANTIC_V2 = False


class MatelixBaseModel(BaseModel):
    if PYDANTIC_V2:
        model_config = ConfigDict(protected_namespaces=())  # type: ignore


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def configure_tf32(allow_tf32: bool) -> None:
    if not torch.cuda.is_available():
        return

    if allow_tf32:
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


csv.field_size_limit(1024 * 1024 * 128)

BASE_DIR = Path(__file__).resolve().parent
TRAINING_OUT_DIR = BASE_DIR / "training_outputs"
DATASETS_DIR = BASE_DIR / "datasets"
STATIC_DIR = BASE_DIR / "static"
WORKER_PATH = BASE_DIR / "matelix_ddp_worker.py"
OPENAI_COMPAT_API_KEY = "matelix-local-dev-key"

TRAINING_OUT_DIR.mkdir(parents=True, exist_ok=True)
DATASETS_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="MaTeLiX AI Lab (Web DDP)", version="6.1-web-ddp-adaptive-scheduler")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/training_outputs", StaticFiles(directory=str(TRAINING_OUT_DIR)), name="training_outputs")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def model_to_dict(model: Any, **kwargs) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump(**kwargs)
    return model.dict(**kwargs)


def _extract_sequences(out: Any):
    return out.sequences if hasattr(out, "sequences") else out


def get_chat_template(template_mode: str = "chat") -> str:
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


DEFAULT_INDEX_HTML = """<!DOCTYPE html>
<html lang="de">
<head>
  <meta charset="utf-8"/>
  <title>MaTeLiX AI LAB</title>
</head>
<body>
  <h1>MaTeLiX LAB Backend läuft</h1>
  <p>Lege deine Oberfläche unter <code>./static/index.html</code> ab.</p>
</body>
</html>
"""


def ensure_index_html() -> Path:
    p = STATIC_DIR / "index.html"
    if not p.exists():
        p.write_text(DEFAULT_INDEX_HTML, encoding="utf-8")
    return p


class LogStore:
    def __init__(self, max_lines: int = 8000):
        self.max_lines = int(max_lines)
        self._lock = threading.Lock()
        self._base_id = 0
        self._lines: deque[str] = deque()
        self._file_path: Optional[Path] = None

    def set_file(self, path: Optional[Path]) -> None:
        with self._lock:
            self._file_path = path
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch(exist_ok=True)

    def clear(self) -> None:
        with self._lock:
            self._base_id = 0
            self._lines.clear()

    def append(self, msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg.rstrip()}"
        with self._lock:
            if self._file_path is not None:
                try:
                    with open(self._file_path, "a", encoding="utf-8") as f:
                        f.write(line + "\n")
                except Exception:
                    pass
            self._lines.append(line)
            while len(self._lines) > self.max_lines:
                self._lines.popleft()
                self._base_id += 1

    @property
    def last_id(self) -> int:
        with self._lock:
            if not self._lines:
                return self._base_id - 1
            return self._base_id + len(self._lines) - 1

    def tail(self, n: int = 200) -> List[str]:
        with self._lock:
            n = max(1, int(n))
            return list(self._lines)[-n:]

    def since(self, last_seen_id: int):
        with self._lock:
            if not self._lines:
                return [], last_seen_id
            current_last = self._base_id + len(self._lines) - 1
            if last_seen_id < self._base_id - 1:
                return list(self._lines), current_last
            if last_seen_id >= current_last:
                return [], last_seen_id
            start = (last_seen_id - self._base_id) + 1
            data = list(self._lines)[start:]
            return data, current_last


class TrainingState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.running = False
        self.step: int = 0
        self.loss: Optional[float] = None
        self.learning_rate: Optional[float] = None
        self.last_preview: str = ""
        self.last_preview_full: str = ""
        self.tokens_per_step: Optional[int] = None
        self.total_tokens: int = 0
        self.eta: str = ""
        self.save_dir: Optional[str] = None
        self.active_config: Optional[Dict[str, Any]] = None
        self.status_text: str = "idle"
        self.error: Optional[str] = None
        self.world_size: int = 1
        self.exit_code: Optional[int] = None

        self.cuda_memory: Optional[Dict[str, Any]] = None
        self.scheduler_total_steps: Optional[int] = None
        self.csv_total_samples_est: Optional[int] = None
        self.total_samples_real: Optional[int] = None
        self.skipped_samples: Optional[int] = None
        self.producer_progress: Optional[Dict[str, Any]] = None
        self.producer_meta: Optional[Dict[str, Any]] = None
        self.scheduler_state: Optional[Dict[str, Any]] = None
        self.scheduler_source: Optional[str] = None
        self.projected_samples: Optional[int] = None
        self.scheduler_rel_change: Optional[float] = None
        self.warmup_steps: Optional[int] = None

        self.adaptive_scheduler: Optional[bool] = None
        self.adaptive_scheduler_frozen: Optional[bool] = None
        self.adaptive_scheduler_never_increase_lr: Optional[bool] = None
        self.adaptive_scheduler_only_extend_steps: Optional[bool] = None
        self.scheduler_mode: Optional[str] = None
        self.producer_done: Optional[bool] = None

        self.log = LogStore()

    def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "running": self.running,
                "step": self.step,
                "loss": self.loss,
                "learning_rate": self.learning_rate,
                "eta": self.eta,
                "tokens_per_step": self.tokens_per_step,
                "total_tokens": self.total_tokens,
                "save_dir": self.save_dir,
                "config": self.active_config,
                "status_text": self.status_text,
                "error": self.error,
                "world_size": self.world_size,
                "exit_code": self.exit_code,
                "cuda_memory": self.cuda_memory,
                "scheduler_total_steps": self.scheduler_total_steps,
                "csv_total_samples_est": self.csv_total_samples_est,
                "total_samples_real": self.total_samples_real,
                "skipped_samples": self.skipped_samples,
                "producer_progress": self.producer_progress,
                "producer_meta": self.producer_meta,
                "scheduler_state": self.scheduler_state,
                "scheduler_source": self.scheduler_source,
                "projected_samples": self.projected_samples,
                "scheduler_rel_change": self.scheduler_rel_change,
                "warmup_steps": self.warmup_steps,
                "adaptive_scheduler": self.adaptive_scheduler,
                "adaptive_scheduler_frozen": self.adaptive_scheduler_frozen,
                "adaptive_scheduler_never_increase_lr": self.adaptive_scheduler_never_increase_lr,
                "adaptive_scheduler_only_extend_steps": self.adaptive_scheduler_only_extend_steps,
                "scheduler_mode": self.scheduler_mode,
                "producer_done": self.producer_done,
            }


TRAIN_STATE = TrainingState()


class WebTrainConfig(MatelixBaseModel):
    model_dir: str = Field(...)
    csv_path: str = Field(...)
    dataset_source: str = "local_csv"
    hf_dataset_id: Optional[str] = None
    hf_dataset_split: str = "train"

    save_dir: Optional[str] = None
    template_mode: str = "chat"
    column_name: str = "text"

    learning_rate: float = 2e-5
    lr_schedule: str = "cosine"
    lr_decay_factor: float = 1.0
    warmup_steps: int = 0
    warmup_ratio: float = 0.03
    min_lr_ratio: float = 0.05
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
    shuffle: bool = False
    sort_by_length: bool = True
    max_history_turns: Optional[int] = None

    device: str = "auto"
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
    precision_mode: str = "auto"
    gradient_checkpointing: bool = False
    dataloader_num_workers: int = 0
    merge_lora_on_save: bool = True
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01

    use_ngrams: bool = False
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

    # Neu: N-Gram Konsistenz / Code
    ngram_conflict_overlap_ratio: float = 0.8
    ngram_prefer_longest_match: bool = True
    ngram_use_code_lines: bool = True
    ngram_code_line_min_chars: int = 12
    ngram_code_line_min_count: int = 2
    ngram_code_line_top_k: int = 400
    ngram_code_line_score_boost: float = 1.35
    ngram_code_pattern_boost: bool = True
    ngram_code_pattern_extra_boost: float = 1.2

    # Neu: Skip-/Oversize-Fokus
    ngram_focus_oversize_samples: bool = True
    ngram_oversize_sample_boost: float = 4.0
    ngram_near_limit_threshold: float = 0.90
    ngram_near_limit_boost: float = 1.5
    ngram_eval_max_seq_length: int = 1024
    ngram_track_saved_samples: bool = True
    ngram_saved_sample_boost: float = 8.0

    ddp_enabled: Optional[bool] = None
    nproc_per_node: Optional[int] = None
    nnodes: int = 1
    node_rank: int = 0
    master_addr: str = "127.0.0.1"
    master_port: int = 29500
    seed: int = 42
    deterministic: bool = False

    find_unused_parameters: Optional[bool] = None
    ddp_find_unused_parameters: Optional[bool] = None

    ddp_broadcast_buffers: bool = False
    ddp_static_graph: bool = False
    val_split: float = 0.05
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 0.0
    log_every_steps: int = 10
    use_tensorboard: bool = True
    save_every_epoch: bool = True
    keep_last_k_checkpoints: int = 3
    resume: Optional[str] = None

    allow_tf32: bool = True
    distributed_debug: bool = False
    nccl_blocking_wait: bool = False
    prefetch_factor: int = 2
    persistent_workers: bool = True

    use_dataset_cache: bool = True
    rebuild_dataset_cache: bool = False
    tokenized_shard_size: int = 5000

    log_cuda_memory: bool = True
    cuda_memory_log_interval_steps: int = 25
    cuda_empty_cache_interval_steps: int = 0


def _wrap_with_start_stop_token(text: str) -> str:
    payload = (text or "").strip()
    if not payload:
        return ""
    return f"<s>{payload}</s>"


def _resolve_hf_dataset_to_csv(cfg: WebTrainConfig) -> str:
    if (cfg.dataset_source or "local_csv").strip().lower() != "huggingface":
        return cfg.csv_path

    dataset_id = (cfg.hf_dataset_id or "").strip()
    if not dataset_id:
        raise HTTPException(status_code=400, detail="hf_dataset_id fehlt für HuggingFace-Datasets.")

    split_name = (cfg.hf_dataset_split or "train").strip() or "train"
    requested_column = (cfg.column_name or "text").strip() or "text"

    try:
        from datasets import load_dataset  # type: ignore
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"datasets-Paket fehlt: {exc}")

    try:
        ds = load_dataset(dataset_id, split=split_name)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"HuggingFace Dataset konnte nicht geladen werden: {exc}")

    available_cols = {str(c).strip(): c for c in ds.column_names}
    lowered_to_real = {str(c).strip().lower(): str(c).strip() for c in ds.column_names}

    chosen_column: Optional[str] = None
    conversion_mode = "direct"

    if requested_column in available_cols:
        chosen_column = requested_column
    else:
        candidate_order = ["text", "output", "answer", "response", "completion", "prompt"]
        for key_name in candidate_order:
            real = lowered_to_real.get(key_name)
            if real is not None:
                chosen_column = real
                break

        has_input = lowered_to_real.get("input") is not None
        has_output = lowered_to_real.get("output") is not None
        if has_input and has_output:
            conversion_mode = "input_output"
            chosen_column = "text"

    if chosen_column is None:
        conversion_mode = "row_with_headers"
        chosen_column = "text"

    key = hashlib.sha256(f"{dataset_id}:{split_name}:{requested_column}:{conversion_mode}:{chosen_column}".encode("utf-8")).hexdigest()[:12]
    safe_name = dataset_id.replace("/", "__").replace(":", "_")
    out_path = DATASETS_DIR / f"hf_{safe_name}_{split_name}_{chosen_column}_{key}.csv"

    if not out_path.exists():
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            # Explizit quoten, damit mehrzeilige/kommahaltige Texte aus HF-Datasets
            # robust serialisiert werden und kein `_csv.Error: need to escape` auftritt.
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerow([chosen_column])
            if conversion_mode == "input_output":
                input_col = lowered_to_real["input"]
                output_col = lowered_to_real["output"]
                for row in ds:
                    in_val = "" if row.get(input_col) is None else str(row.get(input_col))
                    out_val = "" if row.get(output_col) is None else str(row.get(output_col))
                    merged = f"Input:\n{in_val.strip()}\n\nOutput:\n{out_val.strip()}".strip()
                    writer.writerow([_wrap_with_start_stop_token(merged)])
            elif conversion_mode == "row_with_headers":
                for row in ds:
                    parts: List[str] = []
                    for col in ds.column_names:
                        value = row.get(col)
                        if isinstance(value, (dict, list)):
                            value_text = json.dumps(value, ensure_ascii=False)
                        elif value is None:
                            value_text = ""
                        else:
                            value_text = str(value)
                        parts.append(f"{col}: {value_text}")
                    merged = "\n".join(parts).strip()
                    writer.writerow([_wrap_with_start_stop_token(merged)])
            else:
                for value in ds[chosen_column]:
                    writer.writerow([_wrap_with_start_stop_token("" if value is None else str(value))])

    cfg.column_name = chosen_column

    return f"./datasets/{out_path.name}"


def get_new_output_dir(model_name: str, base_dir: Optional[Path] = None) -> Path:
    now = time.strftime("%Y-%m-%d_%H-%M-%S")
    safe_name = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in model_name) or "model"
    base = base_dir or TRAINING_OUT_DIR
    out = base / f"{safe_name}_{now}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def read_json_if_exists(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def normalize_device(requested: str) -> str:
    r = (requested or "auto").strip().lower()
    if r == "cuda" and torch.cuda.is_available():
        return "cuda"
    if r == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    if r == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def query_nvidia_smi() -> List[Dict[str, Any]]:
    if not torch.cuda.is_available():
        return []
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.total,memory.used,name",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
        )
        rows = out.decode("utf-8", errors="ignore").strip().splitlines()
        result = []
        for idx, line in enumerate(rows):
            util, mem_total, mem_used, name = [s.strip() for s in line.split(",")]
            result.append(
                {
                    "id": idx,
                    "name": name,
                    "util": int(util),
                    "mem_total": int(mem_total),
                    "mem_used": int(mem_used),
                }
            )
        return result
    except Exception:
        return []


def get_hardware_info() -> Dict[str, Any]:
    cuda = torch.cuda.is_available()
    mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    gpus = []
    if cuda:
        for i in range(torch.cuda.device_count()):
            gpus.append(torch.cuda.get_device_name(i))
    return {
        "cuda": cuda,
        "mps": mps,
        "num_cuda": torch.cuda.device_count() if cuda else 0,
        "gpus": gpus,
        "num_cpus": os.cpu_count() or 1,
    }


def get_system_status() -> Dict[str, Any]:
    mem = psutil.virtual_memory()
    cpu = psutil.cpu_percent(interval=None)
    gpu_infos = query_nvidia_smi()
    primary = gpu_infos[0] if gpu_infos else None
    return {
        "ram_used": mem.used // (1024 * 1024),
        "ram_total": mem.total // (1024 * 1024),
        "ram_percent": mem.percent,
        "cpu_percent": cpu,
        "gpu_name": primary["name"] if primary else "",
        "gpu_util": primary["util"] if primary else None,
        "gpu_mem": primary["mem_total"] if primary else None,
        "gpu_mem_used": primary["mem_used"] if primary else None,
        "platform": platform.platform(),
        "gpus": gpu_infos,
        "num_cuda": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }


class DDPTrainingManager:
    def __init__(self, state: TrainingState) -> None:
        self.state = state
        self.proc: Optional[subprocess.Popen] = None
        self.runtime_state_path: Optional[Path] = None
        self.status_state_path: Optional[Path] = None
        self.livepreview_path: Optional[Path] = None
        self.run_dir: Optional[Path] = None
        self.stdout_thread: Optional[threading.Thread] = None
        self.wait_thread: Optional[threading.Thread] = None
        self.stop_requested = False
        self.lock = threading.Lock()

    def _merge_state_payload(self, payload: Dict[str, Any], source: str = "unknown") -> None:
        with self.state.lock:
            self.state.step = int(payload.get("step") or self.state.step or 0)

            if payload.get("loss") is not None:
                self.state.loss = payload.get("loss")
            if payload.get("learning_rate") is not None:
                self.state.learning_rate = payload.get("learning_rate")

            self.state.eta = payload.get("eta") or self.state.eta or ""

            if payload.get("tokens_per_step") is not None:
                self.state.tokens_per_step = payload.get("tokens_per_step")
            self.state.total_tokens = int(payload.get("total_tokens") or self.state.total_tokens or 0)

            self.state.last_preview = (payload.get("preview") or self.state.last_preview or "")[:4000]
            self.state.last_preview_full = (
                payload.get("preview_full") or self.state.last_preview_full or self.state.last_preview or ""
            )[:20000]

            self.state.save_dir = payload.get("save_dir") or self.state.save_dir
            self.state.status_text = payload.get("status") or self.state.status_text
            self.state.error = payload.get("error") or self.state.error
            self.state.world_size = int(payload.get("world_size") or self.state.world_size or 1)

            if payload.get("cuda_memory") is not None:
                self.state.cuda_memory = payload.get("cuda_memory")
            if payload.get("scheduler_total_steps") is not None:
                self.state.scheduler_total_steps = int(payload.get("scheduler_total_steps"))
            if payload.get("csv_total_samples_est") is not None:
                self.state.csv_total_samples_est = int(payload.get("csv_total_samples_est"))
            if payload.get("total_samples_real") is not None:
                self.state.total_samples_real = int(payload.get("total_samples_real"))
            if payload.get("skipped_samples") is not None:
                self.state.skipped_samples = int(payload.get("skipped_samples"))
            if payload.get("producer_progress") is not None:
                self.state.producer_progress = payload.get("producer_progress")
            if payload.get("producer_meta") is not None:
                self.state.producer_meta = payload.get("producer_meta")
            if payload.get("scheduler_state") is not None:
                self.state.scheduler_state = payload.get("scheduler_state")
            if payload.get("scheduler_source") is not None:
                self.state.scheduler_source = payload.get("scheduler_source")
            if payload.get("projected_samples") is not None:
                self.state.projected_samples = int(payload.get("projected_samples"))
            if payload.get("scheduler_rel_change") is not None:
                self.state.scheduler_rel_change = float(payload.get("scheduler_rel_change"))
            if payload.get("warmup_steps") is not None:
                self.state.warmup_steps = int(payload.get("warmup_steps"))
            if payload.get("adaptive_scheduler") is not None:
                self.state.adaptive_scheduler = bool(payload.get("adaptive_scheduler"))
            if payload.get("adaptive_scheduler_frozen") is not None:
                self.state.adaptive_scheduler_frozen = bool(payload.get("adaptive_scheduler_frozen"))
            if payload.get("adaptive_scheduler_never_increase_lr") is not None:
                self.state.adaptive_scheduler_never_increase_lr = bool(payload.get("adaptive_scheduler_never_increase_lr"))
            if payload.get("adaptive_scheduler_only_extend_steps") is not None:
                self.state.adaptive_scheduler_only_extend_steps = bool(payload.get("adaptive_scheduler_only_extend_steps"))
            if payload.get("scheduler_mode") is not None:
                self.state.scheduler_mode = str(payload.get("scheduler_mode"))
            if payload.get("producer_done") is not None:
                self.state.producer_done = bool(payload.get("producer_done"))

            if self.proc is not None:
                self.state.running = self.proc.poll() is None and bool(payload.get("running", self.state.running))

            if self.run_dir is None and self.state.save_dir:
                try:
                    self.run_dir = Path(self.state.save_dir)
                except Exception:
                    pass

    def _refresh_from_runtime_file(self) -> None:
        runtime_payload = read_json_if_exists(self.runtime_state_path)
        status_payload = read_json_if_exists(self.status_state_path)
        preview_payload = read_json_if_exists(self.livepreview_path)

        if runtime_payload:
            self._merge_state_payload(runtime_payload, source="runtime_state.json")
        if status_payload:
            self._merge_state_payload(status_payload, source="status.json")
        if preview_payload:
            self._merge_state_payload(preview_payload, source="livepreview.json")

        with self.state.lock:
            if self.proc is not None and self.proc.poll() is not None:
                self.state.running = False
            elif self.proc is not None and self.proc.poll() is None:
                self.state.running = True
                if self.state.status_text in {"idle", "finished", "error"}:
                    self.state.status_text = "running"

    def _resolve_find_unused(self, cfg: WebTrainConfig, ddp_enabled: bool, nproc: int, device: str) -> bool:
        explicit = None
        if cfg.ddp_find_unused_parameters is not None:
            explicit = bool(cfg.ddp_find_unused_parameters)
        elif cfg.find_unused_parameters is not None:
            explicit = bool(cfg.find_unused_parameters)

        if explicit is not None:
            return explicit

        if ddp_enabled and nproc > 1 and device == "cuda":
            return True

        return False

    def start(self, cfg: WebTrainConfig) -> Dict[str, Any]:
        if not WORKER_PATH.exists():
            return {"running": False, "error": f"Worker fehlt: {WORKER_PATH}"}

        with self.state.lock:
            if self.proc is not None and self.proc.poll() is None:
                return {"running": False, "error": "Training läuft bereits."}

        model_name = Path(cfg.model_dir.rstrip("/")).name or "model"
        base_out = Path(cfg.save_dir).expanduser().resolve() if (cfg.save_dir and str(cfg.save_dir).strip()) else TRAINING_OUT_DIR
        run_dir = get_new_output_dir(model_name, base_out)
        runtime_state_path = run_dir / "runtime_state.json"
        status_state_path = run_dir / "status.json"
        livepreview_path = run_dir / "livepreview.json"

        device = normalize_device(cfg.device)
        auto_ddp = (device == "cuda" and torch.cuda.device_count() > 1)
        ddp_enabled = auto_ddp if cfg.ddp_enabled is None else bool(cfg.ddp_enabled)
        if device != "cuda":
            ddp_enabled = False

        nproc = int(cfg.nproc_per_node or (torch.cuda.device_count() if ddp_enabled else 1))
        nproc = max(1, nproc)
        if device == "cuda" and torch.cuda.device_count() > 0:
            nproc = min(nproc, torch.cuda.device_count())
        if not ddp_enabled:
            nproc = 1

        worker_cfg = model_to_dict(
            cfg,
            exclude={
                "save_dir",
                "ddp_enabled",
                "nproc_per_node",
                "nnodes",
                "node_rank",
                "master_addr",
                "master_port",
                "find_unused_parameters",
            },
        )

        worker_cfg["device"] = device
        if bool(worker_cfg.get("train_from_scratch", False)):
            worker_cfg["train_mode"] = "full"
            worker_cfg["merge_lora_on_save"] = False
        if (cfg.train_mode or "full").strip().lower() == "lora" and float(worker_cfg.get("learning_rate", 0.0)) <= 2e-5:
            worker_cfg["learning_rate"] = 2e-4
        worker_cfg["output_dir"] = str(run_dir)
        worker_cfg["save_dir"] = str(run_dir)
        worker_cfg["ddp_find_unused_parameters"] = self._resolve_find_unused(cfg, ddp_enabled, nproc, device)
        worker_cfg["force_template"] = True

        if bool(worker_cfg.get("gradient_checkpointing")) and worker_cfg["ddp_find_unused_parameters"]:
            worker_cfg["ddp_static_graph"] = False

        cfg_path = run_dir / "worker_config.json"
        atomic_write_json(cfg_path, worker_cfg)
        atomic_write_json(
            run_dir / "train_config.json",
            {
                **model_to_dict(cfg),
                "device": device,
                "output_dir": str(run_dir),
                "effective_world_size": nproc,
                "effective_ddp_enabled": bool(ddp_enabled and nproc > 1),
                "effective_ddp_find_unused_parameters": worker_cfg["ddp_find_unused_parameters"],
                "effective_template_mode": worker_cfg.get("template_mode", "chat"),
                "effective_force_template": True,
                "effective_max_history_turns": worker_cfg.get("max_history_turns"),
            },
        )

        if ddp_enabled and nproc > 1:
            cmd = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--nproc_per_node",
                str(nproc),
                "--nnodes",
                str(max(1, int(cfg.nnodes))),
                "--node_rank",
                str(max(0, int(cfg.node_rank))),
                "--master_addr",
                str(cfg.master_addr),
                "--master_port",
                str(int(cfg.master_port)),
                str(WORKER_PATH),
                "--config",
                str(cfg_path),
            ]
        else:
            cmd = [sys.executable, str(WORKER_PATH), "--config", str(cfg_path)]

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["TOKENIZERS_PARALLELISM"] = "false"
        env["MATELIX_ALLOW_TF32"] = "1" if bool(cfg.allow_tf32) else "0"
        env["MATELIX_DETERMINISTIC"] = "1" if bool(cfg.deterministic) else "0"
        env["MATELIX_NCCL_BLOCKING_WAIT"] = "1" if bool(cfg.nccl_blocking_wait) else "0"
        if bool(cfg.distributed_debug):
            env["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
        else:
            env.pop("TORCH_DISTRIBUTED_DEBUG", None)

        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(BASE_DIR),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                text=True,
                bufsize=1,
                universal_newlines=True,
                start_new_session=(os.name != "nt"),
            )
        except Exception as exc:
            return {"running": False, "error": f"Subprocess-Start fehlgeschlagen: {exc}"}

        with self.lock:
            self.proc = proc
            self.runtime_state_path = runtime_state_path
            self.status_state_path = status_state_path
            self.livepreview_path = livepreview_path
            self.run_dir = run_dir
            self.stop_requested = False

        with self.state.lock:
            self.state.running = True
            self.state.step = 0
            self.state.loss = None
            self.state.learning_rate = None
            self.state.eta = ""
            self.state.tokens_per_step = None
            self.state.total_tokens = 0
            self.state.last_preview = ""
            self.state.last_preview_full = ""
            self.state.save_dir = str(run_dir)
            self.state.active_config = worker_cfg
            self.state.status_text = "starting"
            self.state.error = None
            self.state.world_size = nproc
            self.state.exit_code = None

            self.state.cuda_memory = None
            self.state.scheduler_total_steps = None
            self.state.csv_total_samples_est = None
            self.state.total_samples_real = None
            self.state.skipped_samples = None
            self.state.producer_progress = None
            self.state.producer_meta = None
            self.state.scheduler_state = None
            self.state.scheduler_source = None
            self.state.projected_samples = None
            self.state.scheduler_rel_change = None
            self.state.warmup_steps = None
            self.state.adaptive_scheduler = None
            self.state.adaptive_scheduler_frozen = None
            self.state.adaptive_scheduler_never_increase_lr = None
            self.state.adaptive_scheduler_only_extend_steps = None
            self.state.scheduler_mode = None
            self.state.producer_done = None

            self.state.log.clear()
            self.state.log.set_file(run_dir / "training.log")
            self.state.log.append("Training gestartet.")
            self.state.log.append("Command: " + " ".join(cmd))
            self.state.log.append(
                f"DDP effective: enabled={bool(ddp_enabled and nproc > 1)} "
                f"world_size={nproc} "
                f"find_unused_parameters={worker_cfg['ddp_find_unused_parameters']} "
                f"deterministic={worker_cfg.get('deterministic', False)} "
                f"allow_tf32={worker_cfg.get('allow_tf32', True)} "
                f"dataloader_num_workers={worker_cfg.get('dataloader_num_workers', 4)} "
                f"max_history_turns={worker_cfg.get('max_history_turns')} "
                f"log_cuda_memory={worker_cfg.get('log_cuda_memory', True)} "
                f"cuda_memory_log_interval_steps={worker_cfg.get('cuda_memory_log_interval_steps', 25)} "
                f"cuda_empty_cache_interval_steps={worker_cfg.get('cuda_empty_cache_interval_steps', 0)} "
                f"adaptive_scheduler={worker_cfg.get('adaptive_scheduler', True)} "
                f"adaptive_scheduler_freeze_on_producer_done={worker_cfg.get('adaptive_scheduler_freeze_on_producer_done', True)} "
                f"adaptive_scheduler_never_increase_lr={worker_cfg.get('adaptive_scheduler_never_increase_lr', True)} "
                f"adaptive_scheduler_only_extend_steps={worker_cfg.get('adaptive_scheduler_only_extend_steps', True)}"
            )

        self.stdout_thread = threading.Thread(target=self._stdout_pump, args=(proc,), daemon=True)
        self.wait_thread = threading.Thread(target=self._wait_pump, args=(proc,), daemon=True)
        self.stdout_thread.start()
        self.wait_thread.start()

        return {
            "running": True,
            "msg": f"Training gestartet ({'DDP' if ddp_enabled and nproc > 1 else 'Single'} | world_size={nproc})",
            "save_dir": str(run_dir),
            "world_size": nproc,
            "ddp_enabled": bool(ddp_enabled and nproc > 1),
            "ddp_find_unused_parameters": worker_cfg["ddp_find_unused_parameters"],
            "deterministic": worker_cfg.get("deterministic", False),
            "allow_tf32": worker_cfg.get("allow_tf32", True),
        }

    def _stdout_pump(self, proc: subprocess.Popen) -> None:
        try:
            if proc.stdout is None:
                return
            for line in proc.stdout:
                if line is None:
                    break
                stripped = line.rstrip("\n")
                if stripped:
                    self.state.log.append(stripped)
                self._refresh_from_runtime_file()
        except Exception as exc:
            self.state.log.append(f"[LOG-PUMP ERROR] {exc}")

    def _wait_pump(self, proc: subprocess.Popen) -> None:
        try:
            code = proc.wait()
        except Exception as exc:
            self.state.log.append(f"[WAIT ERROR] {exc}")
            code = -1

        self._refresh_from_runtime_file()

        with self.state.lock:
            self.state.running = False
            self.state.exit_code = int(code)
            if self.stop_requested and self.state.status_text not in {"error", "finished"}:
                self.state.status_text = "stopped"
            elif code == 0 and self.state.status_text not in {"error", "stopped"}:
                self.state.status_text = "finished"
            elif code != 0 and not self.state.error:
                self.state.error = f"Worker beendet mit Exit-Code {code}"
                self.state.status_text = "error"

        self.state.log.append(f"Training beendet (exit_code={code}).")

    def stop(self) -> Dict[str, Any]:
        with self.lock:
            proc = self.proc
            self.stop_requested = True

        if proc is None or proc.poll() is not None:
            with self.state.lock:
                self.state.running = False
                self.state.status_text = "idle"
            return {"msg": "Kein laufendes Training gefunden."}

        with self.state.lock:
            self.state.eta = "stopping"
            self.state.status_text = "stopping"

        self.state.log.append("Stop-Signal gesetzt. Beende Trainingsprozess …")

        try:
            # Graceful Stop:
            # Nur den torchrun/Launcher-Prozess terminieren. torchrun leitet SIGTERM
            # an die Worker weiter und kann die DDP-Prozessgruppe sauber abbauen.
            # Ein sofortiges os.killpg(..., SIGTERM) trifft Launcher und Worker
            # gleichzeitig und fuehrt haeufig zu TCPStore/Broken-Pipe-Noise und -9.
            proc.terminate()
        except Exception as exc:
            self.state.log.append(f"[STOP WARN] {exc}")

        def _kill_later(p: subprocess.Popen):
            try:
                # Lange Sequenzen/DDP koennen noch einen laufenden CUDA-Step beenden.
                # Deshalb nicht nach 20s hart killen.
                p.wait(timeout=120)
            except Exception:
                try:
                    self.state.log.append("[STOP WARN] Graceful Stop Timeout, sende SIGKILL an Prozessgruppe.")
                    if os.name != "nt":
                        os.killpg(p.pid, signal.SIGKILL)
                    else:
                        p.kill()
                except Exception:
                    pass

        threading.Thread(target=_kill_later, args=(proc,), daemon=True).start()
        return {"msg": "Stop-Signal gesendet."}

    def status(self) -> Dict[str, Any]:
        self._refresh_from_runtime_file()
        if self.proc is not None and self.proc.poll() is not None:
            with self.state.lock:
                self.state.running = False
        return self.state.snapshot()

    def livepreview(self) -> Dict[str, Any]:
        self._refresh_from_runtime_file()
        with self.state.lock:
            return {"preview": self.state.last_preview, "preview_full": self.state.last_preview_full}


TRAIN_MANAGER = DDPTrainingManager(TRAIN_STATE)


def prepare_tokenizer_for_matelix(tokenizer, force_template: bool = False, template_mode: str = "chat") -> bool:
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


def unload_torch_objects(*objs: Any) -> None:
    for obj in objs:
        try:
            if hasattr(obj, "to"):
                try:
                    obj.to(torch.device("cpu"))
                except Exception:
                    pass
        except Exception:
            pass
        try:
            del obj
        except Exception:
            pass
    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception:
        pass
    try:
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


class InferenceSession:
    def __init__(self):
        self.lock = threading.Lock()
        self.generate_lock = threading.Lock()
        self.loaded_dir: Optional[str] = None
        self.device: Optional[torch.device] = None
        self.tokenizer = None
        self.model = None
        self.is_generating: bool = False
        self.current_source: Optional[str] = None
        self.last_started_at: Optional[float] = None


INFER = InferenceSession()


class ChatRequest(MatelixBaseModel):
    model_dir: Optional[str] = None
    device: Optional[str] = None
    system: Optional[str] = None
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    max_new_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repetition_penalty: float = 1.1
    do_sample: bool = True


def _preferred_device_name(req: Optional[str] = None) -> str:
    return normalize_device(req or "auto")


def _get_device(name: str) -> torch.device:
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if name == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_latest_model_dir() -> Optional[str]:
    candidates: List[Path] = []

    status_dir = TRAIN_MANAGER.status().get("save_dir")
    if status_dir:
        try:
            p = Path(status_dir).expanduser().resolve()
            if p.exists() and p.is_dir():
                candidates.append(p)
        except Exception:
            pass

    if TRAINING_OUT_DIR.exists():
        for p in TRAINING_OUT_DIR.iterdir():
            try:
                if p.is_dir():
                    candidates.append(p.resolve())
            except Exception:
                continue

    valid_candidates: List[Path] = []
    for p in candidates:
        try:
            merged = p / "merged"
            if merged.exists() and (merged / "config.json").exists():
                valid_candidates.append(merged)
            elif (p / "config.json").exists():
                valid_candidates.append(p)
        except Exception:
            continue

    if not valid_candidates:
        return None

    valid_candidates = sorted(valid_candidates, key=lambda p: p.stat().st_mtime, reverse=True)
    return str(valid_candidates[0])


def unload_inference() -> Dict[str, Any]:
    with INFER.lock:
        if INFER.is_generating:
            return {"error": "Modell generiert gerade und kann aktuell nicht entladen werden."}

        unload_torch_objects(INFER.model, INFER.tokenizer)
        INFER.loaded_dir = None
        INFER.device = None
        INFER.tokenizer = None
        INFER.model = None
        INFER.current_source = None
        INFER.last_started_at = None
    return {"msg": "Inferenz-Modell entladen."}


def load_inference_model(model_dir: str, device_name: str = "auto") -> Dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    try:
        from peft import PeftConfig, PeftModel
    except Exception:
        PeftConfig = None
        PeftModel = None

    dev_name = _preferred_device_name(device_name)
    dev = _get_device(dev_name)
    configure_tf32(dev.type == "cuda")

    with INFER.lock:
        if INFER.is_generating:
            return {"error": "Modell generiert gerade. Laden bitte nach Abschluss erneut ausführen."}

        if INFER.loaded_dir and (INFER.loaded_dir != model_dir or (INFER.device and INFER.device.type != dev.type)):
            unload_torch_objects(INFER.model, INFER.tokenizer)
            INFER.loaded_dir = None
            INFER.device = None
            INFER.tokenizer = None
            INFER.model = None

        effective_model_dir = model_dir
        if (Path(model_dir) / "merged" / "config.json").exists():
            effective_model_dir = str(Path(model_dir) / "merged")

        template_mode = "chat"

        template_info_path = Path(model_dir) / "template_info.json"
        if not template_info_path.exists():
            template_info_path = Path(effective_model_dir) / "template_info.json"

        if template_info_path.exists():
            try:
                template_info = json.loads(template_info_path.read_text(encoding="utf-8"))
                template_mode = (template_info.get("template_mode") or "chat").strip().lower()
            except Exception:
                template_mode = "chat"

        tok = AutoTokenizer.from_pretrained(effective_model_dir, trust_remote_code=False)
        need_resize = prepare_tokenizer_for_matelix(
            tok,
            force_template=True,
            template_mode=template_mode,
        )

        dtype = torch.float16 if dev.type == "cuda" else None

        is_adapter = (Path(model_dir) / "adapter_config.json").exists()
        if is_adapter and PeftConfig is not None and PeftModel is not None:
            peft_cfg = PeftConfig.from_pretrained(model_dir)
            base_model_dir = peft_cfg.base_model_name_or_path
            mdl = AutoModelForCausalLM.from_pretrained(
                base_model_dir,
                trust_remote_code=False,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                attn_implementation="sdpa",
            )
            mdl = PeftModel.from_pretrained(mdl, model_dir)
        else:
            mdl = AutoModelForCausalLM.from_pretrained(
                effective_model_dir,
                trust_remote_code=False,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                attn_implementation="sdpa",
            )
        if need_resize and hasattr(mdl, "resize_token_embeddings"):
            mdl.resize_token_embeddings(len(tok))
        if dev.type == "mps":
            mdl = mdl.to(torch.float32)
        if hasattr(mdl, "config"):
            mdl.config.use_cache = True
        mdl.to(dev)
        mdl.eval()

        INFER.loaded_dir = model_dir
        INFER.device = dev
        INFER.tokenizer = tok
        INFER.model = mdl

    return {"msg": f"Modell geladen: {model_dir} auf {dev.type.upper()}"}


def ensure_model_loaded(model_dir: Optional[str], device_name: Optional[str]):
    mdl_dir = model_dir or INFER.loaded_dir or get_latest_model_dir()
    if not mdl_dir:
        raise RuntimeError("Kein Inferenz-Modell gefunden. Trainiere zuerst ein Modell oder gib model_dir an.")

    dev_name = _preferred_device_name(device_name or (INFER.device.type if INFER.device else "auto"))
    target_dev = _get_device(dev_name)

    current_loaded = INFER.loaded_dir
    current_device_type = INFER.device.type if INFER.device else None

    if current_loaded != mdl_dir or current_device_type != target_dev.type:
        res = load_inference_model(mdl_dir, dev_name)
        if res.get("error"):
            raise RuntimeError(res["error"])

    with INFER.lock:
        return mdl_dir, INFER.tokenizer, INFER.model, INFER.device


def prepare_inputs(tokenizer, messages: List[Dict[str, Any]], system: Optional[str], device: torch.device):
    msgs = list(messages or [])
    if msgs and isinstance(msgs[-1], dict):
        if msgs[-1].get("role") == "assistant" and not (msgs[-1].get("content") or "").strip():
            msgs = msgs[:-1]

    if system and system.strip():
        msgs = [{"role": "system", "content": system.strip()}] + msgs

    if hasattr(tokenizer, "apply_chat_template"):
        enc = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    else:
        eos = tokenizer.eos_token or "</s>"
        parts = []
        for m in msgs:
            role = (m.get("role") or "").strip().lower()
            content = (m.get("content") or "").strip()
            if not content:
                continue
            if role == "system":
                parts.append("<|System|>" + content + eos)
            elif role == "user":
                parts.append("<|Benutzer|>" + content + eos)
            elif role == "assistant":
                parts.append("<|Assistentin|>" + content + eos)
        parts.append("<|Assistentin|>")
        enc = tokenizer("".join(parts), return_tensors="pt", add_special_tokens=False)

    pad_id = tokenizer.pad_token_id
    if pad_id is None or pad_id == tokenizer.eos_token_id:
        pad_id = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else (tokenizer.eos_token_id or 0)
    eos_id = tokenizer.eos_token_id or tokenizer.convert_tokens_to_ids("</s>") or pad_id

    if isinstance(enc, dict) or hasattr(enc, "input_ids"):
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"] if "attention_mask" in enc else (input_ids != pad_id).long()
    else:
        input_ids = enc
        attention_mask = (input_ids != pad_id).long()

    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
        attention_mask = attention_mask.unsqueeze(0)

    return input_ids.to(device), attention_mask.to(device), int(pad_id), int(eos_id)


def _prepare_inputs(tokenizer, messages: List[Dict[str, Any]], system: Optional[str], device: torch.device):
    return prepare_inputs(tokenizer, messages, system, device)


def sanitize_sampling_args(
    *,
    max_new_tokens: int,
    temperature: Optional[float],
    top_p: Optional[float],
    top_k: Optional[int],
    repetition_penalty: Optional[float],
    do_sample: Optional[bool],
    pad_id: int,
    eos_id: int,
):
    t = 0.0 if temperature is None else float(temperature)
    p = 1.0 if top_p is None else float(top_p)
    k = 40 if top_k is None else int(top_k)
    rp = 1.1 if repetition_penalty is None else float(repetition_penalty)

    if t <= 0.0:
        safe = {
            "max_new_tokens": max(1, int(max_new_tokens)),
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 0,
            "repetition_penalty": max(0.01, rp),
            "do_sample": False,
            "pad_token_id": pad_id,
            "eos_token_id": eos_id,
        }
    else:
        p = 1.0 if not (0.0 < p <= 1.0) else p
        k = 0 if k < 0 else k
        safe = {
            "max_new_tokens": max(1, int(max_new_tokens)),
            "temperature": max(1e-5, t),
            "top_p": p,
            "top_k": k,
            "repetition_penalty": max(0.01, rp),
            "do_sample": True if do_sample is None else bool(do_sample),
            "pad_token_id": pad_id,
            "eos_token_id": eos_id,
        }

    from transformers import LogitsProcessorList

    class SafeLogitsProcessor:
        def __call__(self, input_ids, scores):
            if not torch.isfinite(scores).all():
                scores = scores.clone()
                scores[~torch.isfinite(scores)] = -1e9
            return torch.clamp(scores, min=-1e9, max=1e9)

    return safe, LogitsProcessorList([SafeLogitsProcessor()])


def begin_generation(source: str) -> None:
    with INFER.lock:
        INFER.is_generating = True
        INFER.current_source = source
        INFER.last_started_at = time.time()


def end_generation() -> None:
    with INFER.lock:
        INFER.is_generating = False
        INFER.current_source = None


@app.get("/hardware")
def api_hardware():
    return get_hardware_info()


@app.get("/sysstatus")
def api_sysstatus():
    return get_system_status()


@app.get("/available_models")
def api_available_models():
    if not TRAINING_OUT_DIR.exists():
        return []
    return sorted([p.name for p in TRAINING_OUT_DIR.iterdir() if p.is_dir()])


@app.get("/available_datasets")
def api_available_datasets():
    if not DATASETS_DIR.exists():
        return []
    return sorted([p.name for p in DATASETS_DIR.iterdir() if p.is_file() and p.suffix.lower() == ".csv"])


@app.get("/trainings")
def api_trainings():
    if not TRAINING_OUT_DIR.exists():
        return []
    trainings = []
    for d in sorted(TRAINING_OUT_DIR.iterdir()):
        if not d.is_dir():
            continue
        train_cfg = d / "train_config.json"
        cfg_obj: Dict[str, Any] = {}
        if train_cfg.exists():
            try:
                cfg_obj = json.loads(train_cfg.read_text(encoding="utf-8"))
            except Exception:
                cfg_obj = {}
        trainings.append({"folder": d.name, "config": cfg_obj})
    return trainings


@app.post("/start")
def api_start_training(cfg: WebTrainConfig):
    cfg.csv_path = _resolve_hf_dataset_to_csv(cfg)
    res = TRAIN_MANAGER.start(cfg)
    if not res.get("running"):
        return JSONResponse(res, status_code=400)
    return res


@app.post("/stop")
def api_stop_training():
    return TRAIN_MANAGER.stop()


@app.get("/status")
def api_status():
    return TRAIN_MANAGER.status()


@app.get("/logs")
def api_logs():
    return {"log": TRAIN_STATE.log.tail(200)}


@app.get("/livepreview")
def api_livepreview():
    return TRAIN_MANAGER.livepreview()


@app.get("/inference_status")
def api_inference_status():
    with INFER.lock:
        return {
            "loaded_dir": INFER.loaded_dir,
            "device": INFER.device.type if INFER.device else None,
            "is_generating": INFER.is_generating,
            "current_source": INFER.current_source,
            "last_started_at": INFER.last_started_at,
            "latest_available_model": get_latest_model_dir(),
        }


@app.websocket("/ws/logs")
async def ws_logs(websocket: WebSocket):
    await websocket.accept()
    cursor = TRAIN_STATE.log.last_id - 200
    try:
        while True:
            lines, cursor = TRAIN_STATE.log.since(cursor)
            if lines:
                await websocket.send_text("\n".join(lines))
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        return


@app.post("/load_inference")
def api_load_inference(cfg: Dict[str, Any] | None = Body(default=None)):
    cfg = cfg or {}
    model_dir = cfg.get("model_dir") or TRAIN_MANAGER.status().get("save_dir") or cfg.get("fallback_model_dir") or get_latest_model_dir()
    if not model_dir:
        return JSONResponse({"error": "Kein Modell gefunden."}, status_code=400)
    try:
        res = load_inference_model(model_dir, cfg.get("device") or "auto")
        if res.get("error"):
            return JSONResponse(res, status_code=409)
        return JSONResponse(res, status_code=200)
    except FileNotFoundError as exc:
        return JSONResponse({"error": str(exc)}, status_code=404)
    except Exception as exc:
        return JSONResponse({"error": f"{exc.__class__.__name__}: {exc}"}, status_code=500)


@app.post("/unload_inference")
def api_unload_inference():
    res = unload_inference()
    if res.get("error"):
        return JSONResponse(res, status_code=409)
    return res


@app.post("/chat")
def api_chat(req: ChatRequest):
    try:
        model_dir, tok, mdl, dev = ensure_model_loaded(req.model_dir, req.device)
    except RuntimeError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except Exception as exc:
        return JSONResponse({"error": f"{exc.__class__.__name__}: {exc}"}, status_code=500)

    input_ids, attention_mask, pad_id, eos_id = prepare_inputs(tok, req.messages, req.system, dev)
    gen_kwargs, logits_processor = sanitize_sampling_args(
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
        repetition_penalty=req.repetition_penalty,
        do_sample=req.do_sample,
        pad_id=pad_id,
        eos_id=eos_id,
    )

    with INFER.generate_lock:
        begin_generation("web")
        try:
            with torch.no_grad():
                out = mdl.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    logits_processor=logits_processor,
                    num_beams=1,
                    num_return_sequences=1,
                    no_repeat_ngram_size=3,
                    **gen_kwargs,
                )
        finally:
            end_generation()

    seq = _extract_sequences(out)
    generated = seq[0, input_ids.shape[-1]:]
    text = tok.decode(generated, skip_special_tokens=True)
    return {"model_dir": model_dir, "response": text.strip()}


@app.post("/chat_stream")
def api_chat_stream(req: ChatRequest):
    from transformers import TextIteratorStreamer

    try:
        model_dir, tok, mdl, dev = ensure_model_loaded(req.model_dir, req.device)
    except RuntimeError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except Exception as exc:
        return JSONResponse({"error": f"{exc.__class__.__name__}: {exc}"}, status_code=500)

    input_ids, attention_mask, pad_id, eos_id = prepare_inputs(tok, req.messages, req.system, dev)
    gen_kwargs, logits_processor = sanitize_sampling_args(
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
        repetition_penalty=req.repetition_penalty,
        do_sample=req.do_sample,
        pad_id=pad_id,
        eos_id=eos_id,
    )
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)

    def _gen():
        with INFER.generate_lock:
            begin_generation("web_stream")
            try:
                with torch.no_grad():
                    mdl.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        logits_processor=logits_processor,
                        num_beams=1,
                        num_return_sequences=1,
                        no_repeat_ngram_size=3,
                        streamer=streamer,
                        **gen_kwargs,
                    )
            except Exception:
                pass
            finally:
                end_generation()

    t = threading.Thread(target=_gen, daemon=True)
    t.start()

    def event_iter():
        try:
            for text in streamer:
                yield text
        except Exception:
            return

    return StreamingResponse(event_iter(), media_type="text/plain; charset=utf-8")


def _auth_dependency(authorization: Optional[str] = Header(default=None)):
    if OPENAI_COMPAT_API_KEY:
        if not authorization or not authorization.lower().startswith("bearer "):
            raise HTTPException(status_code=401, detail="Missing bearer token")
        token = authorization.split(" ", 1)[1].strip()
        if token != OPENAI_COMPAT_API_KEY:
            raise HTTPException(status_code=401, detail="Invalid API key")
    return True


def _oai_error(message: str, status_code: int = 400, code: Optional[str] = None):
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": "invalid_request_error", "param": None, "code": code}},
    )


class OAIChatMessage(MatelixBaseModel):
    role: str
    content: str


class OAIChatRequest(MatelixBaseModel):
    model: Optional[str] = None
    messages: List[OAIChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = 40
    max_tokens: Optional[int] = 4096
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    repetition_penalty: Optional[float] = 1.1
    do_sample: Optional[bool] = None
    seed: Optional[int] = None
    device: Optional[str] = None
    user: Optional[str] = None


class OAICompletionRequest(MatelixBaseModel):
    model: Optional[str] = None
    prompt: Union[str, List[str]]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = 40
    max_tokens: Optional[int] = 4096
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    repetition_penalty: Optional[float] = 1.1
    do_sample: Optional[bool] = None
    seed: Optional[int] = None
    device: Optional[str] = None
    user: Optional[str] = None


def _extract_system(msgs: List[OAIChatMessage]) -> Optional[str]:
    for m in msgs:
        if m.role.strip().lower() == "system":
            return m.content
    return None


def _messages_to_hf(msgs: List[OAIChatMessage]) -> List[Dict[str, str]]:
    out = []
    for m in msgs:
        r = m.role.strip().lower()
        if r in ("user", "assistant"):
            out.append({"role": r, "content": m.content})
    return out


@app.get("/v1/models", dependencies=[Depends(_auth_dependency)])
def oai_models():
    local = api_available_models()
    current = INFER.loaded_dir
    models: List[str] = []

    if current:
        models.append(current)

    latest = get_latest_model_dir()
    if latest and latest not in models:
        models.append(latest)

    for m in local:
        path = str(TRAINING_OUT_DIR / m)
        if path not in models:
            models.append(path)

    return {
        "object": "list",
        "data": [{"id": mid, "object": "model", "created": 0, "owned_by": "owner"} for mid in models],
    }


@app.post("/v1/chat/completions", dependencies=[Depends(_auth_dependency)])
def oai_chat(req: OAIChatRequest):
    try:
        model_id, tok, mdl, dev = ensure_model_loaded(req.model, req.device)
    except Exception as e:
        return _oai_error(str(e), status_code=500)

    if tok is None or mdl is None or dev is None:
        return _oai_error("Inference model not loaded", status_code=500)

    system = _extract_system(req.messages)
    msgs = _messages_to_hf(req.messages)
    if not msgs:
        return _oai_error("messages is empty")

    input_ids, attention_mask, pad_id, eos_id = _prepare_inputs(tok, msgs, system, dev)
    temperature = 0.0 if req.temperature is None else float(req.temperature)
    do_sample = req.do_sample if req.do_sample is not None else (temperature > 0)

    gen_kwargs, logits_processor = sanitize_sampling_args(
        max_new_tokens=int(req.max_tokens or 256),
        temperature=temperature,
        top_p=req.top_p,
        top_k=req.top_k,
        repetition_penalty=req.repetition_penalty,
        do_sample=do_sample,
        pad_id=pad_id,
        eos_id=eos_id,
    )

    created = int(time.time())
    resp_id = f"chatcmpl-{uuid.uuid4().hex}"

    stops = req.stop
    if isinstance(stops, str):
        stops = [stops]

    stopping_criteria = None
    if stops:
        from transformers import StoppingCriteria, StoppingCriteriaList

        class StringStopCriteria(StoppingCriteria):
            def __init__(self, tokenizer, stop_strings: List[str]):
                self.tokenizer = tokenizer
                self.stop_strings = stop_strings
                self.buffer = ""

            def __call__(self, input_ids, scores, **kwargs):
                try:
                    new_text = self.tokenizer.decode(input_ids[0, -1:], skip_special_tokens=False)
                except Exception:
                    return False
                self.buffer += new_text
                return any(s in self.buffer for s in self.stop_strings)

        stopping_criteria = StoppingCriteriaList([StringStopCriteria(tok, stops)])

    if req.seed is not None:
        try:
            torch.manual_seed(int(req.seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(req.seed))
        except Exception:
            pass

    if req.stream:
        from transformers import TextIteratorStreamer

        streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)

        def _run_stream():
            with INFER.generate_lock:
                begin_generation("openai_stream")
                try:
                    with torch.no_grad():
                        mdl.generate(
                            input_ids=input_ids[:1, :],
                            attention_mask=attention_mask[:1, :] if attention_mask is not None else None,
                            streamer=streamer,
                            logits_processor=logits_processor,
                            num_beams=1,
                            num_return_sequences=1,
                            no_repeat_ngram_size=3,
                            stopping_criteria=stopping_criteria,
                            **gen_kwargs,
                        )
                except Exception:
                    pass
                finally:
                    end_generation()

        t = threading.Thread(target=_run_stream, daemon=True)
        t.start()

        def sse():
            first = {
                "id": resp_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_id,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(first, ensure_ascii=False)}\n\n"

            for piece in streamer:
                yield "data: " + json.dumps(
                    {
                        "id": resp_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_id,
                        "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
                    },
                    ensure_ascii=False,
                ) + "\n\n"

            yield "data: " + json.dumps(
                {
                    "id": resp_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_id,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                },
                ensure_ascii=False,
            ) + "\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream")

    with INFER.generate_lock:
        begin_generation("openai")
        try:
            with torch.no_grad():
                out = mdl.generate(
                    input_ids=input_ids[:1, :],
                    attention_mask=attention_mask[:1, :] if attention_mask is not None else None,
                    logits_processor=logits_processor,
                    num_beams=1,
                    num_return_sequences=1,
                    no_repeat_ngram_size=3,
                    stopping_criteria=stopping_criteria,
                    **gen_kwargs,
                )
        finally:
            end_generation()

    seq = _extract_sequences(out)
    prompt_len = input_ids.shape[-1]
    generated = seq[0, prompt_len:]
    text = tok.decode(generated, skip_special_tokens=True).strip()

    usage = {
        "prompt_tokens": int(input_ids.numel()),
        "completion_tokens": int(generated.numel()),
        "total_tokens": int(input_ids.numel() + generated.numel()),
    }

    return {
        "id": resp_id,
        "object": "chat.completion",
        "created": created,
        "model": model_id,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": usage,
        "system_fingerprint": None,
    }


@app.post("/v1/completions", dependencies=[Depends(_auth_dependency)])
def oai_completions(req: OAICompletionRequest):
    try:
        model_id, tok, mdl, dev = ensure_model_loaded(req.model, req.device)
    except Exception as e:
        return _oai_error(str(e), status_code=500)

    if tok is None or mdl is None or dev is None:
        return _oai_error("Inference model not loaded", status_code=500)

    prompts = req.prompt if isinstance(req.prompt, list) else [req.prompt]
    created = int(time.time())
    results = []
    last_usage = None

    if req.seed is not None:
        try:
            torch.manual_seed(int(req.seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(req.seed))
        except Exception:
            pass

    for idx, p in enumerate(prompts):
        msgs = [{"role": "user", "content": p}]
        input_ids, attention_mask, pad_id, eos_id = _prepare_inputs(tok, msgs, None, dev)

        temperature = 0.0 if req.temperature is None else float(req.temperature)
        do_sample = req.do_sample if req.do_sample is not None else (temperature > 0)

        gen_kwargs, logits_processor = sanitize_sampling_args(
            max_new_tokens=int(req.max_tokens or 256),
            temperature=temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            repetition_penalty=req.repetition_penalty,
            do_sample=do_sample,
            pad_id=pad_id,
            eos_id=eos_id,
        )

        with INFER.generate_lock:
            begin_generation("openai_completion")
            try:
                with torch.no_grad():
                    out = mdl.generate(
                        input_ids=input_ids[:1, :],
                        attention_mask=attention_mask[:1, :] if attention_mask is not None else None,
                        logits_processor=logits_processor,
                        num_beams=1,
                        num_return_sequences=1,
                        no_repeat_ngram_size=3,
                        **gen_kwargs,
                    )
            finally:
                end_generation()

        seq = _extract_sequences(out)
        gen_ids = seq[0, input_ids.shape[-1]:]
        text = tok.decode(gen_ids, skip_special_tokens=True).strip()

        last_usage = {
            "prompt_tokens": int(input_ids.numel()),
            "completion_tokens": int(gen_ids.numel()),
            "total_tokens": int(input_ids.numel() + gen_ids.numel()),
        }
        results.append({"text": text, "index": idx, "logprobs": None, "finish_reason": "stop"})

    return {
        "id": f"cmpl-{uuid.uuid4().hex}",
        "object": "text_completion",
        "created": created,
        "model": model_id,
        "choices": results,
        "usage": last_usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


@app.get("/", response_class=HTMLResponse)
def root(_: Request):
    return FileResponse(str(ensure_index_html()))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
