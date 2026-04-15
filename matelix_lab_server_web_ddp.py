#!/usr/bin/env python3
# matelix_lab_server_web_ddp.py
#
# Web-first MaTeLiX Server:
# - startet DDP-Training über die bestehende Weboberfläche
# - kein manuelles torchrun im Terminal nötig
# - kompatibel mit der vorhandenen index.html
#
# Architektur:
#   Browser -> FastAPI -> subprocess (python -m torch.distributed.run ...) -> DDP Worker
#
from __future__ import annotations

import asyncio
import csv
import gc
import json
import os
import platform
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil
import torch
from fastapi import Body, FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
csv.field_size_limit(1024 * 1024 * 128)

BASE_DIR = Path(__file__).resolve().parent
TRAINING_OUT_DIR = BASE_DIR / "training_outputs"
DATASETS_DIR = BASE_DIR / "datasets"
STATIC_DIR = BASE_DIR / "static"
WORKER_PATH = BASE_DIR / "matelix_ddp_worker.py"

TRAINING_OUT_DIR.mkdir(parents=True, exist_ok=True)
DATASETS_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="MaTeLiX AI Lab (Web DDP)", version="5.1-web-ddp")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/training_outputs", StaticFiles(directory=str(TRAINING_OUT_DIR)), name="training_outputs")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------
# Fallback index
# ---------------------------------------------------------------------

DEFAULT_INDEX_HTML = """<!DOCTYPE html>
<html lang="de">
<head>
  <meta charset="utf-8"/>
  <title>MaTeLiX AI LAB</title>
  <style>
    body { background:#07110c; color:#c8ffdd; font-family:Arial,sans-serif; display:flex; align-items:center; justify-content:center; height:100vh; margin:0; }
    .card { background:#102119; border:1px solid #2aff8f; padding:2rem; border-radius:16px; max-width:640px; box-shadow:0 0 24px #1cff7c33; }
    code { background:#06140e; padding:0.2rem 0.5rem; border-radius:6px; border:1px solid #1cff7c55; }
  </style>
</head>
<body>
  <div class="card">
    <h1>MaTeLiX LAB Backend läuft</h1>
    <p>Lege deine Oberfläche unter <code>./static/index.html</code> ab.</p>
    <p>Der Server startet DDP-Training direkt aus der Weboberfläche.</p>
  </div>
</body>
</html>
"""

def ensure_index_html() -> Path:
    p = STATIC_DIR / "index.html"
    if not p.exists():
        p.write_text(DEFAULT_INDEX_HTML, encoding="utf-8")
    return p


# ---------------------------------------------------------------------
# Logging store
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# Runtime state
# ---------------------------------------------------------------------

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
            }


TRAIN_STATE = TrainingState()


# ---------------------------------------------------------------------
# Web TrainConfig
# ---------------------------------------------------------------------

class WebTrainConfig(BaseModel):
    model_dir: str = Field(...)
    csv_path: str = Field(...)

    save_dir: Optional[str] = None
    template_mode: str = "chat"
    column_name: str = "text"

    learning_rate: float = 2e-5
    lr_schedule: str = "cosine"
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    num_train_epochs: float = 3.0
    max_steps: Optional[int] = None
    max_seq_length: int = 1024
    shuffle: bool = False
    sort_by_length: bool = True

    device: str = "auto"
    train_mode: str = "full"
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

    # neue optionale Web-DDP Felder
    ddp_enabled: Optional[bool] = None
    nproc_per_node: Optional[int] = None
    nnodes: int = 1
    node_rank: int = 0
    master_addr: str = "127.0.0.1"
    master_port: int = 29500
    seed: int = 42
    deterministic: bool = True

    # beide Felder erlaubt; Server mappt sauber auf ddp_find_unused_parameters
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


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# DDP subprocess manager
# ---------------------------------------------------------------------

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

    def _is_alive(self) -> bool:
        with self.lock:
            return self.proc is not None and self.proc.poll() is None

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
            self.state.last_preview_full = (payload.get("preview_full") or self.state.last_preview_full or self.state.last_preview or "")[:20000]
            self.state.save_dir = payload.get("save_dir") or self.state.save_dir
            self.state.status_text = payload.get("status") or self.state.status_text
            self.state.error = payload.get("error") or self.state.error
            self.state.world_size = int(payload.get("world_size") or self.state.world_size or 1)
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
                # Falls noch keine State-Datei geschrieben wurde, wenigstens "läuft" anzeigen.
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

        # sicherer Default für Multi-GPU CUDA mit sparsely-used Parametern / MoE / Router
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

        worker_cfg = cfg.model_dump(exclude={
            "save_dir",
            "ddp_enabled",
            "nproc_per_node",
            "nnodes",
            "node_rank",
            "master_addr",
            "master_port",
            "find_unused_parameters",  # server mappt sauber selbst
        })

        worker_cfg["device"] = device
        worker_cfg["output_dir"] = str(run_dir)
        worker_cfg["save_dir"] = str(run_dir)
        worker_cfg["ddp_find_unused_parameters"] = self._resolve_find_unused(cfg, ddp_enabled, nproc, device)

        # DDP + dynamischer Graph + checkpointing => static_graph sicherheitshalber aus
        if bool(worker_cfg.get("gradient_checkpointing")) and worker_cfg["ddp_find_unused_parameters"]:
            worker_cfg["ddp_static_graph"] = False

        cfg_path = run_dir / "worker_config.json"
        atomic_write_json(cfg_path, worker_cfg)
        atomic_write_json(
            run_dir / "train_config.json",
            {
                **cfg.model_dump(),
                "device": device,
                "output_dir": str(run_dir),
                "effective_world_size": nproc,
                "effective_ddp_enabled": bool(ddp_enabled and nproc > 1),
                "effective_ddp_find_unused_parameters": worker_cfg["ddp_find_unused_parameters"],
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
        env.pop("NCCL_ASYNC_ERROR_HANDLING", None)
        env["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
        env.setdefault("TORCH_DISTRIBUTED_DEBUG", "DETAIL")

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
            self.state.log.clear()
            self.state.log.set_file(run_dir / "training.log")
            self.state.log.append("Training gestartet.")
            self.state.log.append("Command: " + " ".join(cmd))
            self.state.log.append(f"DDP effective: enabled={bool(ddp_enabled and nproc > 1)} world_size={nproc} find_unused_parameters={worker_cfg['ddp_find_unused_parameters']}")

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
            if os.name != "nt":
                os.killpg(proc.pid, signal.SIGTERM)
            else:
                proc.terminate()
        except Exception as exc:
            self.state.log.append(f"[STOP WARN] {exc}")

        def _kill_later(p: subprocess.Popen):
            try:
                p.wait(timeout=20)
            except Exception:
                try:
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


# ---------------------------------------------------------------------
# Inference / Chat
# ---------------------------------------------------------------------

def prepare_tokenizer_for_matelix(tokenizer) -> bool:
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
        self.loaded_dir: Optional[str] = None
        self.device: Optional[torch.device] = None
        self.tokenizer = None
        self.model = None


INFER = InferenceSession()


class ChatRequest(BaseModel):
    model_dir: Optional[str] = None
    device: Optional[str] = None
    system: Optional[str] = None
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.05
    do_sample: bool = True


def _preferred_device_name(req: Optional[str] = None) -> str:
    return normalize_device(req or "auto")


def _get_device(name: str) -> torch.device:
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if name == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def unload_inference() -> Dict[str, Any]:
    with INFER.lock:
        unload_torch_objects(INFER.model, INFER.tokenizer)
        INFER.loaded_dir = None
        INFER.device = None
        INFER.tokenizer = None
        INFER.model = None
    return {"msg": "Inferenz-Modell entladen."}


def load_inference_model(model_dir: str, device_name: str = "auto") -> Dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev_name = _preferred_device_name(device_name)
    dev = _get_device(dev_name)

    with INFER.lock:
        if INFER.loaded_dir and (INFER.loaded_dir != model_dir or (INFER.device and INFER.device.type != dev.type)):
            unload_inference()

        tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=False)
        need_resize = prepare_tokenizer_for_matelix(tok)
        dtype = torch.float16 if dev.type == "cuda" else None

        mdl = AutoModelForCausalLM.from_pretrained(
            model_dir,
            trust_remote_code=False,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
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
    mdl_dir = model_dir or INFER.loaded_dir
    if not mdl_dir:
        raise RuntimeError("Kein Inferenz-Modell geladen. Nutze /load_inference oder gib model_dir an.")
    dev_name = _preferred_device_name(device_name or (INFER.device.type if INFER.device else "auto"))
    target_dev = _get_device(dev_name)
    if INFER.loaded_dir != mdl_dir or (INFER.device and INFER.device.type != target_dev.type):
        load_inference_model(mdl_dir, dev_name)
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
        parts = []
        for m in msgs:
            role = (m.get("role") or "").strip().lower()
            content = (m.get("content") or "").strip()
            if not content:
                continue
            if role == "system":
                parts.append("<|System|>\n" + content + "\n</s>\n")
            elif role == "user":
                parts.append("<|Benutzer|>\n" + content + "\n</s>\n")
            elif role == "assistant":
                parts.append("<|Assistentin|>\n" + content + "\n</s>\n")
        parts.append("<|Assistentin|>\n")
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
    k = 50 if top_k is None else int(top_k)
    rp = 1.05 if repetition_penalty is None else float(repetition_penalty)
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


# ---------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------

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
    model_dir = cfg.get("model_dir") or TRAIN_MANAGER.status().get("save_dir") or cfg.get("fallback_model_dir")
    if not model_dir:
        return JSONResponse({"error": "model_dir fehlt."}, status_code=400)
    try:
        return JSONResponse(load_inference_model(model_dir, cfg.get("device") or "auto"), status_code=200)
    except FileNotFoundError as exc:
        return JSONResponse({"error": str(exc)}, status_code=404)
    except Exception as exc:
        return JSONResponse({"error": f"{exc.__class__.__name__}: {exc}"}, status_code=500)


@app.post("/unload_inference")
def api_unload_inference():
    return unload_inference()


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

    with torch.no_grad():
        out = mdl.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            logits_processor=logits_processor,
            num_beams=1,
            num_return_sequences=1,
            **gen_kwargs,
        )

    seq = out.sequences if hasattr(out, "sequences") else out
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
        try:
            with torch.no_grad():
                mdl.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    logits_processor=logits_processor,
                    num_beams=1,
                    num_return_sequences=1,
                    streamer=streamer,
                    **gen_kwargs,
                )
        except Exception:
            pass

    t = threading.Thread(target=_gen, daemon=True)
    t.start()

    def event_iter():
        try:
            for text in streamer:
                yield text
        except Exception:
            return

    return StreamingResponse(event_iter(), media_type="text/plain; charset=utf-8")


@app.get("/", response_class=HTMLResponse)
def root(_: Request):
    return FileResponse(str(ensure_index_html()))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
