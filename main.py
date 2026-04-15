#!/usr/bin/env python3
"""
main.py

Ein modulares, production-taugliches PyTorch-DDP-Trainingsframework für Causal-LM-Training
auf CSV-Daten (Text / Chat / Dialogplus) mit nativer torch.distributed-Integration.

Ziele dieser Vorlage
--------------------
- Native PyTorch DDP statt Trainer-Abstraktion.
- Saubere Modularisierung für Dataset, Modell, Train/Validierung, Logging, Checkpoints und CLI/YAML.
- Stabile Mehr-GPU-Ausführung mit torchrun, optional Multi-Node via env://.
- Deterministische Ausführung, saubere Signalbehandlung und robustes Resume.
- Hook-Points für eigene Modelle, Tokenizer, Datasets und Batch-Logik.

Beispielaufrufe
---------------
Single GPU / CPU:
    python main.py --model_name_or_path gpt2 --train_csv ./datasets/train.csv --template_mode text

Multi-GPU (1 Node):
    torchrun --standalone --nproc_per_node=4 main.py \
        --model_name_or_path /pfad/zum/modell \
        --train_csv ./datasets/train.csv \
        --template_mode chat

Multi-Node:
    torchrun \
      --nnodes=2 \
      --nproc_per_node=4 \
      --node_rank=0 \
      --master_addr=10.0.0.1 \
      --master_port=29500 \
      main.py --config ./config.yaml

Minimale Abhängigkeiten
-----------------------
Pflicht:
- torch
- transformers

Optional:
- pyyaml      (für --config *.yaml)
- tensorboard (für SummaryWriter)
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
import shutil
import signal
import sys
import time
import traceback
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

# Muss vor dem ersten CUDA-Use gesetzt sein.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
os.environ.setdefault("PYTHONUNBUFFERED", "1")
# Für deterministische CUBLAS-Pfade notwendig, wenn Determinismus aktiviert ist.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.distributed as dist
from torch import nn
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer


try:
    import yaml
except Exception:  # pragma: no cover - optional dependency
    yaml = None

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - optional dependency
    SummaryWriter = None

try:
    from torch.distributed.elastic.multiprocessing.errors import record
except Exception:  # pragma: no cover - optional dependency
    def record(fn):
        return fn


csv.field_size_limit(1024 * 1024 * 128)

LOGGER = logging.getLogger("ddp_trainer")


# ============================================================================
# Konfiguration
# ============================================================================

@dataclass
class TrainConfig:
    """
    Vollständige Konfiguration für das Training.

    Die Defaults sind bewusst konservativ gewählt:
    - Stabilität vor maximaler Aggressivität
    - Determinismus standardmäßig aktiv
    - TensorBoard optional
    """

    # Eingaben
    model_name_or_path: str = ""
    train_csv: str = ""
    val_csv: Optional[str] = None
    template_mode: str = "text"  # text | chat | dialogplus
    text_column: str = "text"

    # Ausgabe / Experiment-Struktur
    experiment_name: Optional[str] = None
    output_root: str = "./runs"

    # Datenaufteilung
    val_split: float = 0.05
    split_seed: int = 42

    # Optimierung
    learning_rate: float = 2.0e-5
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.95)
    eps: float = 1.0e-8
    num_epochs: int = 3
    max_steps: Optional[int] = None
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_grad_norm: float = 1.0

    # Scheduler
    scheduler: str = "cosine"  # cosine | linear | constant
    warmup_steps: int = 0
    warmup_ratio: float = 0.03
    min_lr_ratio: float = 0.0

    # Sequenz / Dataloader
    max_seq_length: int = 1024
    sort_by_length: bool = True
    shuffle_train_examples: bool = True
    num_workers: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    pad_to_multiple_of: int = 8

    # Precision / Performance
    mixed_precision: str = "bf16"  # none | fp16 | bf16
    gradient_checkpointing: bool = False
    tf32: bool = True
    compile_model: bool = False
    compile_mode: str = "default"

    # DDP
    ddp_backend: Optional[str] = None  # auto => nccl / gloo
    ddp_timeout_minutes: int = 30
    broadcast_buffers: bool = False
    find_unused_parameters: bool = False
    static_graph: bool = False
    sync_batchnorm: bool = False

    # Determinismus / Stabilität
    seed: int = 42
    deterministic: bool = True
    deterministic_warn_only: bool = False

    # Validierung / Early Stopping
    validate_every_epoch: bool = True
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 0.0
    monitor_metric: str = "val_loss"
    monitor_mode: str = "min"  # min | max

    # Logging / Monitoring
    log_every_steps: int = 10
    use_tensorboard: bool = True
    save_every_epoch: bool = True
    keep_last_k_checkpoints: int = 3

    # Resume
    resume: Optional[str] = None  # Datei oder "auto"

    # Export / Hook-Steuerung
    trust_remote_code: bool = False

    def validate(self) -> None:
        """Validiert die Konfiguration frühzeitig, bevor Prozesse gestartet werden."""
        if not self.model_name_or_path:
            raise ValueError("--model_name_or_path ist erforderlich.")
        if not self.train_csv:
            raise ValueError("--train_csv ist erforderlich.")
        if self.template_mode not in {"text", "chat", "dialogplus"}:
            raise ValueError("--template_mode muss einer von {text, chat, dialogplus} sein.")
        if self.scheduler not in {"cosine", "linear", "constant"}:
            raise ValueError("--scheduler muss einer von {cosine, linear, constant} sein.")
        if self.mixed_precision not in {"none", "fp16", "bf16"}:
            raise ValueError("--mixed_precision muss einer von {none, fp16, bf16} sein.")
        if self.monitor_mode not in {"min", "max"}:
            raise ValueError("--monitor_mode muss min oder max sein.")
        if self.per_device_batch_size < 1:
            raise ValueError("--per_device_batch_size muss >= 1 sein.")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("--gradient_accumulation_steps muss >= 1 sein.")
        if self.max_seq_length < 8:
            raise ValueError("--max_seq_length ist zu klein.")
        if not (0.0 <= self.val_split < 1.0):
            raise ValueError("--val_split muss im Intervall [0, 1) liegen.")
        if self.num_epochs < 1:
            raise ValueError("--num_epochs muss >= 1 sein.")
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError("--max_steps muss >= 1 sein oder leer bleiben.")
        if self.early_stopping_patience < 1:
            raise ValueError("--early_stopping_patience muss >= 1 sein.")
        if self.keep_last_k_checkpoints < 1:
            raise ValueError("--keep_last_k_checkpoints muss >= 1 sein.")
        if self.num_workers < 0:
            raise ValueError("--num_workers muss >= 0 sein.")


# ============================================================================
# DDP / Prozesskontext
# ============================================================================

@dataclass
class DistributedContext:
    """
    Enthält den vollständigen Prozesskontext für Single-Process oder DDP.

    world_size=1 bedeutet nicht-distribuiert.
    """

    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    is_distributed: bool = False
    device: torch.device = torch.device("cpu")

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


@dataclass
class ExperimentPaths:
    """Sammelt alle wichtigen Verzeichnisse eines Laufs an einer Stelle."""
    run_dir: Path
    checkpoints_dir: Path
    logs_dir: Path
    tensorboard_dir: Path
    configs_dir: Path
    artifacts_dir: Path


class ShutdownState:
    """
    Prozesslokaler Shutdown-Zustand.

    Bei SIGINT / SIGTERM setzen wir nur ein Flag. Die Trainingsschleife liest
    dieses Flag regelmäßig und beendet sich kontrolliert an sicheren Punkten.
    """

    def __init__(self) -> None:
        self.stop_requested = False
        self.reason = ""

    def request_stop(self, reason: str) -> None:
        self.stop_requested = True
        self.reason = reason


SHUTDOWN = ShutdownState()


def register_signal_handlers() -> None:
    """Registriert robuste Signal-Handler für SIGINT / SIGTERM."""
    def _handler(signum: int, _frame: Any) -> None:
        signame = signal.Signals(signum).name
        SHUTDOWN.request_stop(signame)

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


def init_distributed(cfg: TrainConfig) -> DistributedContext:
    """
    Initialisiert DDP anhand der torchrun-Umgebungsvariablen.

    Erwartetes Startmuster:
        torchrun --standalone --nproc_per_node=4 main.py ...

    Für Single-Process-Aufrufe ohne torchrun wird kein Prozessverbund initialisiert.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_distributed = world_size > 1

    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank if is_distributed else 0)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    ctx = DistributedContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        is_distributed=is_distributed,
        device=device,
    )

    if is_distributed:
        backend = cfg.ddp_backend or ("nccl" if device.type == "cuda" else "gloo")
        timeout = torch.distributed.constants.default_pg_timeout
        try:
            from datetime import timedelta
            timeout = timedelta(minutes=cfg.ddp_timeout_minutes)
        except Exception:
            pass
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            timeout=timeout,
        )
    return ctx


def cleanup_distributed() -> None:
    """Räumt den Prozessverbund am Ende sauber auf."""
    if dist.is_available() and dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception:
            pass


def abort_distributed() -> None:
    """
    Erzwingt das Ende aller Prozesse bei fatalen Fehlern.

    Das ist bewusst aggressiver als destroy_process_group(), weil destroy bei
    inkonsistenten Zuständen zu Hängern führen kann.
    """
    if dist.is_available() and dist.is_initialized():
        try:
            dist.abort()
            return
        except Exception:
            pass
        try:
            dist.destroy_process_group()
        except Exception:
            pass


def barrier(ctx: DistributedContext) -> None:
    """Sicherer Barrier-Wrapper nur für reguläre, fehlerfreie Pfade."""
    if ctx.is_distributed and dist.is_initialized():
        dist.barrier()


def broadcast_object(obj: Any, ctx: DistributedContext, src: int = 0) -> Any:
    """
    Broadcast für Python-Objekte.

    Wird z. B. für run_dir oder automatisch gefundene Resume-Pfade benutzt.
    """
    if not ctx.is_distributed:
        return obj
    objects = [obj if ctx.rank == src else None]
    dist.broadcast_object_list(objects, src=src)
    return objects[0]


def all_reduce_sum(value: float, ctx: DistributedContext, device: torch.device) -> float:
    """Reduziert einen Float-Wert über alle Prozesse per Summe."""
    if not ctx.is_distributed:
        return float(value)
    tensor = torch.tensor(float(value), device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


def sync_stop_flag(local_stop: bool, ctx: DistributedContext, device: torch.device) -> bool:
    """
    Synchronisiert ein Stop-Flag über alle Prozesse.

    Wenn irgendein Rank stoppen möchte, stoppen alle. So vermeiden wir Deadlocks,
    bei denen einzelne Prozesse aus der Schleife aussteigen und andere weiterlaufen.
    """
    if not ctx.is_distributed:
        return local_stop
    flag = torch.tensor([1 if local_stop else 0], device=device, dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item())


def unwrap_model(model: nn.Module) -> nn.Module:
    """Entfernt DDP-Hüllen für Speichern / Export / direkte Modellmethoden."""
    return model.module if isinstance(model, DDP) else model


# ============================================================================
# Logging / Dateien / Config-IO
# ============================================================================

def setup_logging(paths: ExperimentPaths, ctx: DistributedContext) -> None:
    """
    Initialisiert Logging.

    - Rank 0: Datei + stdout auf INFO
    - Andere Ranks: stdout nur ERROR, damit Logs nicht vervielfacht werden
    """
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

    if ctx.is_main_process:
        file_handler = logging.FileHandler(paths.logs_dir / "train.log", encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(RankFilter())
        LOGGER.addHandler(file_handler)

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        stream_handler.addFilter(RankFilter())
        LOGGER.addHandler(stream_handler)
    else:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.ERROR)
        stream_handler.setFormatter(formatter)
        stream_handler.addFilter(RankFilter())
        LOGGER.addHandler(stream_handler)


def rank0_log(ctx: DistributedContext, message: str) -> None:
    """Hilfsfunktion für Rank-0-Logging."""
    if ctx.is_main_process:
        LOGGER.info(message)


def ensure_experiment_paths(
    cfg: TrainConfig,
    ctx: DistributedContext,
    resolved_resume_path: Optional[Path] = None,
) -> ExperimentPaths:
    """
    Erzeugt die Ausgabe-Struktur und broadcastet den finalen run_dir an alle Ranks.

    Verhalten:
    - frischer Lauf -> neues Verzeichnis
    - Resume        -> bestehendes run_dir des Checkpoints wiederverwenden

    Struktur:
        runs/<experiment>/
            checkpoints/
            logs/
            logs/tensorboard/
            configs/
            artifacts/
    """
    run_dir: Optional[str] = None
    if ctx.is_main_process:
        if resolved_resume_path is not None and resolved_resume_path.exists():
            run_path = resolved_resume_path.parent.parent.resolve()
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            exp_name = cfg.experiment_name or Path(cfg.model_name_or_path.rstrip("/")).name or "experiment"
            safe_name = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in exp_name)
            run_path = Path(cfg.output_root).expanduser().resolve() / f"{timestamp}_{safe_name}"

        run_path.mkdir(parents=True, exist_ok=True)
        for subdir in ["checkpoints", "logs", "logs/tensorboard", "configs", "artifacts"]:
            (run_path / subdir).mkdir(parents=True, exist_ok=True)
        run_dir = str(run_path)

    run_dir = broadcast_object(run_dir, ctx)
    if run_dir is None:
        raise RuntimeError("run_dir konnte nicht initialisiert werden.")

    paths = ExperimentPaths(
        run_dir=Path(run_dir),
        checkpoints_dir=Path(run_dir) / "checkpoints",
        logs_dir=Path(run_dir) / "logs",
        tensorboard_dir=Path(run_dir) / "logs" / "tensorboard",
        configs_dir=Path(run_dir) / "configs",
        artifacts_dir=Path(run_dir) / "artifacts",
    )
    return paths


def save_resolved_config(cfg: TrainConfig, paths: ExperimentPaths, ctx: DistributedContext) -> None:
    """Speichert die aufgelöste Konfiguration in JSON und optional YAML."""
    if not ctx.is_main_process:
        return
    cfg_dict = asdict(cfg)
    (paths.configs_dir / "resolved_config.json").write_text(
        json.dumps(cfg_dict, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if yaml is not None:
        with open(paths.configs_dir / "resolved_config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg_dict, f, sort_keys=False, allow_unicode=True)


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    """Schreibt eine JSON-Zeile in eine Metrikdatei."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def resolve_resume_request(cfg: TrainConfig) -> Optional[Path]:
    """
    Löst die Resume-Anfrage in einen konkreten Checkpoint-Pfad auf.

    Regeln:
    - None  -> kein Resume
    - auto  -> neuester last.pt unter output_root
    - Pfad  -> direkter Dateipfad
    """
    if not cfg.resume:
        return None

    if cfg.resume == "auto":
        root = Path(cfg.output_root).expanduser().resolve()
        if not root.exists():
            return None

        candidates = list(root.glob("*/checkpoints/last.pt"))
        if not candidates:
            return None

        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0].resolve()

    return Path(cfg.resume).expanduser().resolve()


def load_yaml_config(path: str) -> Dict[str, Any]:
    """Lädt YAML-Konfiguration, falls PyYAML verfügbar ist."""
    if yaml is None:
        raise RuntimeError("PyYAML ist nicht installiert, aber --config wurde verwendet.")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Die YAML-Konfiguration muss ein Mapping / Dict sein.")
    return data


# ============================================================================
# Determinismus / Seeds / Worker-Seeds
# ============================================================================

def set_global_seed(seed: int, deterministic: bool, warn_only: bool, tf32: bool) -> None:
    """
    Setzt Seeds und Backend-Flags für reproduzierbare Läufe.

    Hinweis:
    Vollständiger Determinismus kann langsamer sein und bei einigen CUDA-Operationen
    Fehler auslösen, wenn keine deterministische Implementierung verfügbar ist.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=warn_only)
        except TypeError:
            # ältere PyTorch-Versionen kennen warn_only noch nicht
            torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32

    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def make_worker_init_fn(base_seed: int, rank: int):
    """
    Liefert worker_init_fn für DataLoader-Worker.

    Jeder Worker erhält einen deterministischen, rank-spezifischen Seed.
    """
    def _init_fn(worker_id: int) -> None:
        worker_seed = base_seed + rank * 10_000 + worker_id
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    return _init_fn


# ============================================================================
# CSV / Datenaufbereitung
# ============================================================================

@dataclass
class RawSample:
    """
    Ein rohes Trainingsbeispiel.

    - prompt: Kontext / Instruction / vorherige Turns
    - answer: Zieltext
    - length: optionale Schätzung für Sorting/Bucketing
    """
    prompt: str
    answer: str
    length: int = 0


def normalize_id(value: Any) -> str:
    """Normalisiert ID-Felder aus CSV-Dateien."""
    if value is None:
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def prepare_tokenizer_for_chat_sft(tokenizer) -> bool:
    """
    Ergänzt Pad-Token und Chat-Special-Tokens für das CSV-Format.

    Rückgabewert:
        True, wenn die Embeddings im Modell vergrößert werden müssen.
    """
    need_resize = False
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        need_resize = True

    added = tokenizer.add_tokens(
        ["<|System|>", "<|Benutzer|>", "<|Assistentin|>"],
        special_tokens=False,
    )
    if added > 0:
        need_resize = True

    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left"

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


def text_column_iter(csv_path: str, column_name: str) -> Iterator[RawSample]:
    """Liest Plain-Text-Beispiele aus einer frei wählbaren CSV-Spalte."""
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = (row.get(column_name) or "").strip()
            if text:
                yield RawSample(prompt="", answer=text)


def chat_block_iter(csv_path: str) -> Iterator[RawSample]:
    """
    Erzeugt pro Assistenten-Antwort ein einzelnes SFT-Trainingssample im Chat-Format.

    CSV-Spalten wie im bestehenden Skript:
        id,parent_id,system,Benutzer,Kontext,Assistentin
    """
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows: List[Dict[str, Any]] = []
        for row_idx, row in enumerate(reader):
            row = dict(row)
            row["_rowidx"] = row_idx
            row["id"] = normalize_id(row.get("id"))
            row["parent_id"] = normalize_id(row.get("parent_id"))
            rows.append(row)

    id2row = {row["id"]: row for row in rows if row.get("id")}
    candidates = [row for row in rows if (row.get("Assistentin") or "").strip() and row.get("id")]

    if not candidates:
        return

    from functools import lru_cache

    @lru_cache(maxsize=None)
    def _root_and_depth(row_id: str) -> Tuple[str, int]:
        depth = 0
        current = id2row.get(row_id)
        if current is None:
            return "", 0
        while True:
            parent_id = current.get("parent_id", "")
            if not parent_id or parent_id not in id2row:
                return current["id"], depth
            current = id2row[parent_id]
            depth += 1

    threads: Dict[str, List[Tuple[int, int, Dict[str, Any]]]] = {}
    for row in candidates:
        root_id, depth = _root_and_depth(row["id"])
        threads.setdefault(root_id, []).append((depth, int(row["_rowidx"]), row))

    for root_id in list(threads.keys()):
        threads[root_id].sort(key=lambda item: (item[0], item[1]))

    for root_id in threads:
        for _, _, target in threads[root_id]:
            chain: List[Dict[str, Any]] = []
            current = target
            seen = set()
            while current.get("id") and current["id"] not in seen:
                seen.add(current["id"])
                chain.append(current)
                parent_id = current.get("parent_id", "")
                if parent_id and parent_id in id2row:
                    current = id2row[parent_id]
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

            for idx in range(target_idx + 1):
                turn = chain[idx]
                user_text = (turn.get("Benutzer") or "").strip()
                context_text = (turn.get("Kontext") or "").strip()
                assistant_text = (turn.get("Assistentin") or "").strip()

                if user_text:
                    parts.append("<|Benutzer|>\n")
                    parts.append(f"{context_text}\n{user_text}".strip() if context_text else user_text)
                    parts.append("\n")

                if idx < target_idx and assistant_text:
                    parts.append("<|Assistentin|>\n")
                    parts.append(assistant_text)
                    parts.append("\n")
                elif idx == target_idx:
                    parts.append("<|Assistentin|>\n")

            prompt = "".join(parts)
            yield RawSample(prompt=prompt, answer=answer + "\n</s>")


def dialogplus_block_iter(csv_path: str) -> Iterator[RawSample]:
    """
    Erzeugt pro Assistenten-Antwort ein Sample im 'dialogplus'-Format.

    Dieses Format trennt Turns explizit mit </s>.
    """
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows: List[Dict[str, Any]] = []
        for row_idx, row in enumerate(reader):
            row = dict(row)
            row["_rowidx"] = row_idx
            row["id"] = normalize_id(row.get("id"))
            row["parent_id"] = normalize_id(row.get("parent_id"))
            rows.append(row)

    id2row = {row["id"]: row for row in rows if row.get("id")}
    candidates = [row for row in rows if (row.get("Assistentin") or "").strip() and row.get("id")]

    if not candidates:
        return

    from functools import lru_cache

    @lru_cache(maxsize=None)
    def _root_and_depth(row_id: str) -> Tuple[str, int]:
        depth = 0
        current = id2row.get(row_id)
        if current is None:
            return "", 0
        while True:
            parent_id = current.get("parent_id", "")
            if not parent_id or parent_id not in id2row:
                return current["id"], depth
            current = id2row[parent_id]
            depth += 1

    threads: Dict[str, List[Tuple[int, int, Dict[str, Any]]]] = {}
    for row in candidates:
        root_id, depth = _root_and_depth(row["id"])
        threads.setdefault(root_id, []).append((depth, int(row["_rowidx"]), row))

    for root_id in list(threads.keys()):
        threads[root_id].sort(key=lambda item: (item[0], item[1]))

    for root_id in threads:
        for _, _, target in threads[root_id]:
            chain: List[Dict[str, Any]] = []
            current = target
            seen = set()
            while current.get("id") and current["id"] not in seen:
                seen.add(current["id"])
                chain.append(current)
                parent_id = current.get("parent_id", "")
                if parent_id and parent_id in id2row:
                    current = id2row[parent_id]
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

            for idx in range(target_idx + 1):
                turn = chain[idx]
                user_text = (turn.get("Benutzer") or "").strip()
                context_text = (turn.get("Kontext") or "").strip()
                assistant_text = (turn.get("Assistentin") or "").strip()

                if user_text:
                    parts.append("\n<|Benutzer|>\n")
                    parts.append(f"{context_text}\n{user_text}".strip() if context_text else user_text)
                    parts.append("\n</s>")

                if idx < target_idx and assistant_text:
                    parts.append("\n<|Assistentin|>\n")
                    parts.append(assistant_text)
                    parts.append("\n</s>")
                elif idx == target_idx:
                    parts.append("\n<|Assistentin|>\n")

            prompt = "".join(parts)
            yield RawSample(prompt=prompt, answer=answer + "\n</s>")


def build_raw_samples(cfg: TrainConfig) -> List[RawSample]:
    """
    Baut alle Rohbeispiele aus der CSV-Datei.

    Hook-Point:
        Diese Funktion ist die zentrale Austauschstelle, wenn dein Projekt
        statt CSV z. B. JSONL, Parquet, WebDataset oder Streaming-Daten nutzt.
    """
    if cfg.template_mode == "text":
        samples = list(text_column_iter(cfg.train_csv, cfg.text_column))
    elif cfg.template_mode == "chat":
        samples = list(chat_block_iter(cfg.train_csv))
    elif cfg.template_mode == "dialogplus":
        samples = list(dialogplus_block_iter(cfg.train_csv))
    else:
        raise ValueError(f"Unbekannter template_mode: {cfg.template_mode}")

    if not samples:
        raise RuntimeError("Es wurden keine Trainingsbeispiele gefunden.")

    return samples


def split_samples(
    samples: Sequence[RawSample],
    val_split: float,
    seed: int,
) -> Tuple[List[RawSample], List[RawSample]]:
    """
    Deterministische Train/Val-Aufteilung.

    Für reproduzierbare Läufe wird dieselbe Seed-basierte Permutation in allen
    Prozessen verwendet.
    """
    samples = list(samples)
    if val_split <= 0.0 or len(samples) < 2:
        return samples, []

    rng = random.Random(seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)

    val_size = max(1, int(len(samples) * val_split))
    val_indices = set(indices[:val_size])

    train_samples = [samples[i] for i in range(len(samples)) if i not in val_indices]
    val_samples = [samples[i] for i in range(len(samples)) if i in val_indices]

    if not train_samples:
        raise RuntimeError("Train-Split ist leer. Reduziere val_split oder prüfe die Daten.")
    return train_samples, val_samples


class CausalLMDataset(Dataset):
    """
    Map-Style-Dataset für SFT / Causal-LM mit optionalem Prompt-Masking.

    Verhalten:
    - input_ids  = prompt_ids + answer_ids
    - labels     = [-100] * len(prompt_ids) + answer_ids
    - bei zu langen Sequenzen wird zuerst der Prompt von links gekürzt
    - ist auch die Antwort zu lang, wird sie auf max_seq_length begrenzt

    Das Dataset tokenisiert on-the-fly in __getitem__, damit:
    - weniger RAM verbraucht wird
    - Hook-Points für dynamische Augmentierung erhalten bleiben
    """
    def __init__(
        self,
        samples: Sequence[RawSample],
        tokenizer,
        max_seq_length: int,
        sort_by_length: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_seq_length = int(max_seq_length)
        self.samples: List[RawSample] = list(samples)

        if sort_by_length:
            for sample in self.samples:
                prompt_ids = tokenizer(sample.prompt, add_special_tokens=False)["input_ids"]
                answer_ids = tokenizer(sample.answer, add_special_tokens=False)["input_ids"]
                sample.length = len(prompt_ids) + len(answer_ids)
            self.samples.sort(key=lambda s: s.length)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[index]
        prompt_ids = self.tokenizer(sample.prompt, add_special_tokens=False)["input_ids"]
        answer_ids = self.tokenizer(sample.answer, add_special_tokens=False)["input_ids"]

        # Immer Antwort priorisieren. Zuerst Prompt kürzen, dann notfalls Antwort.
        total_len = len(prompt_ids) + len(answer_ids)
        if total_len > self.max_seq_length:
            overflow = total_len - self.max_seq_length
            if overflow > 0 and prompt_ids:
                prompt_ids = prompt_ids[min(len(prompt_ids), overflow):]
                total_len = len(prompt_ids) + len(answer_ids)

            if total_len > self.max_seq_length:
                # Extremfall: Antwort allein ist länger als das Fenster.
                answer_ids = answer_ids[-self.max_seq_length:]
                prompt_ids = []

        input_ids = prompt_ids + answer_ids
        labels = ([-100] * len(prompt_ids)) + answer_ids
        attention_mask = [1] * len(input_ids)

        if len(input_ids) == 0:
            raise RuntimeError("Leeres Sample nach Tokenisierung/Trunkierung gefunden.")

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class CausalLMDataCollator:
    """
    Pad- und Collate-Logik für variable Sequenzlängen.

    Best Practices:
    - Labels werden auf -100 gepadded
    - optional auf multiples von 8/16 padden für bessere Tensor-Core-Auslastung
    """
    def __init__(self, pad_token_id: int, pad_to_multiple_of: Optional[int] = 8) -> None:
        self.pad_token_id = int(pad_token_id)
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        max_len = max(int(item["input_ids"].numel()) for item in features)
        if self.pad_to_multiple_of and self.pad_to_multiple_of > 1:
            max_len = int(math.ceil(max_len / self.pad_to_multiple_of) * self.pad_to_multiple_of)

        input_ids_list: List[torch.Tensor] = []
        attention_masks_list: List[torch.Tensor] = []
        labels_list: List[torch.Tensor] = []

        for item in features:
            seq_len = int(item["input_ids"].numel())
            pad_len = max_len - seq_len

            input_ids = torch.nn.functional.pad(
                item["input_ids"],
                (0, pad_len),
                value=self.pad_token_id,
            )
            attention_mask = torch.nn.functional.pad(
                item["attention_mask"],
                (0, pad_len),
                value=0,
            )
            labels = torch.nn.functional.pad(
                item["labels"],
                (0, pad_len),
                value=-100,
            )

            input_ids_list.append(input_ids)
            attention_masks_list.append(attention_mask)
            labels_list.append(labels)

        return {
            "input_ids": torch.stack(input_ids_list, dim=0),
            "attention_mask": torch.stack(attention_masks_list, dim=0),
            "labels": torch.stack(labels_list, dim=0),
        }


# ============================================================================
# Modell / Optimizer / Scheduler
# ============================================================================

def build_model_and_tokenizer(cfg: TrainConfig, ctx: DistributedContext):
    """
    Baut Modell und Tokenizer.

    Hook-Points:
    - Wenn du ein komplett eigenes torch.nn.Module trainierst, ersetze diese
      Funktion durch deine Modell-Factory und gib denselben Rückgabe-Typ zurück.
    - Bei multimodalen Modellen kann hier auch ein eigener Processor gebaut werden.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name_or_path,
        trust_remote_code=cfg.trust_remote_code,
        use_fast=True,
    )
    need_resize = prepare_tokenizer_for_chat_sft(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name_or_path,
        trust_remote_code=cfg.trust_remote_code,
    )

    if need_resize or model.get_input_embeddings().num_embeddings != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))

    if cfg.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    model.config.use_cache = False
    model.to(ctx.device)

    if cfg.sync_batchnorm and ctx.is_distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if cfg.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model, mode=cfg.compile_mode)

    return model, tokenizer


def wrap_model_for_ddp(model: nn.Module, cfg: TrainConfig, ctx: DistributedContext) -> nn.Module:
    """Verpackt das Modell nur dann in DDP, wenn WORLD_SIZE > 1 ist."""
    if not ctx.is_distributed:
        return model

    ddp_kwargs: Dict[str, Any] = {
        "broadcast_buffers": cfg.broadcast_buffers,
        "find_unused_parameters": cfg.find_unused_parameters,
    }

    # Einige PyTorch-Versionen kennen static_graph noch nicht.
    if "static_graph" in DDP.__init__.__code__.co_varnames:
        ddp_kwargs["static_graph"] = cfg.static_graph
    if "gradient_as_bucket_view" in DDP.__init__.__code__.co_varnames:
        ddp_kwargs["gradient_as_bucket_view"] = True

    if ctx.device.type == "cuda":
        ddp_kwargs["device_ids"] = [ctx.local_rank]
        ddp_kwargs["output_device"] = ctx.local_rank

    model = DDP(model, **ddp_kwargs)
    return model


def build_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    """
    AdamW mit sauberer Decay/No-Decay-Gruppierung.

    LayerNorm- und Bias-Parameter sollten typischerweise nicht gewichtsmäßig
    decayed werden.
    """
    no_decay_terms = ("bias", "LayerNorm.weight", "layernorm.weight", "norm.weight", "ln_f.weight")
    named_params = list(unwrap_model(model).named_parameters())

    decay_params = [
        param for name, param in named_params
        if param.requires_grad and not any(term in name for term in no_decay_terms)
    ]
    no_decay_params = [
        param for name, param in named_params
        if param.requires_grad and any(term in name for term in no_decay_terms)
    ]

    parameter_groups = [
        {"params": decay_params, "weight_decay": cfg.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer = AdamW(
        parameter_groups,
        lr=cfg.learning_rate,
        betas=cfg.betas,
        eps=cfg.eps,
    )
    return optimizer


def compute_total_update_steps(
    train_loader: DataLoader,
    cfg: TrainConfig,
) -> int:
    """
    Bestimmt die Zahl der Optimizer-Updates über den gesamten Lauf.

    DDP-Hinweis:
    len(train_loader) ist pro Rank identisch, wenn DistributedSampler verwendet wird.
    """
    steps_per_epoch = math.ceil(len(train_loader) / cfg.gradient_accumulation_steps)
    if cfg.max_steps is not None:
        return int(cfg.max_steps)
    return int(cfg.num_epochs * steps_per_epoch)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_update_steps: int,
    cfg: TrainConfig,
) -> LambdaLR:
    """
    Erzeugt einen leichten LambdaLR-Scheduler ohne zusätzliche Abhängigkeiten.
    """
    warmup_steps = cfg.warmup_steps
    if warmup_steps <= 0 and cfg.warmup_ratio > 0:
        warmup_steps = int(total_update_steps * cfg.warmup_ratio)

    warmup_steps = max(0, min(warmup_steps, total_update_steps))
    min_lr_ratio = max(0.0, min(1.0, cfg.min_lr_ratio))

    def lr_lambda(current_step: int) -> float:
        if total_update_steps <= 0:
            return 1.0

        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))

        progress_num = current_step - warmup_steps
        progress_den = max(1, total_update_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress_num / progress_den))

        if cfg.scheduler == "constant":
            factor = 1.0
        elif cfg.scheduler == "linear":
            factor = 1.0 - progress
        elif cfg.scheduler == "cosine":
            factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        else:
            raise ValueError(f"Unbekannter Scheduler: {cfg.scheduler}")

        factor = min_lr_ratio + (1.0 - min_lr_ratio) * factor
        return max(min_lr_ratio, factor)

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def choose_amp_dtype(cfg: TrainConfig, device: torch.device) -> Optional[torch.dtype]:
    """
    Wählt den Autocast-Datentyp.

    Regeln:
    - CPU: kein AMP
    - CUDA + bf16: nur wenn unterstützt
    - CUDA + fp16: immer möglich
    """
    if device.type != "cuda":
        return None
    if cfg.mixed_precision == "none":
        return None
    if cfg.mixed_precision == "bf16":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        LOGGER.warning("bf16 angefordert, aber nicht unterstützt. Fallback auf fp16.")
        return torch.float16
    if cfg.mixed_precision == "fp16":
        return torch.float16
    return None


# ============================================================================
# Checkpointing / Resume / Early Stopping
# ============================================================================

@dataclass
class EarlyStoppingState:
    """Speichert den Zustand des Early-Stopping-Kriteriums."""
    best_metric: Optional[float] = None
    bad_epochs: int = 0

    def update(self, value: float, mode: str, min_delta: float) -> bool:
        """
        Aktualisiert den Zustand und liefert zurück, ob eine Verbesserung vorliegt.
        """
        if self.best_metric is None:
            self.best_metric = value
            self.bad_epochs = 0
            return True

        improved = False
        if mode == "min":
            improved = value < (self.best_metric - min_delta)
        elif mode == "max":
            improved = value > (self.best_metric + min_delta)
        else:
            raise ValueError(f"Unbekannter monitor_mode: {mode}")

        if improved:
            self.best_metric = value
            self.bad_epochs = 0
            return True

        self.bad_epochs += 1
        return False


def latest_checkpoint_path(checkpoints_dir: Path) -> Optional[Path]:
    """Findet den neuesten Resume-Punkt im Checkpoint-Verzeichnis."""
    last_ckpt = checkpoints_dir / "last.pt"
    if last_ckpt.exists():
        return last_ckpt

    candidates = sorted(checkpoints_dir.glob("epoch_*.pt"))
    if candidates:
        return candidates[-1]
    return None


def save_checkpoint(
    *,
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR],
    scaler: Optional[GradScaler],
    epoch: int,
    global_step: int,
    early_stopping: EarlyStoppingState,
    cfg: TrainConfig,
    tokenizer: Any,
    paths: ExperimentPaths,
) -> None:
    """
    Speichert einen vollständigen Resume-Checkpoint.

    Gespeichert werden:
    - Modellgewichte
    - Optimizer- / Scheduler- / Scaler-Zustand
    - Epoch / Global Step
    - Early-Stopping-Zustand
    - Konfiguration
    """
    state = {
        "model_state_dict": unwrap_model(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "global_step": global_step,
        "early_stopping": {
            "best_metric": early_stopping.best_metric,
            "bad_epochs": early_stopping.bad_epochs,
        },
        "config": asdict(cfg),
    }
    torch.save(state, path)

    # Zusätzliche Artefakte nur einmalig / best effort.
    tokenizer_dir = paths.artifacts_dir / "tokenizer"
    if not tokenizer_dir.exists():
        try:
            tokenizer.save_pretrained(tokenizer_dir)
        except Exception:
            pass

    hf_config_path = paths.artifacts_dir / "model_config.json"
    if not hf_config_path.exists():
        try:
            unwrap_model(model).config.to_json_file(hf_config_path)
        except Exception:
            pass


def prune_old_epoch_checkpoints(checkpoints_dir: Path, keep_last_k: int) -> None:
    """
    Löscht alte epoch_* Checkpoints, lässt aber last.pt / best.pt unangetastet.
    """
    epoch_checkpoints = sorted(checkpoints_dir.glob("epoch_*.pt"))
    if len(epoch_checkpoints) <= keep_last_k:
        return
    for path in epoch_checkpoints[:-keep_last_k]:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def load_checkpoint(
    *,
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR],
    scaler: Optional[GradScaler],
    device: torch.device,
) -> Dict[str, Any]:
    """
    Lädt einen Resume-Checkpoint und stellt Trainingszustand wieder her.
    """
    checkpoint = torch.load(path, map_location=device)
    unwrap_model(model).load_state_dict(checkpoint["model_state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    return checkpoint


# ============================================================================
# Training / Validierung
# ============================================================================

@dataclass
class EpochStats:
    """Aggregierte Kennzahlen einer Epoche."""
    loss: float
    target_tokens: int
    sequences: int
    steps: int
    elapsed_seconds: float


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    """Bewegt alle Batch-Tensoren effizient auf das Zielgerät."""
    return {
        key: value.to(device, non_blocking=(device.type == "cuda"))
        for key, value in batch.items()
    }


def reduce_epoch_stats(
    *,
    loss_token_sum: float,
    target_tokens: int,
    sequences: int,
    steps: int,
    elapsed_seconds: float,
    ctx: DistributedContext,
    device: torch.device,
) -> EpochStats:
    """
    Reduziert aggregierte Metriken korrekt über alle Prozesse.

    Für Causal-LM wird die Loss tokengewichtet gemittelt.
    """
    total_loss_token_sum = all_reduce_sum(loss_token_sum, ctx, device)
    total_target_tokens = int(all_reduce_sum(float(target_tokens), ctx, device))
    total_sequences = int(all_reduce_sum(float(sequences), ctx, device))
    total_steps = int(all_reduce_sum(float(steps), ctx, device))
    max_elapsed = elapsed_seconds
    if ctx.is_distributed:
        t = torch.tensor(elapsed_seconds, device=device, dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        max_elapsed = float(t.item())

    avg_loss = total_loss_token_sum / max(1, total_target_tokens)
    return EpochStats(
        loss=avg_loss,
        target_tokens=total_target_tokens,
        sequences=total_sequences,
        steps=total_steps,
        elapsed_seconds=max_elapsed,
    )


def train_one_epoch(
    *,
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR],
    scaler: Optional[GradScaler],
    amp_dtype: Optional[torch.dtype],
    epoch: int,
    global_step: int,
    cfg: TrainConfig,
    ctx: DistributedContext,
    writer: Optional[Any],
    metrics_path: Optional[Path],
) -> Tuple[EpochStats, int, bool]:
    """
    Führt genau eine Trainingsepoche aus.

    Rückgabe:
        epoch_stats, neuer_global_step, reached_max_steps
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)

    start_time = time.perf_counter()
    loss_token_sum = 0.0
    target_tokens = 0
    sequences = 0
    optimizer_steps_in_epoch = 0
    reached_max_steps = False

    log_window_loss_token_sum = 0.0
    log_window_target_tokens = 0
    log_window_sequences = 0
    log_window_start = time.perf_counter()

    autocast_context = (
        torch.autocast(device_type="cuda", dtype=amp_dtype)
        if (ctx.device.type == "cuda" and amp_dtype is not None)
        else nullcontext()
    )

    for batch_idx, batch in enumerate(train_loader):
        local_stop = SHUTDOWN.stop_requested
        if sync_stop_flag(local_stop, ctx, ctx.device):
            break

        batch = move_batch_to_device(batch, ctx.device)
        local_target_tokens = int((batch["labels"] != -100).sum().item())
        local_sequences = int(batch["input_ids"].size(0))

        with autocast_context:
            outputs = model(**batch)
            loss = outputs.loss

        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nicht-finite Loss erkannt: {float(loss.detach().item())}")

        detached_loss = float(loss.detach().item())
        loss_for_backward = loss / cfg.gradient_accumulation_steps

        if scaler is not None:
            scaler.scale(loss_for_backward).backward()
        else:
            loss_for_backward.backward()

        loss_token_sum += detached_loss * local_target_tokens
        target_tokens += local_target_tokens
        sequences += local_sequences

        log_window_loss_token_sum += detached_loss * local_target_tokens
        log_window_target_tokens += local_target_tokens
        log_window_sequences += local_sequences

        should_step = (
            ((batch_idx + 1) % cfg.gradient_accumulation_steps == 0)
            or (batch_idx + 1 == len(train_loader))
        )

        if not should_step:
            continue

        if scaler is not None:
            scaler.unscale_(optimizer)

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)

        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)

        if scheduler is not None:
            scheduler.step()

        global_step += 1
        optimizer_steps_in_epoch += 1

        should_log = (global_step % cfg.log_every_steps == 0)
        if should_log:
            reduced_loss_token_sum = all_reduce_sum(log_window_loss_token_sum, ctx, ctx.device)
            reduced_target_tokens = int(all_reduce_sum(float(log_window_target_tokens), ctx, ctx.device))
            reduced_sequences = int(all_reduce_sum(float(log_window_sequences), ctx, ctx.device))

            elapsed_window = max(1e-6, time.perf_counter() - log_window_start)
            avg_loss = reduced_loss_token_sum / max(1, reduced_target_tokens)
            lr = optimizer.param_groups[0]["lr"]
            tokens_per_sec = reduced_target_tokens / elapsed_window
            sequences_per_sec = reduced_sequences / elapsed_window
            grad_norm_value = float(grad_norm.item()) if torch.is_tensor(grad_norm) else float(grad_norm)

            if ctx.is_main_process:
                LOGGER.info(
                    "train | epoch=%d step=%d loss=%.6f lr=%.8f grad_norm=%.4f tok/s=%.2f seq/s=%.2f",
                    epoch,
                    global_step,
                    avg_loss,
                    lr,
                    grad_norm_value,
                    tokens_per_sec,
                    sequences_per_sec,
                )

                if writer is not None:
                    writer.add_scalar("train/loss", avg_loss, global_step)
                    writer.add_scalar("train/lr", lr, global_step)
                    writer.add_scalar("train/grad_norm", grad_norm_value, global_step)
                    writer.add_scalar("train/tokens_per_sec", tokens_per_sec, global_step)
                    writer.add_scalar("train/sequences_per_sec", sequences_per_sec, global_step)

                if metrics_path is not None:
                    append_jsonl(
                        metrics_path,
                        {
                            "phase": "train",
                            "epoch": epoch,
                            "global_step": global_step,
                            "loss": avg_loss,
                            "lr": lr,
                            "grad_norm": grad_norm_value,
                            "tokens_per_sec": tokens_per_sec,
                            "sequences_per_sec": sequences_per_sec,
                            "time": time.time(),
                        },
                    )

            log_window_loss_token_sum = 0.0
            log_window_target_tokens = 0
            log_window_sequences = 0
            log_window_start = time.perf_counter()

        if cfg.max_steps is not None and global_step >= cfg.max_steps:
            reached_max_steps = True
            break

    epoch_stats = reduce_epoch_stats(
        loss_token_sum=loss_token_sum,
        target_tokens=target_tokens,
        sequences=sequences,
        steps=optimizer_steps_in_epoch,
        elapsed_seconds=time.perf_counter() - start_time,
        ctx=ctx,
        device=ctx.device,
    )
    return epoch_stats, global_step, reached_max_steps


@torch.no_grad()
def validate_one_epoch(
    *,
    model: nn.Module,
    val_loader: DataLoader,
    epoch: int,
    global_step: int,
    cfg: TrainConfig,
    ctx: DistributedContext,
    writer: Optional[Any],
    metrics_path: Optional[Path],
) -> EpochStats:
    """
    Führt die Validierung für eine Epoche aus.
    """
    model.eval()

    start_time = time.perf_counter()
    loss_token_sum = 0.0
    target_tokens = 0
    sequences = 0
    steps = 0

    amp_dtype = choose_amp_dtype(cfg, ctx.device)
    autocast_context = (
        torch.autocast(device_type="cuda", dtype=amp_dtype)
        if (ctx.device.type == "cuda" and amp_dtype is not None)
        else nullcontext()
    )

    for batch in val_loader:
        local_stop = SHUTDOWN.stop_requested
        if sync_stop_flag(local_stop, ctx, ctx.device):
            break

        batch = move_batch_to_device(batch, ctx.device)

        with autocast_context:
            outputs = model(**batch)
            loss = outputs.loss

        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nicht-finite Validierungs-Loss erkannt: {float(loss.detach().item())}")

        local_target_tokens = int((batch["labels"] != -100).sum().item())
        local_sequences = int(batch["input_ids"].size(0))
        detached_loss = float(loss.detach().item())

        loss_token_sum += detached_loss * local_target_tokens
        target_tokens += local_target_tokens
        sequences += local_sequences
        steps += 1

    epoch_stats = reduce_epoch_stats(
        loss_token_sum=loss_token_sum,
        target_tokens=target_tokens,
        sequences=sequences,
        steps=steps,
        elapsed_seconds=time.perf_counter() - start_time,
        ctx=ctx,
        device=ctx.device,
    )

    if ctx.is_main_process:
        LOGGER.info(
            "valid | epoch=%d step=%d loss=%.6f tok=%d seq=%d time=%.2fs",
            epoch,
            global_step,
            epoch_stats.loss,
            epoch_stats.target_tokens,
            epoch_stats.sequences,
            epoch_stats.elapsed_seconds,
        )
        if writer is not None:
            writer.add_scalar("val/loss", epoch_stats.loss, global_step)

        if metrics_path is not None:
            append_jsonl(
                metrics_path,
                {
                    "phase": "val",
                    "epoch": epoch,
                    "global_step": global_step,
                    "loss": epoch_stats.loss,
                    "target_tokens": epoch_stats.target_tokens,
                    "sequences": epoch_stats.sequences,
                    "elapsed_seconds": epoch_stats.elapsed_seconds,
                    "time": time.time(),
                },
            )

    return epoch_stats


# ============================================================================
# DataLoader-Bau
# ============================================================================

def build_dataloaders(
    *,
    train_dataset: Dataset,
    val_dataset: Optional[Dataset],
    collator: CausalLMDataCollator,
    cfg: TrainConfig,
    ctx: DistributedContext,
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DistributedSampler], Optional[DistributedSampler]]:
    """
    Erzeugt DataLoader + DistributedSampler für Train und Validation.
    """
    common_loader_kwargs: Dict[str, Any] = {
        "batch_size": cfg.per_device_batch_size,
        "num_workers": cfg.num_workers,
        "pin_memory": bool(cfg.pin_memory and ctx.device.type == "cuda"),
        "collate_fn": collator,
        "worker_init_fn": make_worker_init_fn(cfg.seed, ctx.rank),
    }

    if cfg.num_workers > 0:
        common_loader_kwargs["persistent_workers"] = cfg.persistent_workers
        common_loader_kwargs["prefetch_factor"] = cfg.prefetch_factor

    train_sampler: Optional[DistributedSampler]
    val_sampler: Optional[DistributedSampler]

    if ctx.is_distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=ctx.world_size,
            rank=ctx.rank,
            shuffle=cfg.shuffle_train_examples,
            drop_last=False,
        )
        train_loader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            shuffle=False,
            **common_loader_kwargs,
        )

        if val_dataset is not None:
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=ctx.world_size,
                rank=ctx.rank,
                shuffle=False,
                drop_last=False,
            )
            val_loader = DataLoader(
                val_dataset,
                sampler=val_sampler,
                shuffle=False,
                **common_loader_kwargs,
            )
        else:
            val_sampler = None
            val_loader = None
    else:
        train_sampler = None
        train_loader = DataLoader(
            train_dataset,
            shuffle=cfg.shuffle_train_examples,
            **common_loader_kwargs,
        )

        if val_dataset is not None:
            val_sampler = None
            val_loader = DataLoader(
                val_dataset,
                shuffle=False,
                **common_loader_kwargs,
            )
        else:
            val_sampler = None
            val_loader = None

    return train_loader, val_loader, train_sampler, val_sampler


# ============================================================================
# Parsing / CLI
# ============================================================================

def build_parser() -> argparse.ArgumentParser:
    """Definiert die vollständige CLI."""
    parser = argparse.ArgumentParser(
        description="Modulares PyTorch-DDP-Trainingsframework für Causal-LM-SFT",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--config", type=str, default=None, help="Optionaler Pfad zu YAML-Datei.")

    # Eingaben
    parser.add_argument("--model_name_or_path", type=str, default=TrainConfig.model_name_or_path)
    parser.add_argument("--train_csv", type=str, default=TrainConfig.train_csv)
    parser.add_argument("--val_csv", type=str, default=TrainConfig.val_csv)
    parser.add_argument("--template_mode", type=str, default=TrainConfig.template_mode)
    parser.add_argument("--text_column", type=str, default=TrainConfig.text_column)

    # Ausgabe
    parser.add_argument("--experiment_name", type=str, default=TrainConfig.experiment_name)
    parser.add_argument("--output_root", type=str, default=TrainConfig.output_root)

    # Splits
    parser.add_argument("--val_split", type=float, default=TrainConfig.val_split)
    parser.add_argument("--split_seed", type=int, default=TrainConfig.split_seed)

    # Optimierung
    parser.add_argument("--learning_rate", type=float, default=TrainConfig.learning_rate)
    parser.add_argument("--weight_decay", type=float, default=TrainConfig.weight_decay)
    parser.add_argument("--beta1", type=float, default=TrainConfig.betas[0])
    parser.add_argument("--beta2", type=float, default=TrainConfig.betas[1])
    parser.add_argument("--eps", type=float, default=TrainConfig.eps)
    parser.add_argument("--num_epochs", type=int, default=TrainConfig.num_epochs)
    parser.add_argument("--max_steps", type=int, default=TrainConfig.max_steps)
    parser.add_argument("--per_device_batch_size", type=int, default=TrainConfig.per_device_batch_size)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=TrainConfig.gradient_accumulation_steps)
    parser.add_argument("--max_grad_norm", type=float, default=TrainConfig.max_grad_norm)

    # Scheduler
    parser.add_argument("--scheduler", type=str, default=TrainConfig.scheduler)
    parser.add_argument("--warmup_steps", type=int, default=TrainConfig.warmup_steps)
    parser.add_argument("--warmup_ratio", type=float, default=TrainConfig.warmup_ratio)
    parser.add_argument("--min_lr_ratio", type=float, default=TrainConfig.min_lr_ratio)

    # Sequenzen / Loader
    parser.add_argument("--max_seq_length", type=int, default=TrainConfig.max_seq_length)
    parser.add_argument("--sort_by_length", action=argparse.BooleanOptionalAction, default=TrainConfig.sort_by_length)
    parser.add_argument("--shuffle_train_examples", action=argparse.BooleanOptionalAction, default=TrainConfig.shuffle_train_examples)
    parser.add_argument("--num_workers", type=int, default=TrainConfig.num_workers)
    parser.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=TrainConfig.pin_memory)
    parser.add_argument("--persistent_workers", action=argparse.BooleanOptionalAction, default=TrainConfig.persistent_workers)
    parser.add_argument("--prefetch_factor", type=int, default=TrainConfig.prefetch_factor)
    parser.add_argument("--pad_to_multiple_of", type=int, default=TrainConfig.pad_to_multiple_of)

    # Precision / Performance
    parser.add_argument("--mixed_precision", type=str, default=TrainConfig.mixed_precision)
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=TrainConfig.gradient_checkpointing)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=TrainConfig.tf32)
    parser.add_argument("--compile_model", action=argparse.BooleanOptionalAction, default=TrainConfig.compile_model)
    parser.add_argument("--compile_mode", type=str, default=TrainConfig.compile_mode)

    # DDP
    parser.add_argument("--ddp_backend", type=str, default=TrainConfig.ddp_backend)
    parser.add_argument("--ddp_timeout_minutes", type=int, default=TrainConfig.ddp_timeout_minutes)
    parser.add_argument("--broadcast_buffers", action=argparse.BooleanOptionalAction, default=TrainConfig.broadcast_buffers)
    parser.add_argument("--find_unused_parameters", action=argparse.BooleanOptionalAction, default=TrainConfig.find_unused_parameters)
    parser.add_argument("--static_graph", action=argparse.BooleanOptionalAction, default=TrainConfig.static_graph)
    parser.add_argument("--sync_batchnorm", action=argparse.BooleanOptionalAction, default=TrainConfig.sync_batchnorm)

    # Stabilität
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=TrainConfig.deterministic)
    parser.add_argument("--deterministic_warn_only", action=argparse.BooleanOptionalAction, default=TrainConfig.deterministic_warn_only)

    # Validierung / Early Stopping
    parser.add_argument("--validate_every_epoch", action=argparse.BooleanOptionalAction, default=TrainConfig.validate_every_epoch)
    parser.add_argument("--early_stopping_patience", type=int, default=TrainConfig.early_stopping_patience)
    parser.add_argument("--early_stopping_min_delta", type=float, default=TrainConfig.early_stopping_min_delta)
    parser.add_argument("--monitor_metric", type=str, default=TrainConfig.monitor_metric)
    parser.add_argument("--monitor_mode", type=str, default=TrainConfig.monitor_mode)

    # Logging
    parser.add_argument("--log_every_steps", type=int, default=TrainConfig.log_every_steps)
    parser.add_argument("--use_tensorboard", action=argparse.BooleanOptionalAction, default=TrainConfig.use_tensorboard)
    parser.add_argument("--save_every_epoch", action=argparse.BooleanOptionalAction, default=TrainConfig.save_every_epoch)
    parser.add_argument("--keep_last_k_checkpoints", type=int, default=TrainConfig.keep_last_k_checkpoints)

    # Resume
    parser.add_argument("--resume", type=str, default=TrainConfig.resume)

    # Modell-Hooks
    parser.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=TrainConfig.trust_remote_code)

    return parser


def namespace_to_config(args: argparse.Namespace) -> TrainConfig:
    """Konvertiert argparse Namespace in die dataclass-basierte Konfiguration."""
    cfg = TrainConfig(
        model_name_or_path=args.model_name_or_path,
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        template_mode=args.template_mode,
        text_column=args.text_column,
        experiment_name=args.experiment_name,
        output_root=args.output_root,
        val_split=args.val_split,
        split_seed=args.split_seed,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        num_epochs=args.num_epochs,
        max_steps=args.max_steps,
        per_device_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        scheduler=args.scheduler,
        warmup_steps=args.warmup_steps,
        warmup_ratio=args.warmup_ratio,
        min_lr_ratio=args.min_lr_ratio,
        max_seq_length=args.max_seq_length,
        sort_by_length=args.sort_by_length,
        shuffle_train_examples=args.shuffle_train_examples,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
        pad_to_multiple_of=args.pad_to_multiple_of,
        mixed_precision=args.mixed_precision,
        gradient_checkpointing=args.gradient_checkpointing,
        tf32=args.tf32,
        compile_model=args.compile_model,
        compile_mode=args.compile_mode,
        ddp_backend=args.ddp_backend,
        ddp_timeout_minutes=args.ddp_timeout_minutes,
        broadcast_buffers=args.broadcast_buffers,
        find_unused_parameters=args.find_unused_parameters,
        static_graph=args.static_graph,
        sync_batchnorm=args.sync_batchnorm,
        seed=args.seed,
        deterministic=args.deterministic,
        deterministic_warn_only=args.deterministic_warn_only,
        validate_every_epoch=args.validate_every_epoch,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        monitor_metric=args.monitor_metric,
        monitor_mode=args.monitor_mode,
        log_every_steps=args.log_every_steps,
        use_tensorboard=args.use_tensorboard,
        save_every_epoch=args.save_every_epoch,
        keep_last_k_checkpoints=args.keep_last_k_checkpoints,
        resume=args.resume,
        trust_remote_code=args.trust_remote_code,
    )
    cfg.validate()
    return cfg


def parse_config() -> TrainConfig:
    """
    Zweiphasiges Parsing:
    1. nur --config lesen
    2. YAML-Defaults setzen
    3. komplette CLI parsen, wobei CLI YAML überschreibt
    """
    parser = build_parser()
    pre_args, _ = parser.parse_known_args()

    if pre_args.config:
        yaml_data = load_yaml_config(pre_args.config)
        valid_keys = set(vars(parser.parse_args([])).keys())
        unknown_keys = sorted(set(yaml_data.keys()) - valid_keys)
        if unknown_keys:
            raise ValueError(f"Unbekannte Schlüssel in YAML-Konfiguration: {unknown_keys}")
        parser.set_defaults(**yaml_data)

    args = parser.parse_args()
    cfg = namespace_to_config(args)
    return cfg


# ============================================================================
# Hauptablauf
# ============================================================================

@record
def run() -> None:
    """
    Hauptfunktion des Frameworks.

    Der Kontrollfluss ist absichtlich linear gehalten:
    - parse config
    - init DDP
    - init Verzeichnisse / Logging
    - build data / model
    - optional resume
    - train / validate / checkpoint
    - cleanup
    """
    register_signal_handlers()
    cfg = parse_config()

    ctx = init_distributed(cfg)

    resolved_resume_path = resolve_resume_request(cfg) if ctx.is_main_process else None
    resolved_resume_path = broadcast_object(
        str(resolved_resume_path) if resolved_resume_path is not None else None,
        ctx,
    )
    resolved_resume_path = Path(resolved_resume_path) if resolved_resume_path else None

    paths = ensure_experiment_paths(cfg, ctx, resolved_resume_path=resolved_resume_path)
    setup_logging(paths, ctx)
    save_resolved_config(cfg, paths, ctx)

    writer = None
    if ctx.is_main_process and cfg.use_tensorboard and SummaryWriter is not None:
        writer = SummaryWriter(log_dir=str(paths.tensorboard_dir))

    metrics_path = paths.logs_dir / "metrics.jsonl" if ctx.is_main_process else None

    try:
        set_global_seed(
            seed=cfg.seed,
            deterministic=cfg.deterministic,
            warn_only=cfg.deterministic_warn_only,
            tf32=cfg.tf32,
        )

        rank0_log(ctx, f"Starte Lauf in: {paths.run_dir}")
        rank0_log(ctx, f"Gerät: {ctx.device}")
        rank0_log(ctx, f"Distributed: {ctx.is_distributed} | world_size={ctx.world_size}")
        rank0_log(ctx, f"Konfiguration: {json.dumps(asdict(cfg), ensure_ascii=False)}")

        model, tokenizer = build_model_and_tokenizer(cfg, ctx)
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            raise RuntimeError("Tokenizer besitzt keinen pad_token_id. Prüfe prepare_tokenizer_for_chat_sft().")

        # Daten bauen
        if cfg.val_csv:
            train_samples = build_raw_samples(cfg)
            original_train_csv = cfg.train_csv
            cfg_for_val = TrainConfig(**asdict(cfg))
            cfg_for_val.train_csv = cfg.val_csv
            val_samples = build_raw_samples(cfg_for_val)
            cfg.train_csv = original_train_csv
        else:
            all_samples = build_raw_samples(cfg)
            train_samples, val_samples = split_samples(all_samples, cfg.val_split, cfg.split_seed)

        rank0_log(ctx, f"Train-Samples: {len(train_samples)} | Val-Samples: {len(val_samples)}")

        train_dataset = CausalLMDataset(
            train_samples,
            tokenizer=tokenizer,
            max_seq_length=cfg.max_seq_length,
            sort_by_length=cfg.sort_by_length,
        )
        val_dataset = (
            CausalLMDataset(
                val_samples,
                tokenizer=tokenizer,
                max_seq_length=cfg.max_seq_length,
                sort_by_length=False,
            )
            if val_samples
            else None
        )

        collator = CausalLMDataCollator(
            pad_token_id=pad_token_id,
            pad_to_multiple_of=cfg.pad_to_multiple_of,
        )

        train_loader, val_loader, train_sampler, val_sampler = build_dataloaders(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collator=collator,
            cfg=cfg,
            ctx=ctx,
        )

        rank0_log(
            ctx,
            f"Dataloader: train_batches={len(train_loader)} | "
            f"val_batches={len(val_loader) if val_loader is not None else 0}",
        )
        if val_loader is None:
            rank0_log(ctx, "Hinweis: Keine Validierungsdaten vorhanden. Early stopping basiert dann nicht auf val_loss.")

        optimizer = build_optimizer(model, cfg)
        total_update_steps = compute_total_update_steps(train_loader, cfg)
        scheduler = build_scheduler(optimizer, total_update_steps, cfg)

        amp_dtype = choose_amp_dtype(cfg, ctx.device)
        scaler = GradScaler(enabled=(ctx.device.type == "cuda" and amp_dtype == torch.float16))

        model = wrap_model_for_ddp(model, cfg, ctx)

        # Resume
        early_stopping = EarlyStoppingState()
        start_epoch = 0
        global_step = 0

        resume_path = resolved_resume_path

        if resume_path:
            if resume_path.exists():
                rank0_log(ctx, f"Lade Resume-Checkpoint: {resume_path}")
                checkpoint = load_checkpoint(
                    path=resume_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    device=ctx.device,
                )
                start_epoch = int(checkpoint.get("epoch", -1)) + 1
                global_step = int(checkpoint.get("global_step", 0))
                early_info = checkpoint.get("early_stopping", {}) or {}
                early_stopping.best_metric = early_info.get("best_metric")
                early_stopping.bad_epochs = int(early_info.get("bad_epochs", 0))
                rank0_log(
                    ctx,
                    f"Resume erfolgreich: start_epoch={start_epoch} | global_step={global_step} "
                    f"| best_metric={early_stopping.best_metric}",
                )
            else:
                rank0_log(ctx, f"Resume-Pfad existiert nicht, Training startet frisch: {resume_path}")

        barrier(ctx)

        # Training
        for epoch in range(start_epoch, cfg.num_epochs):
            if sync_stop_flag(SHUTDOWN.stop_requested, ctx, ctx.device):
                break

            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            train_stats, global_step, reached_max_steps = train_one_epoch(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                amp_dtype=amp_dtype,
                epoch=epoch,
                global_step=global_step,
                cfg=cfg,
                ctx=ctx,
                writer=writer,
                metrics_path=metrics_path,
            )

            if ctx.is_main_process:
                LOGGER.info(
                    "epoch_end | train | epoch=%d loss=%.6f tok=%d seq=%d updates=%d time=%.2fs",
                    epoch,
                    train_stats.loss,
                    train_stats.target_tokens,
                    train_stats.sequences,
                    train_stats.steps,
                    train_stats.elapsed_seconds,
                )
                if writer is not None:
                    writer.add_scalar("epoch/train_loss", train_stats.loss, epoch)
                    writer.add_scalar("epoch/train_target_tokens", train_stats.target_tokens, epoch)
                    writer.add_scalar("epoch/train_sequences", train_stats.sequences, epoch)

            current_metric: Optional[float] = None
            if val_loader is not None and cfg.validate_every_epoch:
                if val_sampler is not None:
                    val_sampler.set_epoch(epoch)
                val_stats = validate_one_epoch(
                    model=model,
                    val_loader=val_loader,
                    epoch=epoch,
                    global_step=global_step,
                    cfg=cfg,
                    ctx=ctx,
                    writer=writer,
                    metrics_path=metrics_path,
                )
                current_metric = val_stats.loss
            else:
                val_stats = None

            # Checkpointing nur auf Rank 0
            improved = False
            if ctx.is_main_process:
                if current_metric is not None:
                    improved = early_stopping.update(
                        current_metric,
                        mode=cfg.monitor_mode,
                        min_delta=cfg.early_stopping_min_delta,
                    )
                else:
                    # Ohne Validierung tracken wir Training-Loss als Fallback.
                    improved = early_stopping.update(
                        train_stats.loss,
                        mode="min",
                        min_delta=0.0,
                    )

                # last.pt immer aktualisieren
                save_checkpoint(
                    path=paths.checkpoints_dir / "last.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    global_step=global_step,
                    early_stopping=early_stopping,
                    cfg=cfg,
                    tokenizer=tokenizer,
                    paths=paths,
                )

                # Optional epoche-spezifische Snapshots
                if cfg.save_every_epoch:
                    epoch_path = paths.checkpoints_dir / f"epoch_{epoch:04d}_step_{global_step:08d}.pt"
                    save_checkpoint(
                        path=epoch_path,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        epoch=epoch,
                        global_step=global_step,
                        early_stopping=early_stopping,
                        cfg=cfg,
                        tokenizer=tokenizer,
                        paths=paths,
                    )
                    prune_old_epoch_checkpoints(paths.checkpoints_dir, cfg.keep_last_k_checkpoints)

                if improved:
                    shutil.copy2(paths.checkpoints_dir / "last.pt", paths.checkpoints_dir / "best.pt")
                    LOGGER.info(
                        "checkpoint | best aktualisiert | metric=%s value=%.6f",
                        cfg.monitor_metric,
                        current_metric if current_metric is not None else train_stats.loss,
                    )

                if writer is not None and early_stopping.best_metric is not None:
                    writer.add_scalar("early_stopping/best_metric", early_stopping.best_metric, epoch)
                    writer.add_scalar("early_stopping/bad_epochs", early_stopping.bad_epochs, epoch)

                if metrics_path is not None:
                    append_jsonl(
                        metrics_path,
                        {
                            "phase": "epoch_summary",
                            "epoch": epoch,
                            "global_step": global_step,
                            "train_loss": train_stats.loss,
                            "val_loss": current_metric,
                            "best_metric": early_stopping.best_metric,
                            "bad_epochs": early_stopping.bad_epochs,
                            "time": time.time(),
                        },
                    )

            barrier(ctx)

            should_stop_early = (
                val_loader is not None
                and early_stopping.bad_epochs >= cfg.early_stopping_patience
                if ctx.is_main_process
                else False
            )
            should_stop_early = sync_stop_flag(should_stop_early, ctx, ctx.device)
            if should_stop_early:
                rank0_log(ctx, "Early stopping ausgelöst.")
                break

            if reached_max_steps:
                rank0_log(ctx, f"max_steps={cfg.max_steps} erreicht.")
                break

            if sync_stop_flag(SHUTDOWN.stop_requested, ctx, ctx.device):
                rank0_log(ctx, f"Kontrollierter Abbruch wegen Signal: {SHUTDOWN.reason}")
                break

        if ctx.is_main_process:
            final_model_dir = paths.artifacts_dir / "final_model"
            try:
                final_model_dir.mkdir(parents=True, exist_ok=True)
                unwrapped = unwrap_model(model)
                if hasattr(unwrapped, "config"):
                    unwrapped.config.use_cache = True
                if hasattr(unwrapped, "save_pretrained"):
                    unwrapped.save_pretrained(final_model_dir)
                    tokenizer.save_pretrained(final_model_dir)
                    rank0_log(ctx, f"Finales HF-Modell exportiert nach: {final_model_dir}")
            except Exception as export_exc:
                LOGGER.error("Finaler Modell-Export fehlgeschlagen: %s", export_exc)

        barrier(ctx)
        rank0_log(ctx, "Training erfolgreich beendet.")

    except KeyboardInterrupt:
        LOGGER.error("KeyboardInterrupt erhalten. Training wird abgebrochen.")
        abort_distributed()
        raise
    except Exception as exc:
        LOGGER.error("Fataler Fehler: %s", exc)
        LOGGER.error(traceback.format_exc())
        abort_distributed()
        raise
    finally:
        try:
            if writer is not None:
                writer.flush()
                writer.close()
        except Exception:
            pass
        cleanup_distributed()


def main() -> None:
    """Programm-Einstiegspunkt."""
    run()


if __name__ == "__main__":
    main()
