#!/usr/bin/env python3
# matelix_lab_server.py
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
import json
import os
import platform
import random
import re
import shutil
import subprocess
import threading
import time
import traceback
import uuid
from collections import Counter, deque
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union

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
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator
from tokenizers import AddedToken

# ----------------------------
# Environment
# ----------------------------
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

# ----------------------------
# Globale Settings
# ----------------------------
csv.field_size_limit(1024 * 1024 * 128)

BASE_DIR = Path(__file__).resolve().parent
TRAINING_OUT_DIR = BASE_DIR / "training_outputs"
DATASETS_DIR = BASE_DIR / "datasets"
STATIC_DIR = BASE_DIR / "static"

TRAINING_OUT_DIR.mkdir(parents=True, exist_ok=True)
DATASETS_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

HF_TRUST_REMOTE_CODE = False
OPENAI_COMPAT_API_KEY = "matelix-local-dev-key"

# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI(title="MaTeLiX AI Lab", version="5.0-chunked-dataset")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/training_outputs", StaticFiles(directory=str(TRAINING_OUT_DIR)), name="training_outputs")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ----------------------------
# Minimal UI
# ----------------------------
DEFAULT_INDEX_HTML = """<!DOCTYPE html>
<html lang="de">
<head>
  <meta charset="utf-8"/>
  <title>MaTeLiX LAB</title>
  <style>
    body { margin:0; background:#07110c; color:#c8ffdd; font-family:Arial,sans-serif;
           display:flex; align-items:center; justify-content:center; height:100vh; }
    .card { background:#0f1f17; border:1px solid #28ff96; border-radius:16px;
            padding:2rem 2.5rem; max-width:520px; text-align:center; }
    h1 { margin:0 0 0.5rem 0; color:#28ff96; letter-spacing:0.08em; font-size:1.2rem; }
    code { background:#06140e; border-radius:6px; padding:0.2rem 0.4rem; border:1px solid #20ff83; }
  </style>
</head>
<body>
  <div class="card">
    <h1>MaTeLiX LAB Backend läuft</h1>
    <p>Lege deine UI unter <code>./static/index.html</code> ab.</p>
  </div>
</body>
</html>
"""


def _ensure_index_html() -> Path:
    p = STATIC_DIR / "index.html"
    if not p.exists():
        p.write_text(DEFAULT_INDEX_HTML, encoding="utf-8")
    return p


# ----------------------------
# Thread-safe LogStore
# ----------------------------
class LogStore:
    def __init__(self, max_lines: int = 4000):
        self.max_lines = int(max_lines)
        self._lock = threading.Lock()
        self._base_id = 0
        self._lines: deque[str] = deque()
        self._file_path: Optional[str] = None

    def set_file(self, path: Optional[str]) -> None:
        with self._lock:
            self._file_path = path
            if path:
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                Path(path).touch(exist_ok=True)

    def clear(self) -> None:
        with self._lock:
            self._base_id = 0
            self._lines.clear()

    def append(self, msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        with self._lock:
            if self._file_path:
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

    def since(self, last_seen_id: int) -> Tuple[List[str], int]:
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


# ----------------------------
# RoPE / YaRN config cleanup
# ----------------------------
_ALLOWED_YARN_KEYS = {
    "type",
    "factor",
    "original_max_position_embeddings",
    "low_freq_factor",
    "high_freq_factor",
    "finetuned",
}


def _normalize_and_clean_rope(d: Dict[str, Any]) -> Dict[str, Any]:
    d = dict(d or {})
    rp = d.get("rope_parameters")
    rs = d.get("rope_scaling")

    if rp and not rs:
        rs = {
            "type": rp.get("rope_type") or rp.get("type") or "yarn",
            "factor": rp.get("factor"),
            "original_max_position_embeddings": rp.get("original_max_position_embeddings"),
        }
        d["rope_scaling"] = {k: v for k, v in rs.items() if v is not None}

    if isinstance(d.get("rope_scaling"), dict):
        clean: Dict[str, Any] = {}
        for k, v in d["rope_scaling"].items():
            if k in _ALLOWED_YARN_KEYS:
                clean[k] = v
        clean["type"] = (clean.get("type") or "yarn").lower()
        if clean.get("factor", None) is None:
            clean.pop("factor", None)
        if clean.get("original_max_position_embeddings", None) is None:
            clean.pop("original_max_position_embeddings", None)
        d["rope_scaling"] = clean

    d.pop("rope_parameters", None)

    def _strip_truncate(obj: Any) -> None:
        if isinstance(obj, dict):
            obj.pop("truncate", None)
            for v in obj.values():
                _strip_truncate(v)
        elif isinstance(obj, list):
            for it in obj:
                _strip_truncate(it)

    _strip_truncate(d)
    return d


def patch_local_config_json_if_exists(model_dir: str) -> None:
    from transformers import AutoConfig

    cfg_path = Path(model_dir) / "config.json"
    if not cfg_path.exists():
        return
    try:
        raw = AutoConfig.from_pretrained(model_dir, trust_remote_code=HF_TRUST_REMOTE_CODE)
        d = _normalize_and_clean_rope(raw.to_dict())
        cfg_path.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def load_clean_config(model_dir: str):
    from transformers import AutoConfig

    raw = AutoConfig.from_pretrained(model_dir, trust_remote_code=HF_TRUST_REMOTE_CODE)
    d = _normalize_and_clean_rope(raw.to_dict())
    cls = raw.__class__
    return cls(**d)


# ----------------------------
# Hardware helpers
# ----------------------------
def get_hardware_info() -> Dict[str, Any]:
    try:
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
    except Exception:
        return {"cuda": False, "mps": False, "num_cuda": 0, "gpus": [], "num_cpus": os.cpu_count() or 1}


def _query_nvidia_smi() -> List[Dict[str, Any]]:
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
        lines = out.decode("utf-8", errors="ignore").strip().split("\n")
        infos: List[Dict[str, Any]] = []
        for idx, line in enumerate(lines):
            util, mem_total, mem_used, name = [s.strip() for s in line.split(",")]
            infos.append({"id": idx, "name": name, "util": int(util), "mem_total": int(mem_total), "mem_used": int(mem_used)})
        return infos
    except Exception:
        return []


def get_system_status() -> Dict[str, Any]:
    mem = psutil.virtual_memory()
    cpu = psutil.cpu_percent(interval=None)
    gpu_infos = _query_nvidia_smi()
    primary = gpu_infos[0] if gpu_infos else None
    num_cuda = torch.cuda.device_count() if torch.cuda.is_available() else 0
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
        "num_cuda": num_cuda,
    }


# ----------------------------
# Token / N-Gram helpers
# ----------------------------
def is_code_ngram(text: str) -> bool:
    code_keywords = ["function", "for", "while", "if", "else", "elif", "def", "class", "return", "const", "let", "var", "try", "catch", "=>"]
    code_chars = set("(){}[]=;.,<>:+-*/%\"'|\\&!^#@")
    if any(word in text for word in code_keywords):
        return True
    if sum(1 for c in text if c in code_chars) >= 2:
        return True
    if re.search(r"[a-zA-Z0-9_]+\s*=\s*[a-zA-Z0-9_\[\]\"']+", text):
        return True
    if re.match(r".+\(.*\)", text):
        return True
    return False


def is_probably_code_text(t: str) -> bool:
    if not t:
        return False
    if "```" in t:
        return True
    code_chars = set("(){}[]=;.,<>:+-*/%\"'|\\&!^#@")
    ratio = sum(1 for c in t if c in code_chars) / max(1, len(t))
    if ratio > 0.07:
        return True
    kw = ("def ", "class ", "function ", "=>", "import ", "from ", "return ", "if (", "for (", "while (", "try:", "except:", "catch (")
    return any(k in t for k in kw)


def decode_ngram_text(tokenizer, ids: Tuple[int, ...]) -> str:
    try:
        s = tokenizer.decode(list(ids), clean_up_tokenization_spaces=False)
    except TypeError:
        s = tokenizer.decode(list(ids))
    return (s or "").rstrip()


def find_frequent_ngrams_diverse(
    tokenizer,
    texts: Iterable[str],
    max_ngram: int = 8,
    top_k: int = 1500,
    min_chars: int = 32,
    min_words: int = 4,
    code_boost: float = 2.0,
    similarity_thresh: float = 0.89,
    preselect_factor: int = 5,
    code_phrases_extra: Optional[List[str]] = None,
    min_count: int = 2,
    max_tokens_per_text: int = 4096,
    max_token_chars: int = 384,
    max_ngram_code: Optional[int] = None,
) -> List[Tuple[int, ...]]:
    import heapq

    special_tokens = {"<|System|>", "<|Benutzer|>", "<|Assistentin|>", "<s>", "</s>", "<pad>", "<unk>", "<|pad|>"}
    if getattr(tokenizer, "additional_special_tokens", None):
        special_tokens |= set(tokenizer.additional_special_tokens or [])
    if getattr(tokenizer, "pad_token", None):
        special_tokens.add(tokenizer.pad_token)

    def sim_tokens(s: str, is_code: bool) -> set:
        if is_code:
            return set(re.findall(r"[A-Za-z_]\w*|\d+|==|!=|<=|>=|->|=>|[{}()\[\];,.:=+\-*/%<>]", s))
        return set(s.lower().replace(".", "").split())

    def similarity(a: str, b: str, is_code: bool) -> float:
        A = sim_tokens(a, is_code)
        B = sim_tokens(b, is_code)
        if not A or not B:
            return 0.0
        return len(A & B) / len(A | B)

    code_sizes = (8, 12, 16, 24, 32, 48, 64)
    max_ngram_text = int(max_ngram)
    max_ngram_code_eff = int(max_ngram_code) if isinstance(max_ngram_code, int) else max(max_ngram_text, 64)

    c = Counter()
    for t in texts:
        ids = tokenizer(t, add_special_tokens=False)["input_ids"]
        if not ids:
            continue
        if len(ids) > int(max_tokens_per_text):
            ids = ids[: int(max_tokens_per_text)]
        L = len(ids)
        if L < 2:
            continue
        code_heavy = is_probably_code_text(t)
        if code_heavy:
            lengths = [n for n in code_sizes if 2 <= n <= L and n <= max_ngram_code_eff]
            lengths += list(range(2, min(max_ngram_text, L) + 1))
            lengths = sorted(set(lengths))
        else:
            lengths = list(range(2, min(max_ngram_text, L) + 1))

        for n in lengths:
            step = 1
            if code_heavy and n >= 16:
                step = max(1, n // 8)
            for i in range(0, L - n + 1, step):
                c[tuple(ids[i : i + n])] += 1

    if not c:
        return []

    M = max(1, int(top_k) * max(1, int(preselect_factor)))
    by_count = c.most_common(M)
    by_gain = heapq.nlargest(M, c.items(), key=lambda kv: kv[1] * max(1, (len(kv[0]) - 1)))

    cand_map: Dict[Tuple[int, ...], int] = {}
    for ng, cnt in list(by_count) + list(by_gain):
        if int(cnt) >= int(min_count):
            cand_map[ng] = max(int(cnt), cand_map.get(ng, 0))

    scored_code: List[Tuple[Tuple[int, ...], str, float]] = []
    scored_text: List[Tuple[Tuple[int, ...], str, float]] = []

    for ngram, count in cand_map.items():
        klartext = decode_ngram_text(tokenizer, ngram)
        if not klartext:
            continue
        if len(klartext) > int(max_token_chars):
            continue
        if any(tok in klartext for tok in special_tokens):
            continue
        code_like = is_code_ngram(klartext)
        if code_like:
            if len(klartext) < max(8, int(min_chars // 2)):
                continue
        else:
            if len(klartext) < int(min_chars):
                continue
            if klartext.count(" ") < (int(min_words) - 1):
                continue
        gain = float(count) * float(max(1, (len(ngram) - 1)))
        score = gain * (float(code_boost) if code_like else 1.0)
        (scored_code if code_like else scored_text).append((ngram, klartext, score))

    scored_code.sort(key=lambda x: x[2], reverse=True)
    scored_text.sort(key=lambda x: x[2], reverse=True)

    diverse: List[Tuple[int, ...]] = []
    chosen_code: List[str] = []
    chosen_text: List[str] = []

    def accept(candidate: str, pool: List[str], is_code: bool) -> bool:
        for t in pool:
            if similarity(candidate, t, is_code) > float(similarity_thresh):
                return False
            if candidate.rstrip(".").strip() == t.rstrip(".").strip():
                return False
            if candidate in t or t in candidate:
                return False
        return True

    for ng, txt, _ in scored_code:
        if accept(txt, chosen_code, True):
            diverse.append(ng)
            chosen_code.append(txt)
        if len(diverse) >= int(top_k):
            break

    if len(diverse) < int(top_k):
        for ng, txt, _ in scored_text:
            if accept(txt, chosen_text, False):
                diverse.append(ng)
                chosen_text.append(txt)
            if len(diverse) >= int(top_k):
                break

    if code_phrases_extra:
        for phrase in code_phrases_extra:
            if not phrase:
                continue
            if any(phrase in t or t in phrase for t in (chosen_code + chosen_text)):
                continue
            ids = tokenizer(phrase, add_special_tokens=False)["input_ids"]
            if len(ids) > 1:
                diverse.insert(0, tuple(ids))

    return diverse[: int(top_k)]


def add_ngram_tokens_and_init_embeddings(tokenizer, model, ngram_list: List[Tuple[int, ...]]) -> Dict[str, int]:
    orig_vocab = tokenizer.get_vocab()
    texts: List[Tuple[str, Tuple[int, ...]]] = []
    seen = set()

    for ng in ngram_list:
        try:
            t = decode_ngram_text(tokenizer, ng)
        except Exception:
            continue
        if not t or t in seen:
            continue
        seen.add(t)
        texts.append((t, ng))

    to_add: List[AddedToken] = []
    new_texts: List[Tuple[str, Tuple[int, ...]]] = []
    for t, ng in texts:
        if t in orig_vocab:
            continue
        to_add.append(AddedToken(t, single_word=False, lstrip=False, rstrip=False, normalized=False))
        new_texts.append((t, ng))

    added = tokenizer.add_tokens(to_add) if to_add else 0
    if added > 0 and hasattr(model, "resize_token_embeddings"):
        model.resize_token_embeddings(len(tokenizer))

    updated = 0
    if new_texts:
        emb = model.get_input_embeddings()
        with torch.no_grad():
            for t, ids_src in new_texts:
                tid = tokenizer.convert_tokens_to_ids(t)
                if tid is None or tid < 0:
                    continue
                vec = torch.stack([emb.weight.data[i] for i in ids_src]).mean(dim=0)
                emb.weight.data[tid] = vec
                updated += 1

    return {"added": int(added), "updated": int(updated)}


def maybe_load_ngrams_from_map(model_dir: str, tokenizer, model) -> Dict[str, int]:
    map_path = Path(model_dir) / "ngram_token_map.json"
    if not map_path.exists():
        return {"found": 0, "added": 0, "updated": 0}

    try:
        nmap = json.loads(map_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {"found": 1, "added": 0, "updated": 0}

    vocab = tokenizer.get_vocab()
    to_add: List[AddedToken] = []
    init_src: List[Tuple[str, List[int]]] = []

    for phrase in list(nmap.keys()):
        if not isinstance(phrase, str) or not phrase.strip():
            continue
        if phrase in vocab:
            continue
        ids_src = tokenizer(phrase, add_special_tokens=False)["input_ids"]
        if len(ids_src) <= 1:
            continue
        init_src.append((phrase, ids_src))
        to_add.append(AddedToken(phrase, single_word=False, lstrip=False, rstrip=False, normalized=False))

    added = tokenizer.add_tokens(to_add) if to_add else 0
    updated = 0
    if added > 0 and hasattr(model, "resize_token_embeddings"):
        model.resize_token_embeddings(len(tokenizer))
        emb = model.get_input_embeddings()
        with torch.no_grad():
            for phrase, ids_src in init_src:
                tid = tokenizer.convert_tokens_to_ids(phrase)
                if tid is None or tid < 0:
                    continue
                vec = torch.stack([emb.weight.data[i] for i in ids_src]).mean(dim=0)
                emb.weight.data[tid] = vec
                updated += 1

    return {"found": 1, "added": int(added), "updated": int(updated)}


# ----------------------------
# Dataset: CSV -> (prompt, answer)
# ----------------------------
def _normalize_id(val: Any) -> str:
    if val is None:
        return ""
    s = str(val).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


def _row_to_messages(row: Dict[str, Any]) -> List[Tuple[str, str]]:
    msgs: List[Tuple[str, str]] = []
    user = (row.get("Benutzer") or "").strip()
    ctx = (row.get("Kontext") or "").strip()
    asst = (row.get("Assistentin") or "").strip()

    if user:
        content = f"{ctx}\n{user}".strip() if ctx else user
        msgs.append(("user", content))
    if asst:
        msgs.append(("assistant", asst))
    return msgs


def _build_chain_to_root(target: Dict[str, Any], id2row: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
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
    return chain


def _chain_to_message_list(chain: List[Dict[str, Any]]) -> Tuple[str, List[Tuple[str, str]]]:
    system_text = (chain[0].get("system") or "").strip() if chain else ""
    messages: List[Tuple[str, str]] = []
    for row in chain:
        messages.extend(_row_to_messages(row))
    return system_text, messages


def _render_chat_prompt(system_text: str, history: List[Tuple[str, str]]) -> str:
    parts: List[str] = ["<s>\n"]
    if system_text:
        parts += ["<|System|>\n", system_text, "\n"]
    for role, content in history:
        if role == "user":
            parts += ["<|Benutzer|>\n", content, "\n"]
        elif role == "assistant":
            parts += ["<|Assistentin|>\n", content, "\n"]
    parts += ["<|Assistentin|>\n"]
    return "".join(parts)


def _render_dialogplus_prompt(system_text: str, history: List[Tuple[str, str]]) -> str:
    parts: List[str] = []
    if system_text:
        parts += ["<|System|>\n", system_text, "\n</s>\n"]
    for role, content in history:
        if role == "user":
            parts += ["<|Benutzer|>\n", content, "\n</s>\n"]
        elif role == "assistant":
            parts += ["<|Assistentin|>\n", content, "\n</s>\n"]
    parts += ["<|Assistentin|>\n"]
    return "".join(parts)


def dialogplus_block_iter(csv_path: str, shuffle_threads: bool = False) -> Iterator[Tuple[str, str]]:
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows: List[Dict[str, Any]] = []
        for idx, row in enumerate(reader):
            row["_rowidx"] = idx
            row["id"] = _normalize_id(row.get("id", ""))
            row["parent_id"] = _normalize_id(row.get("parent_id", ""))
            rows.append(row)

    id2row = {r["id"]: r for r in rows if r.get("id")}
    candidates = [r for r in rows if r.get("id") and (r.get("Assistentin") or "").strip()]
    if not candidates:
        return

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
        root_id, depth = root_and_depth(r["id"])
        threads.setdefault(root_id, []).append((depth, int(r["_rowidx"]), r))

    for root_id in list(threads.keys()):
        threads[root_id].sort(key=lambda t: (t[0], t[1]))

    order = list(threads.keys())
    if shuffle_threads:
        random.shuffle(order)

    for root_id in order:
        for _, __, target in threads[root_id]:
            chain = _build_chain_to_root(target, id2row)
            if not chain:
                continue
            system_text, messages = _chain_to_message_list(chain)
            if not messages or messages[-1][0] != "assistant":
                continue
            target_text = messages[-1][1].strip()
            if not target_text:
                continue
            history = messages[:-1]
            yield _render_dialogplus_prompt(system_text, history), (target_text + "\n</s>")


def chat_block_iter(csv_path: str, shuffle_threads: bool = False) -> Iterator[Tuple[str, str]]:
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows: List[Dict[str, Any]] = []
        for idx, row in enumerate(reader):
            row["_rowidx"] = idx
            row["id"] = _normalize_id(row.get("id", ""))
            row["parent_id"] = _normalize_id(row.get("parent_id", ""))
            rows.append(row)

    id2row = {r["id"]: r for r in rows if r.get("id")}
    candidates = [r for r in rows if r.get("id") and (r.get("Assistentin") or "").strip()]
    if not candidates:
        return

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
        root_id, depth = root_and_depth(r["id"])
        threads.setdefault(root_id, []).append((depth, int(r["_rowidx"]), r))

    for root_id in list(threads.keys()):
        threads[root_id].sort(key=lambda t: (t[0], t[1]))

    order = list(threads.keys())
    if shuffle_threads:
        random.shuffle(order)

    for root_id in order:
        for _, __, target in threads[root_id]:
            chain = _build_chain_to_root(target, id2row)
            if not chain:
                continue
            system_text, messages = _chain_to_message_list(chain)
            if not messages or messages[-1][0] != "assistant":
                continue
            target_text = messages[-1][1].strip()
            if not target_text:
                continue
            history = messages[:-1]
            yield _render_chat_prompt(system_text, history), (target_text + "\n</s>")


def iter_chunked_training_blocks(
    csv_path: str,
    tokenizer,
    *,
    template_mode: str,
    shuffle: bool,
    sort_by_length: bool,
    chunk_size: int,
    pairs_per_block: int = 10_000,
    preview_callback=None,
) -> Iterator[List[Dict[str, Any]]]:
    """
    Liefert blockweise ein gepaddetes Dataset:
      - input_ids: links gepadded auf chunk_size
      - attention_mask: 0 für pad links, 1 für tokens
      - labels:
          * template_mode == "chat": FULL-LOSS -> labels == input_ids (Padding links = -100)
          * template_mode == "dialogplus": ANSWER-ONLY -> Prompt = -100, Answer = token ids (Padding links = -100)
    """
    chunk_len = int(chunk_size)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    pad_id = int(pad_id)

    def pad_left(seq: List[int], target_len: int, pad_value: int) -> List[int]:
        if len(seq) >= target_len:
            return seq[-target_len:]
        return [pad_value] * (target_len - len(seq)) + seq

    def pad_left_mask(length: int, target_len: int) -> List[int]:
        if length >= target_len:
            return [1] * target_len
        return [0] * (target_len - length) + [1] * length

    def pad_left_labels(labels: List[int], target_len: int) -> List[int]:
        if len(labels) >= target_len:
            return labels[-target_len:]
        return [-100] * (target_len - len(labels)) + labels

    preview_done = False

    def add_pair(dataset_block: List[Dict[str, Any]], prompt_text: str, answer_text: str) -> None:
        nonlocal preview_done

        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(answer_text, add_special_tokens=False)["input_ids"]
        if not prompt_ids and not answer_ids:
            return

        if callable(preview_callback) and not preview_done:
            try:
                preview_callback((prompt_text + answer_text)[:20000])
            except Exception:
                pass
            preview_done = True

        input_ids_full = list(prompt_ids) + list(answer_ids)

        if template_mode == "dialogplus":
            labels_full = ([-100] * len(prompt_ids)) + list(answer_ids)
        else:
            labels_full = list(input_ids_full)

        for i in range(0, len(input_ids_full), chunk_len):
            block_in = input_ids_full[i : i + chunk_len]
            block_lb = labels_full[i : i + chunk_len]
            if len(block_in) < 2:
                continue
            dataset_block.append(
                {
                    "input_ids": pad_left(block_in, chunk_len, pad_id),
                    "attention_mask": pad_left_mask(len(block_in), chunk_len),
                    "labels": pad_left_labels(block_lb, chunk_len),
                }
            )

    if template_mode == "chat":
        pair_iter = chat_block_iter(csv_path, shuffle_threads=shuffle)
    elif template_mode == "dialogplus":
        pair_iter = dialogplus_block_iter(csv_path, shuffle_threads=shuffle)
    else:
        raise ValueError("template_mode muss chat oder dialogplus sein.")

    buf: List[Tuple[str, str]] = []
    for p, a in pair_iter:
        buf.append((p, a))
        if len(buf) >= int(pairs_per_block):
            if sort_by_length:
                buf.sort(key=lambda pa: len(tokenizer(pa[0] + pa[1], add_special_tokens=False)["input_ids"]))
            dataset_block: List[Dict[str, Any]] = []
            for pp, aa in buf:
                add_pair(dataset_block, pp, aa)
            yield dataset_block
            buf = []

    if buf:
        if sort_by_length:
            buf.sort(key=lambda pa: len(tokenizer(pa[0] + pa[1], add_special_tokens=False)["input_ids"]))
        dataset_block = []
        for pp, aa in buf:
            add_pair(dataset_block, pp, aa)
        yield dataset_block

# ----------------------------
# LoRA target detection & Utils
# ----------------------------
def detect_lora_targets(model) -> List[str]:
    wanted = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj", "out_proj",
        "fc1", "fc2", "wq", "wk", "wv", "wo", "w1", "w2", "w3",
    ]
    found = set()
    for full_name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            tail = full_name.split(".")[-1].lower()
            for frag in wanted:
                if frag in tail:
                    found.add(tail)
                    break
    return sorted(found)


def count_model_parameters(model) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return int(total), int(trainable)


def _empty_device_caches() -> None:
    try:
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
            except Exception:
                pass
    except Exception:
        pass


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
    _empty_device_caches()


def get_new_output_dir(model_name: str, base_dir: Optional[Path] = None) -> Path:
    now = time.strftime("%Y-%m-%d_%H-%M-%S")
    base = base_dir or TRAINING_OUT_DIR
    out = base / f"{model_name}_{now}"
    out.mkdir(parents=True, exist_ok=True)
    return out


# ----------------------------
# Tokenizer Setup
# ----------------------------
def prepare_tokenizer_for_matelix(tokenizer) -> bool:
    need_resize = False

    if tokenizer.pad_token_id is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        need_resize = True

    role_tokens = [
        AddedToken("<|System|>", single_word=False, lstrip=False, rstrip=False, normalized=False, special=False),
        AddedToken("<|Benutzer|>", single_word=False, lstrip=False, rstrip=False, normalized=False, special=False),
        AddedToken("<|Assistentin|>", single_word=False, lstrip=False, rstrip=False, normalized=False, special=False),
    ]
    if tokenizer.add_tokens(role_tokens):
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


# ----------------------------
# Training configuration
# ----------------------------
class TrainConfig(BaseModel):
    model_dir: str = Field(..., description="HF repo id oder lokaler Pfad")
    csv_path: str = Field(..., description="CSV dataset path")

    device: str = Field(default="auto", description="auto|cpu|cuda|mps")
    train_mode: str = Field(default="full", description="full|lora")

    learning_rate: float = 4e-4
    lr_schedule: str = "linear"

    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 7
    num_train_epochs: float = 3.0
    max_steps: Optional[int] = None

    chunk_size: int = 1024
    max_seq_length: Optional[int] = None

    template_mode: str = "chat"  # chat|dialogplus
    shuffle: bool = False
    sort_by_length: bool = True

    # chunked dataset
    pairs_per_block: int = 10_000

    # LoRA
    lora_r: int = 8
    lora_alpha: int = 16
    merge_lora_on_save: bool = True

    # precision & misc
    precision_mode: str = "auto"  # auto|fp32|fp16|bf16
    gradient_checkpointing: bool = False
    save_dir: Optional[str] = None
    dataloader_num_workers: int = 0
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01

    # ngrams
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

    @model_validator(mode="after")
    def _sync_max_seq_length(self):
        if self.max_seq_length is not None:
            self.chunk_size = int(self.max_seq_length)
        return self


# ----------------------------
# Ngram optimization (light)
# ----------------------------
def optimize_tokenizer_ngrams(
    tokenizer,
    csv_path: str,
    *,
    template_mode: str,
    max_ngram: int,
    top_k: int,
    min_chars: int,
    min_words: int,
    max_samples: int,
    min_count: int,
    max_token_chars: int,
    max_tokens_per_text: int,
    log_fn=None,
) -> List[Tuple[int, ...]]:
    def log(s: str):
        if callable(log_fn):
            log_fn(s)

    # Für Ngrams nutzen wir die lineare Dialog-Repräsentation (einfacher):
    texts: List[str] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= int(max_samples):
                break
            # sehr simple: concat Spalten (für ngram scanning ausreichend)
            sys_ = (row.get("system") or "").strip()
            u = (row.get("Benutzer") or "").strip()
            k = (row.get("Kontext") or "").strip()
            a = (row.get("Assistentin") or "").strip()
            t = "\n".join([x for x in [sys_, k, u, a] if x])
            if t:
                texts.append(t)

    if not texts:
        log("Keine Texte gefunden für N-Gramme.")
        return []

    log(f"Scanne N-Gramme (max_ngram={max_ngram}, top_k={top_k}) …")
    return find_frequent_ngrams_diverse(
        tokenizer,
        texts,
        max_ngram=int(max_ngram),
        top_k=int(top_k),
        min_chars=int(min_chars),
        min_words=int(min_words),
        min_count=int(min_count),
        max_token_chars=int(max_token_chars),
        max_tokens_per_text=int(max_tokens_per_text),
    )


# ----------------------------
# Global state (Training)
# ----------------------------
class TrainingState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.running: bool = False
        self.step: int = 0
        self.loss: Optional[float] = None
        self.learning_rate: Optional[float] = None
        self.last_preview: str = ""
        self.last_preview_full: str = ""
        self.tokens_per_step: Optional[int] = None
        self.total_tokens: int = 0
        self.eta: str = ""
        self.log = LogStore(max_lines=4000)
        self.active_config: Optional[Dict[str, Any]] = None
        self.save_dir: Optional[str] = None

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
                "config": self.active_config,
                "save_dir": self.save_dir,
            }


TRAIN_STATE = TrainingState()


class TrainingManager:
    def __init__(self, state: TrainingState) -> None:
        self.state = state
        self.thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()

    def start(self, cfg: TrainConfig) -> Dict[str, Any]:
        with self.state.lock:
            if self.state.running:
                return {"msg": "Training läuft bereits", "running": True}
            self.state.running = True
            self.state.step = 0
            self.state.loss = None
            self.state.learning_rate = None
            self.state.last_preview = ""
            self.state.last_preview_full = ""
            self.state.tokens_per_step = None
            self.state.total_tokens = 0
            self.state.eta = ""
            self.state.active_config = cfg.model_dump()
            self.state.save_dir = None
            self.state.log.clear()
            self.state.log.append("Training gestartet.")

        self.stop_event.clear()
        self.thread = threading.Thread(target=self._train_thread, args=(cfg,), daemon=True)
        self.thread.start()
        return {"msg": "Training gestartet", "running": True}

    def stop(self) -> Dict[str, Any]:
        self.stop_event.set()
        with self.state.lock:
            self.state.eta = "stopping"
        self.state.log.append("Stop-Signal gesetzt. Training wird beendet …")
        return {"msg": "Stop-Signal gesendet"}

    def _train_thread(self, cfg: TrainConfig) -> None:
        trainer = None
        model = None
        tokenizer = None
        adapter_injected = False

        try:
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                Trainer,
                TrainerCallback,
                TrainingArguments,
                default_data_collator,
            )

            model_name = Path(cfg.model_dir.rstrip("/")).name or "Model"
            base_out = Path(cfg.save_dir).expanduser() if (cfg.save_dir and cfg.save_dir.strip()) else TRAINING_OUT_DIR
            save_dir = get_new_output_dir(model_name=model_name, base_dir=base_out)
            save_dir.mkdir(parents=True, exist_ok=True)

            with self.state.lock:
                self.state.save_dir = str(save_dir)

            self.state.log.set_file(str(save_dir / "training.log"))
            self.state.log.append(f"Logfile: {save_dir / 'training.log'}")

            (save_dir / "train_config.json").write_text(json.dumps(cfg.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8")

            device = self._select_device(cfg.device)
            self.state.log.append(f"Device: {device.type.upper()} (requested={cfg.device})")

            tokenizer = AutoTokenizer.from_pretrained(cfg.model_dir, trust_remote_code=HF_TRUST_REMOTE_CODE)
            need_resize = prepare_tokenizer_for_matelix(tokenizer)

            # optional ngrams
            ngram_list: List[Tuple[int, ...]] = []
            if cfg.use_ngrams:
                self.state.log.append("N-Gramm-Optimierung aktiv …")
                ngram_list = optimize_tokenizer_ngrams(
                    tokenizer,
                    cfg.csv_path,
                    template_mode=cfg.template_mode,
                    max_ngram=cfg.ngram_max,
                    top_k=cfg.ngram_top_k,
                    min_chars=cfg.ngram_min_chars,
                    min_words=cfg.ngram_min_words,
                    max_samples=cfg.ngram_max_samples,
                    min_count=cfg.ngram_min_count,
                    max_token_chars=cfg.ngram_max_token_chars,
                    max_tokens_per_text=cfg.ngram_max_tokens_per_text,
                    log_fn=self.state.log.append,
                )
                self.state.log.append(f"N-Gramm selected: {len(ngram_list)}")
                if ngram_list:
                    need_resize = True

            load_dtype, fp16, bf16 = self._select_precision(cfg.precision_mode, device)
            self.state.log.append(f"Precision: load_dtype={load_dtype}, fp16={fp16}, bf16={bf16}")

            patch_local_config_json_if_exists(cfg.model_dir)
            clean_cfg = load_clean_config(cfg.model_dir)

            model = AutoModelForCausalLM.from_pretrained(
                cfg.model_dir,
                config=clean_cfg,
                trust_remote_code=HF_TRUST_REMOTE_CODE,
                torch_dtype=load_dtype,
                low_cpu_mem_usage=True,
            )

            if device.type in ("cpu", "mps") or cfg.precision_mode.lower() == "fp32":
                model = model.to(torch.float32)

            if need_resize or model.get_input_embeddings().weight.shape[0] != len(tokenizer):
                model.resize_token_embeddings(len(tokenizer))
                self.state.log.append(f"Embeddings resized auf vocab={len(tokenizer)}")

            if cfg.use_ngrams and ngram_list:
                res = add_ngram_tokens_and_init_embeddings(tokenizer, model, ngram_list)
                self.state.log.append(f"N-Gramm Embeddings: added={res['added']} updated={res['updated']}")

                try:
                    phrase_map: Dict[str, int] = {}
                    for ng in ngram_list:
                        txt = decode_ngram_text(tokenizer, ng)
                        if txt and txt not in phrase_map:
                            phrase_map[txt] = 1
                    (save_dir / "ngram_token_map.json").write_text(json.dumps(phrase_map, ensure_ascii=False, indent=2), encoding="utf-8")
                except Exception:
                    pass

            model.to(device)
            model.train()
            if hasattr(model, "config"):
                model.config.use_cache = False

            train_mode = cfg.train_mode.lower().strip()
            if train_mode == "lora":
                try:
                    from peft import LoraConfig, TaskType, get_peft_model

                    detected = detect_lora_targets(model)
                    self.state.log.append(f"[LoRA] Detected targets: {detected or '—'}")
                    preferred = [t for t in detected if t in {"q_proj", "k_proj", "v_proj", "o_proj"}]
                    target_modules: Union[str, List[str]] = sorted(set(preferred)) if preferred else (sorted(set(detected)) if detected else "all-linear")

                    lcfg = LoraConfig(
                        r=int(cfg.lora_r),
                        lora_alpha=int(cfg.lora_alpha),
                        target_modules=target_modules,
                        lora_dropout=0.1,
                        bias="none",
                        task_type=TaskType.CAUSAL_LM,
                        modules_to_save=["lm_head"],
                    )
                    model = get_peft_model(model, lcfg)
                    total, trainable = count_model_parameters(model)
                    adapter_injected = trainable > 0
                    self.state.log.append(f"[LoRA] enabled. total={total:,} trainable={trainable:,}")
                except Exception as e:
                    self.state.log.append(f"[LoRA][FAIL] {e} -> full")
                    self.state.log.append(traceback.format_exc())
                    train_mode = "full"

            if train_mode == "full":
                for p in model.parameters():
                    p.requires_grad_(True)
                total, trainable = count_model_parameters(model)
                self.state.log.append(f"[FULL] total={total:,} trainable={trainable:,}")

            # gradient checkpointing
            try:
                if cfg.gradient_checkpointing and device.type == "cuda":
                    model.gradient_checkpointing_enable()
                    if hasattr(model, "enable_input_require_grads"):
                        model.enable_input_require_grads()
                    self.state.log.append("Gradient Checkpointing: ON")
                else:
                    if hasattr(model, "gradient_checkpointing_disable"):
                        model.gradient_checkpointing_disable()
                    self.state.log.append("Gradient Checkpointing: OFF")
            except Exception as e:
                self.state.log.append(f"Gradient Checkpointing Hinweis: {e}")

            def preview_cb(s: str) -> None:
                with self.state.lock:
                    self.state.last_preview_full = s
                    self.state.last_preview = s[:4000]

            # TrainingArguments
            num_train_epochs = float(cfg.num_train_epochs or 1.0)
            ta_kwargs: Dict[str, Any] = dict(
                output_dir=str(save_dir),
                per_device_train_batch_size=int(cfg.per_device_train_batch_size),
                gradient_accumulation_steps=int(cfg.gradient_accumulation_steps),
                num_train_epochs=num_train_epochs,
                save_strategy="epoch",
                logging_strategy="steps",
                logging_steps=1,
                logging_first_step=True,
                report_to="none",
                lr_scheduler_type=str(cfg.lr_schedule),
                learning_rate=float(cfg.learning_rate),
                dataloader_num_workers=0,
                optim="adamw_torch",
                fp16=bool(fp16),
                bf16=bool(bf16),
                dataloader_pin_memory=(device.type == "cuda"),
                disable_tqdm=False,
                max_grad_norm=float(cfg.max_grad_norm),
                weight_decay=float(cfg.weight_decay),
            )
            if isinstance(cfg.max_steps, int) and cfg.max_steps > 0:
                ta_kwargs["max_steps"] = int(cfg.max_steps)

            training_args = TrainingArguments(**ta_kwargs)

            self_outer = self

            class StopCallback(TrainerCallback):
                def on_step_end(self, args, state, control, **kwargs):
                    if self_outer.stop_event.is_set():
                        control.should_training_stop = True
                    return control

            class WebUICallback(TrainerCallback):
                def __init__(self):
                    self.t0 = time.time()

                def on_log(self, args, state, control, logs=None, **kwargs):
                    if not logs:
                        return
                    step = int(getattr(state, "global_step", 0) or 0)
                    loss = logs.get("loss")
                    lr = logs.get("learning_rate")

                    with self_outer.state.lock:
                        self_outer.state.step = step
                        self_outer.state.loss = float(loss) if loss is not None else None
                        self_outer.state.learning_rate = float(lr) if lr is not None else None

                        bs = int(getattr(args, "per_device_train_batch_size", 1) or 1)
                        ga = int(getattr(args, "gradient_accumulation_steps", 1) or 1)
                        world = int(getattr(args, "world_size", 1) or 1)
                        tps = int(cfg.chunk_size) * bs * ga * world
                        self_outer.state.tokens_per_step = tps
                        self_outer.state.total_tokens = step * tps

                    if loss is not None:
                        self_outer.state.log.append(f"Step {step} | Loss: {float(loss):.6f} | LR: {lr}")

            class PreviewCollator:
                def __init__(self, base_collator, tokenizer_, state_: TrainingState, max_chars: int = 4000):
                    self.base = base_collator
                    self.tok = tokenizer_
                    self.state = state_
                    self.max_chars = max_chars

                def __call__(self, features):
                    batch = self.base(features)
                    try:
                        input_ids = batch.get("input_ids")
                        if input_ids is not None:
                            ids0_full = input_ids[0].tolist()
                            text_full = self.tok.decode(ids0_full, skip_special_tokens=False)
                            with self.state.lock:
                                self.state.last_preview_full = text_full
                                self.state.last_preview = text_full[: self.max_chars]
                    except Exception:
                        pass
                    return batch

            preview_collator = PreviewCollator(default_data_collator, tokenizer, self.state)

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=[],  # placeholder
                data_collator=preview_collator,
                callbacks=[WebUICallback(), StopCallback()],
            )

            # -------------------------
            # CHUNKED TRAINING LOOP
            # -------------------------
            self.state.log.append("Chunked Dataset Build + Training startet …")

            trained_any = False
            block_iter = iter_chunked_training_blocks(
                cfg.csv_path,
                tokenizer,
                template_mode=cfg.template_mode,
                shuffle=bool(cfg.shuffle),
                sort_by_length=bool(cfg.sort_by_length),
                chunk_size=int(cfg.chunk_size),
                pairs_per_block=int(cfg.pairs_per_block),
                preview_callback=preview_cb,
            )

            for block_idx, dataset_block in enumerate(block_iter, start=1):
                if self.stop_event.is_set():
                    self.state.log.append("Stop erkannt -> beende.")
                    break

                self.state.log.append(f"[CHUNK] Block {block_idx}: {len(dataset_block)} Samples")

                if not dataset_block:
                    continue

                # einmaliger Debug-Check
                if not trained_any:
                    one = dataset_block[0]
                    batch_dbg = {
                        "input_ids": torch.tensor(one["input_ids"], device=device, dtype=torch.long).unsqueeze(0),
                        "attention_mask": torch.tensor(one["attention_mask"], device=device, dtype=torch.long).unsqueeze(0),
                        "labels": torch.tensor(one["labels"], device=device, dtype=torch.long).unsqueeze(0),
                    }
                    out_dbg = model(**batch_dbg)
                    loss_dbg = out_dbg.loss
                    self.state.log.append(f"[DEBUG] Loss={float(loss_dbg):.6f}, requires_grad={loss_dbg.requires_grad}")
                    if not loss_dbg.requires_grad:
                        raise RuntimeError("Loss requires_grad=False -> keine trainierbaren Parameter aktiv.")
                    loss_dbg.backward()
                    self.state.log.append("[DEBUG] Backward OK")

                trainer.train_dataset = dataset_block
                trainer.train(resume_from_checkpoint=False)
                trained_any = True

                gc.collect()
                _empty_device_caches()

            if not trained_any:
                raise RuntimeError("Kein Trainingssample gefunden (alle Blöcke leer).")

            self.state.log.append("Training abgeschlossen.")

            # Save
            self.state.log.append(f"Speichere Modell nach: {save_dir}")

            final_model = trainer.model
            try:
                if hasattr(final_model, "config"):
                    clean = _normalize_and_clean_rope(final_model.config.to_dict())
                    final_model.config.__dict__.update(clean)
            except Exception:
                pass

            if cfg.train_mode.lower() == "lora" and adapter_injected and cfg.merge_lora_on_save:
                try:
                    final_model = final_model.merge_and_unload(progressbar=True, safe_merge=True)
                    final_model.save_pretrained(str(save_dir))
                    self.state.log.append("LoRA gemerged und Modell gespeichert.")
                except Exception as e:
                    self.state.log.append(f"[SAVE][LoRA-MERGE FAIL] {e} -> speichere Adapter-Modell …")
                    self.state.log.append(traceback.format_exc())
                    trainer.save_model(str(save_dir))
            else:
                trainer.save_model(str(save_dir))

            if tokenizer is not None:
                tokenizer.save_pretrained(str(save_dir))
                spm_path = getattr(tokenizer, "vocab_file", None)
                if spm_path and Path(spm_path).is_file():
                    shutil.copy2(spm_path, str(save_dir / "tokenizer.model"))

            self.state.log.append("Modell & Tokenizer gespeichert.")

        except Exception as e:
            self.state.log.append(f"[TRAIN ERROR] {e.__class__.__name__}: {e}")
            self.state.log.append(traceback.format_exc())
        finally:
            self.state.log.append("Cleanup: entlade Trainer/Model/Tokenizer …")
            try:
                if model is not None:
                    try:
                        model.to("cpu")
                    except Exception:
                        pass
                unload_torch_objects(trainer, model, tokenizer)
            except Exception:
                pass

            with self.state.lock:
                self.state.running = False
                if self.stop_event.is_set():
                    self.state.eta = ""

            self.state.log.append("Training beendet.")

    @staticmethod
    def _select_device(requested: str) -> torch.device:
        r = (requested or "").strip().lower()
        if r == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if r == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        if r == "cpu":
            return torch.device("cpu")
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @staticmethod
    def _select_precision(mode: str, device: torch.device) -> Tuple[Optional[torch.dtype], bool, bool]:
        want = (mode or "auto").strip().lower()
        if want not in {"auto", "fp32", "fp16", "bf16"}:
            want = "auto"
        if device.type in ("cpu", "mps"):
            return None, False, False
        if want == "fp32":
            return None, False, False
        if want == "bf16":
            can = torch.cuda.is_bf16_supported()
            return (torch.bfloat16 if can else None), False, bool(can)
        if want == "fp16":
            return torch.float16, True, False
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16, False, True
        return torch.float16, True, False


TRAIN_MANAGER = TrainingManager(TRAIN_STATE)

# ----------------------------
# Inference / Chat
# ----------------------------
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


class SafeLogitsProcessor:
    def __call__(self, input_ids, scores):
        if not torch.isfinite(scores).all():
            scores = scores.clone()
            scores[~torch.isfinite(scores)] = -1e9
        return torch.clamp(scores, min=-1e9, max=1e9)


def _preferred_device_name(req: Optional[str] = None) -> str:
    r = (req or "").strip().lower()
    if r in {"cpu", "cuda", "mps"}:
        return r
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _get_device(name: str) -> torch.device:
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if name == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _prepare_inputs(tokenizer, messages: List[Dict[str, Any]], system: Optional[str], device: torch.device):
    msgs = list(messages or [])
    if msgs and isinstance(msgs[-1], dict):
        if msgs[-1].get("role") == "assistant" and not (msgs[-1].get("content") or "").strip():
            msgs = msgs[:-1]

    if system and system.strip():
        msgs = [{"role": "system", "content": system.strip()}] + msgs

    if hasattr(tokenizer, "apply_chat_template"):
        enc = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    else:
        parts: List[str] = []
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
        text = "".join(parts)
        enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)

    pad_id = tokenizer.pad_token_id
    if pad_id is None or pad_id == tokenizer.eos_token_id:
        pad_id = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else (tokenizer.eos_token_id or 0)

    eos_id = tokenizer.eos_token_id or tokenizer.convert_tokens_to_ids("</s>") or pad_id

    input_ids = enc["input_ids"]
    attention_mask = enc.get("attention_mask") or (input_ids != pad_id).long()

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

    return safe, LogitsProcessorList([SafeLogitsProcessor()])


class InferenceSession:
    def __init__(self):
        self.lock = threading.Lock()
        self.loaded_dir: Optional[str] = None
        self.device: Optional[torch.device] = None
        self.tokenizer = None
        self.model = None


INFER = InferenceSession()


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

        tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=HF_TRUST_REMOTE_CODE)
        need_resize = prepare_tokenizer_for_matelix(tok)

        patch_local_config_json_if_exists(model_dir)
        clean_cfg = load_clean_config(model_dir)

        dtype = torch.float16 if dev.type == "cuda" else None
        mdl = AutoModelForCausalLM.from_pretrained(
            model_dir,
            config=clean_cfg,
            trust_remote_code=HF_TRUST_REMOTE_CODE,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )

        if dev.type == "mps":
            mdl = mdl.to(torch.float32)

        ng = maybe_load_ngrams_from_map(model_dir, tok, mdl)
        if (ng.get("added", 0) or ng.get("updated", 0)) and not need_resize:
            need_resize = True

        if need_resize and hasattr(mdl, "resize_token_embeddings"):
            mdl.resize_token_embeddings(len(tok))

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


# ----------------------------
# API endpoints: training
# ----------------------------
@app.get("/hardware")
def api_hardware():
    return get_hardware_info()


@app.get("/sysstatus")
def api_sysstatus():
    return get_system_status()


@app.get("/trainings")
def api_trainings():
    base_dir = TRAINING_OUT_DIR
    if not base_dir.exists():
        return []
    trainings = []
    for d in sorted(base_dir.iterdir()):
        if not d.is_dir():
            continue
        train_cfg = d / "train_config.json"
        if not train_cfg.exists():
            continue
        try:
            cfg_obj = json.loads(train_cfg.read_text(encoding="utf-8"))
        except Exception:
            cfg_obj = {}
        trainings.append({"folder": d.name, "config": cfg_obj})
    return trainings


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


@app.post("/start")
def api_start_training(cfg: TrainConfig):
    res = TRAIN_MANAGER.start(cfg)
    if not res.get("running"):
        return JSONResponse(res, status_code=400)
    return res


@app.post("/stop")
def api_stop_training():
    return TRAIN_MANAGER.stop()


@app.get("/status")
def api_status():
    return TRAIN_STATE.snapshot()


@app.get("/logs")
def api_logs():
    return {"log": TRAIN_STATE.log.tail(200)}


@app.get("/livepreview")
def api_livepreview():
    with TRAIN_STATE.lock:
        return {"preview": TRAIN_STATE.last_preview, "preview_full": TRAIN_STATE.last_preview_full}


@app.websocket("/ws/logs")
async def ws_logs(websocket: WebSocket):
    await websocket.accept()
    store: LogStore = TRAIN_STATE.log
    cursor = store.last_id - 200
    try:
        while True:
            lines, cursor = store.since(cursor)
            if lines:
                await websocket.send_text("\n".join(lines))
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        return


# ----------------------------
# API endpoints: inference
# ----------------------------
@app.post("/load_inference")
def api_load_inference(cfg: Dict[str, Any] | None = Body(default=None)):
    cfg = cfg or {}
    model_dir = cfg.get("model_dir") or TRAIN_STATE.snapshot().get("save_dir") or cfg.get("fallback_model_dir")
    if not model_dir:
        return JSONResponse({"error": "model_dir fehlt."}, status_code=400)
    device = _preferred_device_name(cfg.get("device") or "auto")
    try:
        return JSONResponse(load_inference_model(model_dir, device), status_code=200)
    except FileNotFoundError as e:
        return JSONResponse({"error": str(e)}, status_code=404)
    except Exception as e:
        return JSONResponse({"error": f"{e.__class__.__name__}: {e}"}, status_code=500)


@app.post("/unload_inference")
def api_unload_inference():
    return unload_inference()


@app.post("/chat")
def api_chat(req: ChatRequest):
    try:
        model_dir, tok, mdl, dev = ensure_model_loaded(req.model_dir, req.device)
    except RuntimeError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": f"{e.__class__.__name__}: {e}"}, status_code=500)

    if tok is None or mdl is None or dev is None:
        return JSONResponse({"error": "Inference model not loaded."}, status_code=500)

    input_ids, attention_mask, pad_id, eos_id = _prepare_inputs(tok, req.messages, req.system, dev)
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
    generated = seq[0, input_ids.shape[-1] :]
    text = tok.decode(generated, skip_special_tokens=True)
    return {"model_dir": model_dir, "response": text.strip()}


@app.post("/chat_stream")
def api_chat_stream(req: ChatRequest):
    from transformers import TextIteratorStreamer

    try:
        model_dir, tok, mdl, dev = ensure_model_loaded(req.model_dir, req.device)
    except RuntimeError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": f"{e.__class__.__name__}: {e}"}, status_code=500)

    if tok is None or mdl is None or dev is None:
        return JSONResponse({"error": "Inference model not loaded."}, status_code=500)

    input_ids, attention_mask, pad_id, eos_id = _prepare_inputs(tok, req.messages, req.system, dev)
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
    thread = threading.Thread(
        target=mdl.generate,
        kwargs=dict(
            input_ids=input_ids[:1, :],
            attention_mask=attention_mask[:1, :] if attention_mask is not None else None,
            streamer=streamer,
            logits_processor=logits_processor,
            num_beams=1,
            num_return_sequences=1,
            **gen_kwargs,
        ),
        daemon=True,
    )
    thread.start()

    def generator():
        try:
            for piece in streamer:
                yield piece
        finally:
            thread.join(timeout=0.1)

    return StreamingResponse(
        generator(),
        media_type="text/plain; charset=utf-8",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ----------------------------
# OpenAI-kompatible API (/v1/*)
# ----------------------------
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


class OAIChatMessage(BaseModel):
    role: str
    content: str


class OAIChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[OAIChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = 50
    max_tokens: Optional[int] = 256
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    repetition_penalty: Optional[float] = 1.05
    do_sample: Optional[bool] = None
    device: Optional[str] = None


class OAICompletionRequest(BaseModel):
    model: Optional[str] = None
    prompt: Union[str, List[str]]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = 50
    max_tokens: Optional[int] = 256
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    repetition_penalty: Optional[float] = 1.05
    do_sample: Optional[bool] = None
    device: Optional[str] = None


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
    for m in local:
        path = str(TRAINING_OUT_DIR / m)
        if path not in models:
            models.append(path)
    return {"object": "list", "data": [{"id": mid, "object": "model", "created": 0, "owned_by": "owner"} for mid in models]}


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
                new_text = self.tokenizer.decode(input_ids[0][-1:], skip_special_tokens=False)
                self.buffer += new_text
                for s in self.stop_strings:
                    if s in self.buffer:
                        return True
                return False

        stopping_criteria = StoppingCriteriaList([StringStopCriteria(tok, stops)])

    if req.stream:
        from transformers import TextIteratorStreamer
        streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
        t = threading.Thread(
            target=mdl.generate,
            kwargs=dict(
                input_ids=input_ids[:1, :],
                attention_mask=attention_mask[:1, :] if attention_mask is not None else None,
                streamer=streamer,
                logits_processor=logits_processor,
                num_beams=1,
                num_return_sequences=1,
                stopping_criteria=stopping_criteria,
                **gen_kwargs,
            ),
            daemon=True,
        )
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

    with torch.no_grad():
        out = mdl.generate(
            input_ids=input_ids[:1, :],
            attention_mask=attention_mask[:1, :] if attention_mask is not None else None,
            logits_processor=logits_processor,
            num_beams=1,
            num_return_sequences=1,
            stopping_criteria=stopping_criteria,
            **gen_kwargs,
        )
    seq = out.sequences if hasattr(out, "sequences") else out
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
    for p in prompts:
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
        with torch.no_grad():
            out = mdl.generate(
                input_ids=input_ids[:1, :],
                attention_mask=attention_mask[:1, :] if attention_mask is not None else None,
                logits_processor=logits_processor,
                num_beams=1,
                num_return_sequences=1,
                **gen_kwargs,
            )
        seq = out.sequences if hasattr(out, "sequences") else out
        gen_ids = seq[0, input_ids.shape[-1]:]
        text = tok.decode(gen_ids, skip_special_tokens=True).strip()
        last_usage = {
            "prompt_tokens": int(input_ids.numel()),
            "completion_tokens": int(gen_ids.numel()),
            "total_tokens": int(input_ids.numel() + gen_ids.numel()),
        }
        results.append({"text": text, "index": 0, "logprobs": None, "finish_reason": "stop"})
    return {
        "id": f"cmpl-{uuid.uuid4().hex}",
        "object": "text_completion",
        "created": created,
        "model": model_id,
        "choices": results,
        "usage": last_usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


# ----------------------------
# Root
# ----------------------------
@app.get("/", response_class=HTMLResponse)
def root(_: Request):
    p = _ensure_index_html()
    return FileResponse(str(p))


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    import uvicorn
    import webbrowser

    def _open_browser():
        time.sleep(1)
        try:
            webbrowser.open("http://127.0.0.1:8002/")
        except Exception:
            pass

    threading.Thread(target=_open_browser, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=8002)
