#!/usr/bin/env python3
# matelix_ngram_pipeline.py
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

import csv
import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

csv.field_size_limit(1024 * 1024 * 128)

NGRAM_TOKEN_PREFIX = "<|ng:"
NGRAM_TOKEN_SUFFIX = "|>"


@dataclass
class NgramConfig:
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
    template_mode: str = "chat"
    column_name: str = "text"
    csv_path: str = ""


@dataclass
class NgramStats:
    scanned_samples: int
    selected_count: int
    candidate_count: int
    estimated_total_savings: float


@dataclass
class NgramState:
    version: int
    config: Dict[str, Any]
    tokens: List[str]
    phrases: List[str]
    phrase_to_token: Dict[str, str]
    stats: Dict[str, Any]


def normalize_id(val: Any) -> str:
    if val is None:
        return ""
    s = str(val).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


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


def _iter_candidate_chains(csv_path: str) -> Iterable[List[Dict[str, Any]]]:
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

    for root_id in threads:
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


def iter_training_texts(csv_path: str, template_mode: str, column_name: str) -> Iterable[str]:
    mode = (template_mode or "chat").strip().lower()

    if mode in {"chat", "dialogplus"}:
        for chain in _iter_candidate_chains(csv_path):
            parts: List[str] = []
            target_idx = len(chain) - 1
            if target_idx < 0:
                continue

            system_text = (chain[0].get("system") or "").strip()
            if system_text:
                parts.append(system_text)

            for j in range(target_idx + 1):
                row = chain[j]
                user = (row.get("Benutzer") or "").strip()
                ctx = (row.get("Kontext") or "").strip()
                asst = (row.get("Assistentin") or "").strip()

                if user:
                    parts.append(f"{ctx}\n{user}".strip() if ctx else user)
                if asst:
                    parts.append(asst)

            text = "\n".join(p for p in parts if p.strip()).strip()
            if text:
                yield text
        return

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            txt = (row.get(column_name) or "").strip()
            if txt:
                yield txt


_WORD_RE = re.compile(r"\S+")


def _tokenize_words(text: str) -> List[str]:
    return _WORD_RE.findall(text or "")


def _is_candidate_valid(
    phrase: str,
    cfg: NgramConfig,
) -> bool:
    phrase = (phrase or "").strip()
    if not phrase:
        return False
    if len(phrase) < int(cfg.ngram_min_chars):
        return False
    if len(_tokenize_words(phrase)) < int(cfg.ngram_min_words):
        return False
    if len(phrase) > int(cfg.ngram_max_token_chars):
        return False
    if phrase.startswith(NGRAM_TOKEN_PREFIX) and phrase.endswith(NGRAM_TOKEN_SUFFIX):
        return False
    return True


def collect_ngram_candidates(
    cfg: NgramConfig,
) -> Tuple[Counter, int]:
    counter: Counter = Counter()
    scanned_samples = 0
    max_n = max(2, int(cfg.ngram_max))

    for text in iter_training_texts(cfg.csv_path, cfg.template_mode, cfg.column_name):
        scanned_samples += 1
        if scanned_samples > int(cfg.ngram_max_samples):
            break

        words = _tokenize_words(text)
        if len(words) < 2:
            continue

        limit = min(len(words), int(cfg.ngram_max_tokens_per_text))
        words = words[:limit]

        for n in range(2, min(max_n, len(words)) + 1):
            for i in range(0, len(words) - n + 1):
                phrase = " ".join(words[i:i + n]).strip()
                if _is_candidate_valid(phrase, cfg):
                    counter[phrase] += 1

    if int(cfg.ngram_min_count) > 1:
        counter = Counter({k: v for k, v in counter.items() if v >= int(cfg.ngram_min_count)})

    return counter, scanned_samples


def _estimate_phrase_savings(
    tokenizer,
    phrase: str,
    count: int,
) -> float:
    try:
        base_ids = tokenizer(phrase, add_special_tokens=False)["input_ids"]
    except Exception:
        return 0.0

    base_len = len(base_ids)
    if base_len <= 1:
        return 0.0

    token_cost_with_ngram = 1
    gross_gain = base_len - token_cost_with_ngram
    if gross_gain <= 0:
        return 0.0

    # leichte Regularisierung gegen sehr seltene Kandidaten
    return float(gross_gain * max(0, count - 1))


def select_best_ngrams(
    tokenizer,
    cfg: NgramConfig,
    counter: Counter,
) -> Tuple[List[str], Dict[str, float]]:
    scored: List[Tuple[str, float]] = []
    score_map: Dict[str, float] = {}

    for phrase, count in counter.items():
        score = _estimate_phrase_savings(tokenizer, phrase, count)
        if score > 0:
            scored.append((phrase, score))
            score_map[phrase] = score

    scored.sort(key=lambda x: (-x[1], -counter[x[0]], -len(x[0]), x[0]))

    if not scored:
        return [], score_map

    top_k = max(1, int(cfg.ngram_top_k))
    if not bool(cfg.ngram_budgeted):
        return [p for p, _ in scored[:top_k]], score_map

    total_score = sum(score for _, score in scored)
    if total_score <= 0:
        return [p for p, _ in scored[:top_k]], score_map

    target_fit = float(cfg.ngram_target_fit)
    target_fit = min(max(target_fit, 0.0), 1.0)
    target_score = total_score * target_fit

    selected: List[str] = []
    running = 0.0
    for phrase, score in scored:
        if len(selected) >= top_k:
            break
        selected.append(phrase)
        running += score
        if running >= target_score:
            break

    return selected, score_map


def phrase_to_ngram_token(phrase: str) -> str:
    digest = hashlib.sha1(phrase.encode("utf-8")).hexdigest()[:16]
    return f"{NGRAM_TOKEN_PREFIX}{digest}{NGRAM_TOKEN_SUFFIX}"


def extend_tokenizer_with_ngrams(
    tokenizer,
    phrases: Sequence[str],
) -> Tuple[List[str], Dict[str, str], int]:
    clean_phrases: List[str] = []
    seen = set()
    for phrase in phrases:
        p = (phrase or "").strip()
        if not p or p in seen:
            continue
        seen.add(p)
        clean_phrases.append(p)

    tokens = [phrase_to_ngram_token(p) for p in clean_phrases]
    phrase_to_token = dict(zip(clean_phrases, tokens))

    added = tokenizer.add_tokens(tokens, special_tokens=False)
    return tokens, phrase_to_token, int(added)


def build_ngram_state(
    tokenizer,
    cfg: NgramConfig,
) -> NgramState:
    counter, scanned_samples = collect_ngram_candidates(cfg)
    selected_phrases, score_map = select_best_ngrams(tokenizer, cfg, counter)
    tokens, phrase_to_token, _ = extend_tokenizer_with_ngrams(tokenizer, selected_phrases)

    estimated_total_savings = float(sum(score_map.get(p, 0.0) for p in selected_phrases))
    stats = NgramStats(
        scanned_samples=scanned_samples,
        selected_count=len(selected_phrases),
        candidate_count=len(counter),
        estimated_total_savings=estimated_total_savings,
    )

    return NgramState(
        version=1,
        config=asdict(cfg),
        tokens=tokens,
        phrases=selected_phrases,
        phrase_to_token=phrase_to_token,
        stats=asdict(stats),
    )


def save_ngram_state(state: NgramState, outdir: Path) -> Path:
    outdir = Path(outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "ngram_state.json"
    path.write_text(json.dumps(asdict(state), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_ngram_state(path: Path) -> Optional[NgramState]:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return NgramState(
            version=int(payload.get("version", 1)),
            config=dict(payload.get("config", {})),
            tokens=list(payload.get("tokens", [])),
            phrases=list(payload.get("phrases", [])),
            phrase_to_token=dict(payload.get("phrase_to_token", {})),
            stats=dict(payload.get("stats", {})),
        )
    except Exception:
        return None


def apply_ngram_tokens_to_tokenizer(tokenizer, state: Optional[NgramState]) -> int:
    if state is None:
        return 0
    if not state.tokens:
        return 0
    added = tokenizer.add_tokens(list(state.tokens), special_tokens=False)
    return int(added)


def build_or_load_ngram_state(
    tokenizer,
    cfg: NgramConfig,
    outdir: Path,
    rebuild: bool = False,
) -> Optional[NgramState]:
    if not bool(cfg.use_ngrams):
        return None

    outdir = Path(outdir).expanduser().resolve()
    state_path = outdir / "ngram_state.json"

    if state_path.exists() and not rebuild:
        state = load_ngram_state(state_path)
        if state is not None:
            apply_ngram_tokens_to_tokenizer(tokenizer, state)
            return state

    state = build_ngram_state(tokenizer, cfg)
    save_ngram_state(state, outdir)
    return state


def ngram_summary_text(state: Optional[NgramState]) -> str:
    if state is None:
        return "N-Gramme deaktiviert."
    stats = state.stats or {}
    return (
        f"N-Gramme aktiv | selected={stats.get('selected_count', 0)} "
        f"candidates={stats.get('candidate_count', 0)} "
        f"scanned_samples={stats.get('scanned_samples', 0)} "
        f"estimated_savings={stats.get('estimated_total_savings', 0.0):.2f}"
    )
