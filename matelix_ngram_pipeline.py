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
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

csv.field_size_limit(1024 * 1024 * 128)

NGRAM_TOKEN_PREFIX = "<|ng:"
NGRAM_TOKEN_SUFFIX = "|>"

_WORD_RE = re.compile(r"\S+")

_RE_DEF = re.compile(r"^\s*def\s+[A-Za-z_][A-Za-z0-9_]*\s*\(")
_RE_CLASS = re.compile(r"^\s*class\s+[A-Za-z_][A-Za-z0-9_]*\s*[:(]")
_RE_RETURN = re.compile(r"^\s*return\b")
_RE_IF = re.compile(r"^\s*if\b.+:\s*$")
_RE_ELIF = re.compile(r"^\s*elif\b.+:\s*$")
_RE_ELSE = re.compile(r"^\s*else\s*:\s*$")
_RE_FOR = re.compile(r"^\s*for\b.+:\s*$")
_RE_WHILE = re.compile(r"^\s*while\b.+:\s*$")
_RE_TRY = re.compile(r"^\s*try\s*:\s*$")
_RE_EXCEPT = re.compile(r"^\s*except\b.*:\s*$")
_RE_IMPORT = re.compile(r"^\s*import\b")
_RE_FROM_IMPORT = re.compile(r"^\s*from\b.+\bimport\b")
_RE_ASSIGN = re.compile(r"^\s*[A-Za-z_][A-Za-z0-9_,\s]*=\s*.+$")
_RE_CALL = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\s*\(")
_RE_JSONISH = re.compile(r'^\s*["\']?[A-Za-z0-9_\- ]+["\']?\s*:\s*')
_RE_SYMBOL_HEAVY = re.compile(r"[{}()[\]=<>:+\-*/%,.;]")


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

    # Overlap / Konsistenz
    ngram_conflict_overlap_ratio: float = 0.8
    ngram_prefer_longest_match: bool = True

    # Code-Lines
    ngram_use_code_lines: bool = True
    ngram_code_line_min_chars: int = 12
    ngram_code_line_min_count: int = 2
    ngram_code_line_top_k: int = 400
    ngram_code_line_score_boost: float = 1.35

    # Code-Muster
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


@dataclass
class NgramStats:
    scanned_samples: int
    selected_count: int
    candidate_count: int
    estimated_total_savings: float
    code_line_candidate_count: int = 0
    selected_code_line_count: int = 0
    oversize_candidates: int = 0
    near_limit_candidates: int = 0
    estimated_saved_samples: int = 0


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


def _tokenize_words(text: str) -> List[str]:
    return _WORD_RE.findall(text or "")


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _normalize_code_line(line: str) -> str:
    line = line.rstrip()
    line = line.replace("\t", "    ")
    line = re.sub(r"\s+$", "", line)
    return line


def _looks_like_code_line(line: str) -> bool:
    s = (line or "").rstrip()
    if not s:
        return False
    if len(s) < 4:
        return False
    if s.startswith("#") or s.startswith("//"):
        return True
    if _RE_SYMBOL_HEAVY.search(s):
        return True
    if _RE_DEF.search(s) or _RE_CLASS.search(s):
        return True
    if _RE_RETURN.search(s) or _RE_IF.search(s) or _RE_ELIF.search(s) or _RE_ELSE.search(s):
        return True
    if _RE_FOR.search(s) or _RE_WHILE.search(s) or _RE_TRY.search(s) or _RE_EXCEPT.search(s):
        return True
    if _RE_IMPORT.search(s) or _RE_FROM_IMPORT.search(s):
        return True
    if _RE_ASSIGN.search(s) or _RE_CALL.search(s):
        return True
    if _RE_JSONISH.search(s):
        return True
    return False


def _code_pattern_multiplier(line: str, cfg: NgramConfig) -> float:
    if not bool(cfg.ngram_code_pattern_boost):
        return 1.0

    s = (line or "").rstrip()
    mult = 1.0
    extra = max(1.0, float(cfg.ngram_code_pattern_extra_boost))

    if _RE_DEF.search(s):
        mult *= extra * 1.35
    elif _RE_CLASS.search(s):
        mult *= extra * 1.30
    elif _RE_RETURN.search(s):
        mult *= extra * 1.20
    elif _RE_IF.search(s) or _RE_ELIF.search(s) or _RE_ELSE.search(s):
        mult *= extra * 1.15
    elif _RE_FOR.search(s) or _RE_WHILE.search(s):
        mult *= extra * 1.15
    elif _RE_TRY.search(s) or _RE_EXCEPT.search(s):
        mult *= extra * 1.12
    elif _RE_IMPORT.search(s) or _RE_FROM_IMPORT.search(s):
        mult *= extra * 1.18
    elif _RE_ASSIGN.search(s):
        mult *= extra * 1.08
    elif _RE_JSONISH.search(s):
        mult *= extra * 1.10
    elif _RE_CALL.search(s):
        mult *= extra * 1.06

    return mult


def _is_candidate_valid(phrase: str, cfg: NgramConfig) -> bool:
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


def _is_code_line_candidate_valid(line: str, cfg: NgramConfig) -> bool:
    line = _normalize_code_line(line)
    if not line:
        return False
    if len(line) < int(cfg.ngram_code_line_min_chars):
        return False
    if len(line) > int(cfg.ngram_max_token_chars):
        return False
    if not _looks_like_code_line(line):
        return False
    if line.startswith(NGRAM_TOKEN_PREFIX) and line.endswith(NGRAM_TOKEN_SUFFIX):
        return False
    return True


def _sample_length(tokenizer, text: str) -> int:
    try:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])
    except Exception:
        return 0


def _estimate_phrase_token_gain(tokenizer, phrase: str) -> int:
    try:
        base_ids = tokenizer(phrase, add_special_tokens=False)["input_ids"]
    except Exception:
        return 0
    return max(0, len(base_ids) - 1)


def _occurrence_count_in_text(text: str, phrase: str) -> int:
    if not text or not phrase:
        return 0
    return text.count(phrase)


def _phrases_conflict(a: str, b: str, overlap_ratio: float = 0.8) -> bool:
    a_words = _tokenize_words(a)
    b_words = _tokenize_words(b)

    if not a_words or not b_words:
        return False

    a_join = " ".join(a_words)
    b_join = " ".join(b_words)

    if a_join == b_join:
        return True

    if a_join in b_join or b_join in a_join:
        return True

    a_set = set(a_words)
    b_set = set(b_words)
    overlap = len(a_set & b_set)
    denom = max(1, min(len(a_set), len(b_set)))
    return (overlap / denom) >= float(overlap_ratio)


def collect_ngram_candidates(
    tokenizer,
    cfg: NgramConfig,
) -> Tuple[Counter, Counter, int, Dict[str, Dict[str, int]]]:
    phrase_counter: Counter = Counter()
    code_line_counter: Counter = Counter()
    phrase_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {
        "count": 0,
        "oversize_hits": 0,
        "near_limit_hits": 0,
        "saved_samples": 0,
    })

    scanned_samples = 0
    max_n = max(2, int(cfg.ngram_max))
    max_seq_length = max(1, int(cfg.ngram_eval_max_seq_length))
    near_limit_threshold = min(1.0, max(0.0, float(cfg.ngram_near_limit_threshold)))

    for text in iter_training_texts(cfg.csv_path, cfg.template_mode, cfg.column_name):
        scanned_samples += 1
        if scanned_samples > int(cfg.ngram_max_samples):
            break

        seq_len = _sample_length(tokenizer, text)
        is_oversize = seq_len > max_seq_length
        is_near_limit = seq_len >= int(max_seq_length * near_limit_threshold)

        words = _tokenize_words(text)
        seen_in_sample = set()

        if len(words) >= 2:
            limit = min(len(words), int(cfg.ngram_max_tokens_per_text))
            words = words[:limit]

            for n in range(2, min(max_n, len(words)) + 1):
                for i in range(0, len(words) - n + 1):
                    phrase = " ".join(words[i:i + n]).strip()
                    phrase = _normalize_space(phrase)
                    if not _is_candidate_valid(phrase, cfg):
                        continue

                    phrase_counter[phrase] += 1
                    phrase_stats[phrase]["count"] += 1

                    if phrase not in seen_in_sample:
                        if is_oversize:
                            phrase_stats[phrase]["oversize_hits"] += 1
                        if is_near_limit:
                            phrase_stats[phrase]["near_limit_hits"] += 1
                        seen_in_sample.add(phrase)

        if bool(cfg.ngram_use_code_lines):
            for raw_line in text.splitlines():
                line = _normalize_code_line(raw_line)
                if not _is_code_line_candidate_valid(line, cfg):
                    continue

                code_line_counter[line] += 1
                phrase_stats[line]["count"] += 1

                if line not in seen_in_sample:
                    if is_oversize:
                        phrase_stats[line]["oversize_hits"] += 1
                    if is_near_limit:
                        phrase_stats[line]["near_limit_hits"] += 1
                    seen_in_sample.add(line)

        if bool(cfg.ngram_track_saved_samples) and seq_len > 0:
            for phrase in seen_in_sample:
                gain = _estimate_phrase_token_gain(tokenizer, phrase)
                if gain <= 0:
                    continue
                occurrences = _occurrence_count_in_text(text, phrase)
                saved_tokens = gain * max(1, occurrences)

                if is_oversize and (seq_len - saved_tokens) <= max_seq_length:
                    phrase_stats[phrase]["saved_samples"] += 1

    if int(cfg.ngram_min_count) > 1:
        phrase_counter = Counter({k: v for k, v in phrase_counter.items() if v >= int(cfg.ngram_min_count)})

    if int(cfg.ngram_code_line_min_count) > 1:
        code_line_counter = Counter({k: v for k, v in code_line_counter.items() if v >= int(cfg.ngram_code_line_min_count)})

    # phrase_stats auf gültige Kandidaten begrenzen
    valid_keys = set(phrase_counter.keys()) | set(code_line_counter.keys())
    phrase_stats = {k: v for k, v in phrase_stats.items() if k in valid_keys}

    return phrase_counter, code_line_counter, scanned_samples, phrase_stats


def _estimate_phrase_score(
    tokenizer,
    phrase: str,
    stats: Dict[str, int],
    cfg: NgramConfig,
    *,
    is_code_line: bool = False,
) -> float:
    count = int(stats.get("count", 0))
    base_score = float(_estimate_phrase_token_gain(tokenizer, phrase) * max(0, count - 1))
    if base_score <= 0:
        return 0.0

    score = base_score

    if bool(cfg.ngram_focus_oversize_samples):
        score += float(stats.get("oversize_hits", 0)) * float(cfg.ngram_oversize_sample_boost)
        score += float(stats.get("near_limit_hits", 0)) * float(cfg.ngram_near_limit_boost)

    if bool(cfg.ngram_track_saved_samples):
        score += float(stats.get("saved_samples", 0)) * float(cfg.ngram_saved_sample_boost)

    if is_code_line:
        score *= float(cfg.ngram_code_line_score_boost)
        score *= _code_pattern_multiplier(phrase, cfg)

    return score


def select_best_ngrams(
    tokenizer,
    cfg: NgramConfig,
    phrase_counter: Counter,
    code_line_counter: Optional[Counter],
    phrase_stats: Dict[str, Dict[str, int]],
) -> Tuple[List[str], Dict[str, float], int, int]:
    scored: List[Tuple[str, float, str]] = []
    score_map: Dict[str, float] = {}
    selected_code_line_count = 0
    estimated_saved_samples = 0

    for phrase in phrase_counter.keys():
        stats = phrase_stats.get(phrase, {})
        score = _estimate_phrase_score(tokenizer, phrase, stats, cfg, is_code_line=False)
        if score > 0:
            scored.append((phrase, score, "phrase"))
            score_map[phrase] = score

    code_line_counter = code_line_counter or Counter()
    if code_line_counter:
        top_code_k = max(1, int(cfg.ngram_code_line_top_k))
        ranked_code_lines = sorted(
            code_line_counter.items(),
            key=lambda x: (-x[1], -len(x[0]), x[0]),
        )[:top_code_k]

        for line, _count in ranked_code_lines:
            stats = phrase_stats.get(line, {})
            score = _estimate_phrase_score(tokenizer, line, stats, cfg, is_code_line=True)
            if score > 0:
                scored.append((line, score, "code"))
                score_map[line] = score

    if bool(cfg.ngram_prefer_longest_match):
        scored.sort(key=lambda x: (-x[1], -len(x[0]), -len(_tokenize_words(x[0])), x[0]))
    else:
        scored.sort(key=lambda x: (-x[1], x[0]))

    if not scored:
        return [], score_map, selected_code_line_count, estimated_saved_samples

    top_k = max(1, int(cfg.ngram_top_k))
    overlap_ratio = float(cfg.ngram_conflict_overlap_ratio)

    def conflicts_with_selected(candidate: str, selected: List[str]) -> bool:
        for chosen in selected:
            if _phrases_conflict(candidate, chosen, overlap_ratio=overlap_ratio):
                return True
        return False

    selected: List[str] = []

    if not bool(cfg.ngram_budgeted):
        for phrase, _score, kind in scored:
            if len(selected) >= top_k:
                break
            if conflicts_with_selected(phrase, selected):
                continue
            selected.append(phrase)
            estimated_saved_samples += int(phrase_stats.get(phrase, {}).get("saved_samples", 0))
            if kind == "code":
                selected_code_line_count += 1
        return selected, score_map, selected_code_line_count, estimated_saved_samples

    total_score = sum(score for _, score, _ in scored)
    if total_score <= 0:
        return [], score_map, selected_code_line_count, estimated_saved_samples

    target_fit = float(cfg.ngram_target_fit)
    target_fit = min(max(target_fit, 0.0), 1.0)
    target_score = total_score * target_fit

    running = 0.0
    for phrase, score, kind in scored:
        if len(selected) >= top_k:
            break
        if conflicts_with_selected(phrase, selected):
            continue
        selected.append(phrase)
        running += score
        estimated_saved_samples += int(phrase_stats.get(phrase, {}).get("saved_samples", 0))
        if kind == "code":
            selected_code_line_count += 1
        if running >= target_score:
            break

    return selected, score_map, selected_code_line_count, estimated_saved_samples


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


def build_ngram_state(tokenizer, cfg: NgramConfig) -> NgramState:
    phrase_counter, code_line_counter, scanned_samples, phrase_stats = collect_ngram_candidates(
        tokenizer=tokenizer,
        cfg=cfg,
    )

    selected_phrases, score_map, selected_code_line_count, estimated_saved_samples = select_best_ngrams(
        tokenizer=tokenizer,
        cfg=cfg,
        phrase_counter=phrase_counter,
        code_line_counter=code_line_counter,
        phrase_stats=phrase_stats,
    )

    tokens, phrase_to_token, _ = extend_tokenizer_with_ngrams(tokenizer, selected_phrases)

    estimated_total_savings = float(sum(score_map.get(p, 0.0) for p in selected_phrases))
    oversize_candidates = sum(1 for k, v in phrase_stats.items() if int(v.get("oversize_hits", 0)) > 0)
    near_limit_candidates = sum(1 for k, v in phrase_stats.items() if int(v.get("near_limit_hits", 0)) > 0)

    stats = NgramStats(
        scanned_samples=scanned_samples,
        selected_count=len(selected_phrases),
        candidate_count=len(phrase_counter) + len(code_line_counter),
        estimated_total_savings=estimated_total_savings,
        code_line_candidate_count=len(code_line_counter),
        selected_code_line_count=selected_code_line_count,
        oversize_candidates=oversize_candidates,
        near_limit_candidates=near_limit_candidates,
        estimated_saved_samples=estimated_saved_samples,
    )

    return NgramState(
        version=4,
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
        f"code_candidates={stats.get('code_line_candidate_count', 0)} "
        f"selected_code={stats.get('selected_code_line_count', 0)} "
        f"oversize_candidates={stats.get('oversize_candidates', 0)} "
        f"near_limit_candidates={stats.get('near_limit_candidates', 0)} "
        f"estimated_saved_samples={stats.get('estimated_saved_samples', 0)} "
        f"scanned_samples={stats.get('scanned_samples', 0)} "
        f"estimated_savings={stats.get('estimated_total_savings', 0.0):.2f}"
    )
