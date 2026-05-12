"""Phase C language substrate harness (shadow-only).

Implements a deterministic, reset-safe training lane for a bounded adapter-style
student model using grounded prompt/response pairs from the language corpus.
The model never drives live user-facing responses.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
import hashlib
import json
import logging
from pathlib import Path
import re
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

JARVIS_DIR = Path.home() / ".jarvis"
LANGUAGE_CORPUS_DIR = JARVIS_DIR / "language_corpus"
PHASEC_DIR = LANGUAGE_CORPUS_DIR / "phase_c"

BASELINE_LOCK_PATH = PHASEC_DIR / "baseline_lock.json"
TOKENIZER_STRATEGY_PATH = PHASEC_DIR / "tokenizer_strategy.json"
DATASET_PATH = PHASEC_DIR / "dataset.jsonl"
DATASET_MANIFEST_PATH = PHASEC_DIR / "dataset_manifest.json"
SPLIT_MANIFEST_PATH = PHASEC_DIR / "split_manifest.json"
CHECKPOINT_PATH = PHASEC_DIR / "student_checkpoint.json"
TRAIN_RUNS_PATH = PHASEC_DIR / "train_runs.jsonl"

CORPUS_PATH = LANGUAGE_CORPUS_DIR / "examples.jsonl"
CORPUS_ROTATED_PATH = LANGUAGE_CORPUS_DIR / "examples.jsonl.1"

MIN_TRAIN_SAMPLES = 30
MAX_RECENT_RUNS = 20
MODEL_SCHEMA_VERSION = 1
PHASEC_SHADOW_ONLY = True

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[^\w\s]")

_BOUNDED_CLASSES = (
    "self_status",
    "self_introspection",
    "memory_recall",
    "recent_learning",
    "recent_research",
    "identity_answer",
    "capability_status",
)

_PHASEC_BASELINE = {
    "baseline_version": "phasec_v1",
    "source_todo": "Priority 2 / Phase C Language Substrate",
    "checklist": [
        "Choose tokenizer / vocab strategy",
        "Define first generative objective",
        "Create training harness",
        "Start bounded generator or adapter-based student",
        "Run shadow inference only",
    ],
    "wired_components": {
        "language_corpus_capture": True,
        "language_quality_telemetry": True,
        "shadow_style_model": True,
        "eval_gate_scoring": True,
        "dashboard_language_panel": True,
    },
}


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def _json_load(path: Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
    if not path.exists():
        return dict(default or {})
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            return raw
    except Exception:
        logger.debug("Phase C JSON load failed for %s", path, exc_info=True)
    return dict(default or {})


def _iter_corpus_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in (CORPUS_ROTATED_PATH, CORPUS_PATH):
        if not path.exists():
            continue
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    if isinstance(rec, dict):
                        records.append(rec)
        except Exception:
            logger.debug("Phase C corpus read failed for %s", path, exc_info=True)
    return records


def _tokens_bpe_like(text: str) -> list[str]:
    toks = _TOKEN_RE.findall((text or "").strip().lower())
    if not toks:
        return []
    out: list[str] = []
    for tok in toks:
        if tok.isalnum() and len(tok) > 12:
            out.extend([tok[i:i + 4] for i in range(0, len(tok), 4)])
        else:
            out.append(tok)
    return out


def _tokens_sentencepiece_like(text: str) -> list[str]:
    # Fallback approximation only used for deterministic local comparison.
    raw = (text or "").strip().lower()
    if not raw:
        return []
    pieces = []
    for word in re.findall(r"\S+", raw):
        if len(word) <= 6:
            pieces.append(word)
            continue
        head = word[:3]
        tail = word[3:]
        pieces.append(head)
        pieces.extend([f"##{tail[i:i + 3]}" for i in range(0, len(tail), 3)])
    return pieces


def _score_tokenizer(texts: list[str], mode: str) -> dict[str, Any]:
    if not texts:
        return {
            "available": True,
            "sample_count": 0,
            "avg_tokens_per_sample": 0.0,
            "avg_chars_per_token": 0.0,
            "punctuation_robustness": 0.0,
            "score": 0.0,
        }
    tokenize = _tokens_bpe_like if mode == "bpe" else _tokens_sentencepiece_like
    token_counts = []
    chars_per_token = []
    punct_hits = 0
    punct_total = 0
    for text in texts:
        tokens = tokenize(text)
        if not tokens:
            continue
        token_counts.append(len(tokens))
        chars_per_token.append(len(text) / max(1, len(tokens)))
        punct_total += len(re.findall(r"[.,;:!?]", text))
        punct_hits += sum(1 for t in tokens if re.search(r"[.,;:!?]", t))
    sample_count = len(token_counts)
    if sample_count == 0:
        return {
            "available": True,
            "sample_count": 0,
            "avg_tokens_per_sample": 0.0,
            "avg_chars_per_token": 0.0,
            "punctuation_robustness": 0.0,
            "score": 0.0,
        }
    avg_tokens = sum(token_counts) / sample_count
    avg_cpt = sum(chars_per_token) / sample_count
    punct_rob = 1.0 if punct_total == 0 else min(1.0, punct_hits / max(1, punct_total))
    # Lower avg_tokens and higher chars/token are preferred; punctuation robustness helps STT noise.
    score = max(0.0, (avg_cpt / 8.0) * 0.55 + punct_rob * 0.45 - (avg_tokens / 100.0))
    return {
        "available": True,
        "sample_count": sample_count,
        "avg_tokens_per_sample": round(avg_tokens, 3),
        "avg_chars_per_token": round(avg_cpt, 3),
        "punctuation_robustness": round(punct_rob, 3),
        "score": round(score, 4),
    }


def _extract_text_candidates(records: list[dict[str, Any]], *, limit: int = 500) -> list[str]:
    texts: list[str] = []
    for rec in records:
        q = str(rec.get("query", "") or "")
        fa = str(rec.get("final_answer", "") or "")
        mf = rec.get("meaning_frame") if isinstance(rec.get("meaning_frame"), dict) else {}
        lead = str(mf.get("lead", "") or "")
        if q:
            texts.append(q)
        if lead:
            texts.append(lead)
        if fa:
            texts.append(fa)
        if len(texts) >= limit:
            break
    return texts[:limit]


def evaluate_tokenizer_strategy(records: list[dict[str, Any]]) -> dict[str, Any]:
    texts = _extract_text_candidates(records)
    sp_available = False
    try:
        import sentencepiece  # type: ignore  # noqa: F401
        sp_available = True
    except Exception:
        sp_available = False

    candidates: dict[str, Any] = {
        "bpe": _score_tokenizer(texts, "bpe"),
        "sentencepiece": _score_tokenizer(texts, "sentencepiece"),
    }
    if not sp_available:
        candidates["sentencepiece"]["available"] = False
        candidates["sentencepiece"]["unavailable_reason"] = "sentencepiece_dependency_missing"
        candidates["sentencepiece"]["score"] = 0.0

    chosen = "bpe"
    reason = "bpe_default"
    if candidates["sentencepiece"].get("available"):
        bpe_score = float(candidates["bpe"].get("score", 0.0) or 0.0)
        sp_score = float(candidates["sentencepiece"].get("score", 0.0) or 0.0)
        if sp_score > bpe_score:
            chosen = "sentencepiece"
            reason = "sentencepiece_higher_score"
        else:
            chosen = "bpe"
            reason = "bpe_higher_or_equal_score"
    elif texts:
        reason = "sentencepiece_unavailable_bpe_selected"
    else:
        reason = "empty_corpus_bpe_default"

    vocab_tokens = Counter()
    tokenize = _tokens_bpe_like if chosen == "bpe" else _tokens_sentencepiece_like
    for text in texts:
        vocab_tokens.update(tokenize(text))

    result = {
        "strategy": chosen,
        "decision_reason": reason,
        "evaluated_at": time.time(),
        "sample_count": len(texts),
        "estimated_vocab_size": len(vocab_tokens),
        "candidates": candidates,
    }
    _json_dump(TOKENIZER_STRATEGY_PATH, result)
    return result


def _format_prompt(record: dict[str, Any]) -> str:
    response_class = str(record.get("response_class", "") or "unknown")
    route = str(record.get("route", "") or "unknown")
    query = str(record.get("query", "") or "").strip()
    mf = record.get("meaning_frame") if isinstance(record.get("meaning_frame"), dict) else {}
    lead = str(mf.get("lead", "") or "").strip()
    facts = mf.get("facts", []) if isinstance(mf.get("facts", []), list) else []
    facts_text = " | ".join(str(f)[:80] for f in facts[:5])
    lines = [
        f"CLASS:{response_class}",
        f"ROUTE:{route}",
        f"QUERY:{query}",
    ]
    if lead:
        lines.append(f"LEAD:{lead}")
    if facts_text:
        lines.append(f"FACTS:{facts_text}")
    lines.append("ANSWER:")
    return "\n".join(lines).strip()


def _sample_id(conversation_id: str, query: str, final_answer: str) -> str:
    key = f"{conversation_id}|{query}|{final_answer}".encode("utf-8", errors="ignore")
    return hashlib.sha1(key).hexdigest()[:16]


def build_dataset(records: list[dict[str, Any]]) -> dict[str, Any]:
    PHASEC_DIR.mkdir(parents=True, exist_ok=True)
    trainable_classes = set(_BOUNDED_CLASSES)
    rows: list[dict[str, Any]] = []
    skipped_negative = 0
    skipped_empty = 0
    skipped_low_conf = 0
    class_counts = Counter()
    prov_counts = Counter()
    for rec in records:
        response_class = str(rec.get("response_class", "") or "unknown")
        provenance = str(rec.get("provenance_verdict", "") or "unknown")
        if response_class == "negative_example" or provenance.startswith("negative:"):
            skipped_negative += 1
            continue
        confidence = float(rec.get("confidence", 0.0) or 0.0)
        if confidence <= 0.0:
            skipped_low_conf += 1
            continue
        query = str(rec.get("query", "") or "").strip()
        final_answer = str(rec.get("final_answer", "") or "").strip()
        if not query or not final_answer:
            skipped_empty += 1
            continue
        conversation_id = str(rec.get("conversation_id", "") or "")
        sid = _sample_id(conversation_id, query, final_answer)
        prompt = _format_prompt(rec)
        row = {
            "sample_id": sid,
            "conversation_id": conversation_id,
            "response_class": response_class,
            "trainable_response_class": response_class in trainable_classes,
            "route": str(rec.get("route", "") or "unknown"),
            "prompt": prompt[:2000],
            "target": final_answer[:2000],
            "objective": "next_token_grounded_pair",
            "provenance_verdict": provenance,
            "source_timestamp": float(rec.get("timestamp", 0.0) or 0.0),
        }
        rows.append(row)
        class_counts[response_class] += 1
        prov_counts[provenance] += 1

    rows.sort(key=lambda r: r["sample_id"])
    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    digest = hashlib.sha1()
    for row in rows:
        digest.update(f"{row['sample_id']}|{row['prompt']}|{row['target']}".encode("utf-8", errors="ignore"))
    dataset_hash = digest.hexdigest()

    manifest = {
        "schema_version": 1,
        "objective": "next_token_grounded_pair",
        "dataset_path": str(DATASET_PATH),
        "dataset_hash": dataset_hash,
        "total_samples": len(rows),
        "counts_by_response_class": dict(class_counts),
        "counts_by_provenance": dict(prov_counts),
        "skipped_negative": skipped_negative,
        "skipped_empty": skipped_empty,
        "skipped_low_confidence": skipped_low_conf,
        "built_at": time.time(),
    }
    _json_dump(DATASET_MANIFEST_PATH, manifest)
    return manifest


def _deterministic_split(sample_id: str, train_pct: int = 90) -> str:
    h = hashlib.sha1(sample_id.encode("utf-8", errors="ignore")).hexdigest()
    bucket = int(h[:8], 16) % 100
    return "train" if bucket < train_pct else "val"


def build_split_manifest() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    if DATASET_PATH.exists():
        with open(DATASET_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if isinstance(rec, dict):
                    rows.append(rec)

    train_count = 0
    val_count = 0
    by_class = defaultdict(lambda: {"train": 0, "val": 0})
    split_digest = hashlib.sha1()
    for row in rows:
        sid = str(row.get("sample_id", "") or "")
        split = _deterministic_split(sid)
        row["split"] = split
        response_class = str(row.get("response_class", "") or "unknown")
        by_class[response_class][split] += 1
        if split == "train":
            train_count += 1
        else:
            val_count += 1
        split_digest.update(f"{sid}:{split}".encode("utf-8", errors="ignore"))

    # Rewrite dataset with split labels to keep contract explicit.
    if rows:
        with open(DATASET_PATH, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=True) + "\n")

    manifest = {
        "schema_version": 1,
        "dataset_hash": _json_load(DATASET_MANIFEST_PATH).get("dataset_hash", ""),
        "split_hash": split_digest.hexdigest(),
        "train_count": train_count,
        "val_count": val_count,
        "counts_by_response_class": dict(by_class),
        "built_at": time.time(),
        "split_rule": "sha1(sample_id)%100 < 90 => train else val",
    }
    _json_dump(SPLIT_MANIFEST_PATH, manifest)
    return manifest


class PhaseCAdapterStudent:
    """Simple bigram adapter student for shadow-only language experiments."""

    def __init__(self) -> None:
        self._bigram: dict[str, Counter[str]] = defaultdict(Counter)
        self._unigram: Counter[str] = Counter()
        self._vocab: set[str] = set()
        self._available = False
        self._reason = "not_trained"
        self._trained_at = 0.0
        self._last_val_perplexity = 0.0
        self._epochs = 0
        self._tokenizer_strategy = ""
        self._dataset_hash = ""
        self._lock = threading.Lock()
        self._recent_generations: deque[dict[str, Any]] = deque(maxlen=20)

    @property
    def available(self) -> bool:
        return self._available

    def get_status(self) -> dict[str, Any]:
        return {
            "available": self._available,
            "reason": self._reason,
            "trained_at": self._trained_at,
            "epochs": self._epochs,
            "vocab_size": len(self._vocab),
            "tokenizer_strategy": self._tokenizer_strategy,
            "dataset_hash": self._dataset_hash,
            "val_perplexity": round(self._last_val_perplexity, 4),
            "checkpoint_exists": CHECKPOINT_PATH.exists(),
            "checkpoint_age_s": max(0.0, time.time() - self._trained_at) if self._trained_at else 0.0,
            "recent_generations": list(self._recent_generations),
            "shadow_only_enforced": PHASEC_SHADOW_ONLY,
            "live_routing_enabled": not PHASEC_SHADOW_ONLY,
        }

    def _tokenize(self, text: str) -> list[str]:
        if self._tokenizer_strategy == "sentencepiece":
            return _tokens_sentencepiece_like(text)
        return _tokens_bpe_like(text)

    def _detokenize(self, toks: list[str]) -> str:
        if not toks:
            return ""
        out = []
        for tok in toks:
            if re.match(r"^[.,;:!?]$", tok):
                if out:
                    out[-1] = out[-1] + tok
                else:
                    out.append(tok)
            elif tok.startswith("##") and out:
                out[-1] = out[-1] + tok[2:]
            else:
                out.append(tok)
        return " ".join(out).strip()

    def _seq_tokens(self, prompt: str, target: str) -> list[str]:
        seq = self._tokenize(prompt) + ["<ANS>"] + self._tokenize(target) + ["<EOS>"]
        return [t for t in seq if t]

    def _update_counts(self, seq: list[str]) -> None:
        if not seq:
            return
        self._vocab.update(seq)
        self._unigram.update(seq)
        prev = "<BOS>"
        for tok in seq:
            self._bigram[prev][tok] += 1
            prev = tok

    def _prob(self, prev: str, tok: str) -> float:
        # Add-one smoothing over observed vocabulary.
        vocab = max(1, len(self._vocab))
        next_counts = self._bigram.get(prev)
        if not next_counts:
            return 1.0 / vocab
        num = float(next_counts.get(tok, 0) + 1)
        den = float(sum(next_counts.values()) + vocab)
        return num / den if den > 0 else 1.0 / vocab

    def _val_perplexity(self, rows: list[dict[str, Any]]) -> float:
        import math

        total_nll = 0.0
        total_tokens = 0
        for row in rows:
            seq = self._seq_tokens(str(row.get("prompt", "")), str(row.get("target", "")))
            prev = "<BOS>"
            for tok in seq:
                p = self._prob(prev, tok)
                total_nll += -math.log(max(p, 1e-12))
                total_tokens += 1
                prev = tok
        if total_tokens == 0:
            return 0.0
        return float(math.exp(total_nll / total_tokens))

    def train(
        self,
        rows: list[dict[str, Any]],
        *,
        tokenizer_strategy: str,
        dataset_hash: str,
        max_epochs: int = 1,
    ) -> dict[str, Any]:
        with self._lock:
            train_rows = [r for r in rows if r.get("split") == "train"]
            val_rows = [r for r in rows if r.get("split") == "val"]
            if len(train_rows) < MIN_TRAIN_SAMPLES:
                self._available = False
                self._reason = f"insufficient_train_samples:{len(train_rows)}<{MIN_TRAIN_SAMPLES}"
                return {
                    "trained": False,
                    "reason": self._reason,
                    "train_count": len(train_rows),
                    "val_count": len(val_rows),
                }

            self._tokenizer_strategy = tokenizer_strategy or "bpe"
            self._dataset_hash = dataset_hash
            self._bigram = defaultdict(Counter)
            self._unigram = Counter()
            self._vocab = set()

            for _ in range(max(1, int(max_epochs))):
                for row in train_rows:
                    seq = self._seq_tokens(str(row.get("prompt", "")), str(row.get("target", "")))
                    self._update_counts(seq)

            self._last_val_perplexity = self._val_perplexity(val_rows)
            self._epochs = max(1, int(max_epochs))
            self._available = True
            self._reason = "trained"
            self._trained_at = time.time()

            return {
                "trained": True,
                "train_count": len(train_rows),
                "val_count": len(val_rows),
                "epochs": self._epochs,
                "vocab_size": len(self._vocab),
                "val_perplexity": self._last_val_perplexity,
            }

    def generate_shadow(
        self,
        *,
        query: str,
        response_class: str,
        prompt: str,
        max_tokens: int = 48,
    ) -> str | None:
        with self._lock:
            if not self._available:
                return None
            if response_class not in _BOUNDED_CLASSES:
                return None
            toks = self._tokenize(prompt)
            prev = toks[-1] if toks else "<BOS>"
            out: list[str] = []
            for _ in range(max(8, max_tokens)):
                next_counts = self._bigram.get(prev)
                if not next_counts:
                    break
                tok = next_counts.most_common(1)[0][0]
                if tok in {"<EOS>", "<ANS>"}:
                    break
                out.append(tok)
                prev = tok
            text = self._detokenize(out).strip()
            if not text:
                return None
            self._recent_generations.append({
                "ts": time.time(),
                "response_class": response_class,
                "query": (query or "")[:120],
                "reply": text[:220],
            })
            return text

    def save_checkpoint(self) -> bool:
        with self._lock:
            if not self._trained_at:
                return False
            payload = {
                "schema_version": MODEL_SCHEMA_VERSION,
                "trained_at": self._trained_at,
                "epochs": self._epochs,
                "tokenizer_strategy": self._tokenizer_strategy,
                "dataset_hash": self._dataset_hash,
                "val_perplexity": self._last_val_perplexity,
                "unigram": dict(self._unigram),
                "bigram": {k: dict(v) for k, v in self._bigram.items()},
                "vocab": sorted(self._vocab),
            }
            try:
                _json_dump(CHECKPOINT_PATH, payload)
                return True
            except Exception:
                logger.warning("Phase C checkpoint save failed", exc_info=True)
                return False

    def load_checkpoint(self) -> bool:
        payload = _json_load(CHECKPOINT_PATH)
        if not payload:
            self._available = False
            self._reason = "checkpoint_missing"
            return False
        if int(payload.get("schema_version", 0) or 0) != MODEL_SCHEMA_VERSION:
            self._available = False
            self._reason = "checkpoint_schema_mismatch"
            return False
        try:
            with self._lock:
                self._trained_at = float(payload.get("trained_at", 0.0) or 0.0)
                self._epochs = int(payload.get("epochs", 0) or 0)
                self._tokenizer_strategy = str(payload.get("tokenizer_strategy", "") or "bpe")
                self._dataset_hash = str(payload.get("dataset_hash", "") or "")
                self._last_val_perplexity = float(payload.get("val_perplexity", 0.0) or 0.0)
                self._unigram = Counter(payload.get("unigram", {}))
                self._bigram = defaultdict(Counter)
                for k, v in (payload.get("bigram", {}) or {}).items():
                    if isinstance(v, dict):
                        self._bigram[str(k)] = Counter({str(x): int(y) for x, y in v.items()})
                self._vocab = set(str(x) for x in (payload.get("vocab", []) or []))
                self._available = bool(self._vocab) and self._trained_at > 0
                self._reason = "trained" if self._available else "checkpoint_invalid"
            return self._available
        except Exception:
            logger.warning("Phase C checkpoint load failed", exc_info=True)
            self._available = False
            self._reason = "checkpoint_corrupt"
            return False


class PhaseCHarness:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last_run: dict[str, Any] = {}

    def _rows_from_dataset(self) -> list[dict[str, Any]]:
        rows = []
        if not DATASET_PATH.exists():
            return rows
        with open(DATASET_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if isinstance(rec, dict):
                    rows.append(rec)
        return rows

    def run_training_cycle(self) -> dict[str, Any]:
        if not self._lock.acquire(blocking=False):
            return {"ok": False, "status": "busy"}
        try:
            PHASEC_DIR.mkdir(parents=True, exist_ok=True)
            baseline = lock_phasec_baseline()
            records = _iter_corpus_records()
            tokenizer = evaluate_tokenizer_strategy(records)
            dataset_manifest = build_dataset(records)
            split_manifest = build_split_manifest()
            rows = self._rows_from_dataset()
            train_result = phasec_shadow_student.train(
                rows,
                tokenizer_strategy=str(tokenizer.get("strategy", "bpe")),
                dataset_hash=str(dataset_manifest.get("dataset_hash", "")),
                max_epochs=1,
            )
            if train_result.get("trained"):
                phasec_shadow_student.save_checkpoint()
            run = {
                "ts": time.time(),
                "status": "trained" if train_result.get("trained") else "deferred",
                "baseline_version": baseline.get("baseline_version", "phasec_v1"),
                "tokenizer_strategy": tokenizer.get("strategy", "bpe"),
                "dataset_samples": dataset_manifest.get("total_samples", 0),
                "train_count": split_manifest.get("train_count", 0),
                "val_count": split_manifest.get("val_count", 0),
                "result": train_result,
            }
            with open(TRAIN_RUNS_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(run, ensure_ascii=True) + "\n")
            self._last_run = run
            return {"ok": True, **run}
        except Exception:
            logger.warning("Phase C harness cycle failed", exc_info=True)
            return {"ok": False, "status": "error"}
        finally:
            self._lock.release()

    def get_status(self) -> dict[str, Any]:
        baseline = _json_load(BASELINE_LOCK_PATH, default={**_PHASEC_BASELINE, "locked_at": 0.0})
        tokenizer = _json_load(TOKENIZER_STRATEGY_PATH, default={})
        dataset = _json_load(DATASET_MANIFEST_PATH, default={})
        split = _json_load(SPLIT_MANIFEST_PATH, default={})
        runs: list[dict[str, Any]] = []
        if TRAIN_RUNS_PATH.exists():
            try:
                with open(TRAIN_RUNS_PATH, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except Exception:
                            continue
                        if isinstance(rec, dict):
                            runs.append(rec)
            except Exception:
                logger.debug("Phase C runs read failed", exc_info=True)
        runs = runs[-MAX_RECENT_RUNS:]
        total_samples = int(dataset.get("total_samples", 0) or 0)
        train_samples = int(split.get("train_count", 0) or 0)
        return {
            "baseline": baseline,
            "tokenizer": tokenizer,
            "dataset": dataset,
            "split": split,
            "student": phasec_shadow_student.get_status(),
            "recent_runs": runs,
            "last_run": self._last_run or (runs[-1] if runs else {}),
            "shadow_only_enforced": PHASEC_SHADOW_ONLY,
            "live_routing_enabled": not PHASEC_SHADOW_ONLY,
            "minimum_train_samples": MIN_TRAIN_SAMPLES,
            "reset_aware_context": {
                "train_samples_observed": train_samples,
                "train_samples_required": MIN_TRAIN_SAMPLES,
                "dataset_samples_observed": total_samples,
                "below_training_threshold": train_samples < MIN_TRAIN_SAMPLES,
            },
        }


def lock_phasec_baseline() -> dict[str, Any]:
    existing = _json_load(BASELINE_LOCK_PATH, default={})
    if existing.get("baseline_version") == _PHASEC_BASELINE["baseline_version"]:
        return existing
    payload = {
        **_PHASEC_BASELINE,
        "locked_at": time.time(),
        "gaps_initial": [
            "tokenizer_strategy_undecided",
            "objective_manifest_missing",
            "checkpoint_resume_uninitialized",
            "phasec_shadow_student_untrained",
        ],
    }
    _json_dump(BASELINE_LOCK_PATH, payload)
    return payload


def get_phasec_status() -> dict[str, Any]:
    return phasec_harness.get_status()


def is_live_routing_enabled() -> bool:
    """Explicit runtime guard for shadow-only Phase C."""
    return not PHASEC_SHADOW_ONLY


phasec_shadow_student = PhaseCAdapterStudent()
phasec_harness = PhaseCHarness()

try:
    phasec_shadow_student.load_checkpoint()
except Exception:
    logger.debug("Phase C student checkpoint warm-load failed", exc_info=True)

