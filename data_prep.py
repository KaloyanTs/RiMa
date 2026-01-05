import os
import json
import logging
import re
from pathlib import Path
from statistics import mean
from typing import List, Tuple

try:
    import sentencepiece as spm  # optional
except Exception:
    spm = None

# Config
DATA_DIR = "chitanka_poems_step1"
OUT_DIR = Path("prepared")
CLEAN_FILE = OUT_DIR / "poems_clean.txt"
TRAIN_FILE = OUT_DIR / "train.txt"
VAL_FILE = OUT_DIR / "val.txt"
STATS_FILE = OUT_DIR / "stats.json"
SPM_INPUT = CLEAN_FILE  # train tokenizer on cleaned corpus
LOG_FILE = OUT_DIR / "data_prep.log"
VAL_FRACTION = 0.10

# Mirror tokenizer training settings used in the project
SPM_PREFIX = "poems_cond_bpe"
SPM_VOCAB_SIZE = 8000
SPM_CHAR_COVERAGE = 0.9995

OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
LOG = logging.getLogger("data_prep")


def _collapse_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _bad_latin_or_digits(s: str) -> bool:
    return re.search(r"[A-Za-z0-9]", s) is not None


def load_lines_from_dir(data_dir: str) -> List[str]:
    """
    Load and lightly clean poem lines from a directory of .txt files.
    - drop empty lines
    - drop lines containing Latin letters or digits (to keep Bulgarian only)
    - global deduplication (case-insensitive, whitespace-collapsed)
    """
    LOG.info(f"[data] scanning directory: {data_dir}")
    files = sorted(Path(data_dir).glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt files found under {data_dir}")

    LOG.info(f"[data] found {len(files)} files")
    lines: List[str] = []
    kept = 0
    dropped_empty = 0
    dropped_latin = 0
    for fp in files:
        txt = fp.read_text(encoding="utf-8", errors="ignore")
        for ln in txt.splitlines():
            ln = _collapse_ws(ln)
            if not ln:
                dropped_empty += 1
                continue
            if _bad_latin_or_digits(ln):
                dropped_latin += 1
                continue
            lines.append(ln)
            kept += 1
    # global dedup
    LOG.info("[data] applying global deduplication")
    seen = set()
    deduped = []
    for ln in lines:
        key = _collapse_ws(ln).lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ln)
    LOG.info(f"[data] dedup removed={len(lines)-len(deduped)}")
    lines = deduped
    kept = len(lines)
    LOG.info(f"[data] kept={kept} dropped_empty={dropped_empty} dropped_latin_or_digits={dropped_latin}")
    if not lines:
        raise RuntimeError("No usable lines after cleaning.")
    return lines


def split_train_val(items: List[str], val_fraction: float = 0.1, seed: int = 42) -> Tuple[List[str], List[str]]:
    LOG.info(f"[split] total={len(items)} val_fraction={val_fraction}")
    rng = __import__("random").Random(seed)
    idx = list(range(len(items)))
    rng.shuffle(idx)
    n_val = max(1, int(len(items) * val_fraction))
    val_idx = set(idx[:n_val])
    train = [items[i] for i in range(len(items)) if i not in val_idx]
    val = [items[i] for i in range(len(items)) if i in val_idx]
    LOG.info(f"[split] train={len(train)} val={len(val)}")
    return train, val


def _write_lines(path: Path, lines: List[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOG.info(f"[write] {path} lines={len(lines)}")


def _length_stats(lines: List[str]) -> dict:
    if not lines:
        return {"count": 0}
    lens = [len(s) for s in lines]
    lens_sorted = sorted(lens)
    n = len(lens_sorted)
    def pct(p):
        if n == 0:
            return 0
        i = min(n-1, max(0, int(p * (n-1))))
        return lens_sorted[i]
    return {
        "count": n,
        "avg_chars": round(mean(lens), 2),
        "p50": pct(0.50),
        "p90": pct(0.90),
        "p95": pct(0.95),
        "max": max(lens),
        "min": min(lens),
    }


def train_spm_on_clean_corpus(input_path: Path) -> Tuple[str, str]:
    if spm is None:
        LOG.info("[spm] sentencepiece not installed; skipping tokenizer training")
        return ("", "")
    if not input_path.exists():
        raise FileNotFoundError(f"SPM input not found: {input_path}")
    LOG.info(f"[spm] training on cleaned corpus: {input_path}")
    spm.SentencePieceTrainer.train(
        input=str(input_path),
        model_prefix=SPM_PREFIX,
        vocab_size=SPM_VOCAB_SIZE,
        model_type="bpe",
        character_coverage=SPM_CHAR_COVERAGE,
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
    )
    model_path = f"{SPM_PREFIX}.model"
    vocab_path = f"{SPM_PREFIX}.vocab"
    LOG.info(f"[spm] trained {model_path} and {vocab_path}")
    return (model_path, vocab_path)


def main():
    base = Path(DATA_DIR)
    if not base.exists():
        raise FileNotFoundError(f"Data directory not found: {base}")

    LOG.info(f"[scan] loading and cleaning poems from: {base}")
    lines = load_lines_from_dir(str(base))
    LOG.info(f"[scan] cleaned lines={len(lines)}")

    # Write cleaned corpus
    _write_lines(CLEAN_FILE, lines)

    # Split and write train/val
    train_lines, val_lines = split_train_val(lines, val_fraction=VAL_FRACTION, seed=42)
    _write_lines(TRAIN_FILE, train_lines)
    _write_lines(VAL_FILE, val_lines)

    # Stats
    stats = {
        "data_dir": str(base),
        "clean": _length_stats(lines),
        "train": _length_stats(train_lines),
        "val": _length_stats(val_lines),
    }
    STATS_FILE.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    LOG.info(f"[stats] wrote {STATS_FILE}")
    LOG.info(f"[stats] summary clean_count={stats['clean']['count']} avg_chars={stats['clean'].get('avg_chars', 0)}")

    # Train SentencePiece on cleaned corpus (optional)
    try:
        train_spm_on_clean_corpus(SPM_INPUT)
    except Exception as e:
        LOG.info(f"[spm] training failed/skipped: {e}")


if __name__ == "__main__":
    main()
