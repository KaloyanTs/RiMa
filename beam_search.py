# -*- coding: utf-8 -*-
"""
SOFT (non-strict) meter generator using beam search.

- Loads words_combined.csv locally
- Uses frequency columns (if present) to prefer common words
- Generates a single line with soft iamb/trochee meter: mismatches are penalized, not forbidden

Run:
  python soft_meter_beam.py

Edit CSV_PATH and maybe METER/N_FEET.
"""

from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# -------------------- config --------------------
CSV_PATH = "hf_dataset/words_combined.csv"  # <-- change to your local path

RANDOM_SEED = 7
METER = "iamb"       # "iamb" or "trochee"
N_FEET = 5           # 5 feet => 10 syllables target
MAX_WORD_SYLL = 6    # ignore longer words (keeps search feasible)

# Beam search parameters
BEAM_WIDTH = 250
EXPAND_PER_STATE = 60     # how many candidate next-words to sample per partial line
MAX_STEPS = 20            # max words in a line (safety)

# Scoring (tune these)
METER_MISMATCH_WEIGHT = 2.5   # bigger => more rhythmic
WORD_REPEAT_PENALTY = 1.2     # penalize repeating same word
SHORT_WORD_PENALTY = 0.15     # slightly penalize 1-syll words spam
# -----------------------------------------------

VOWELS = set("а ъ о у е и ю я ѝ".split())


@dataclass(frozen=True)
class WordInfo:
    clean: str
    stressed: str
    syllables: int
    stress_syllable: int   # 1-based
    weight: float          # frequency weight (bigger => more likely)


@dataclass
class State:
    words: List[WordInfo]
    pos: int               # syllable-position in the target pattern (0..target_len)
    score: float           # lower is better
    used: Dict[str, int]   # repetition counter


def syllable_count(word: str) -> int:
    return sum(1 for c in word.lower() if c in VOWELS)


def parse_stress(name_stressed: str) -> Optional[Tuple[str, int, int]]:
    """
    Returns (clean_word, syllables, stress_syllable_1based) or None.
    Stress mark: backtick ` (and also supports combining grave U+0300).
    If multiple marks exist, take the last one as primary.
    """
    s = (name_stressed or "").strip()
    if not s:
        return None

    STRESS_MARKS = {"`", "\u0300"}
    clean_chars: List[str] = []
    stressed_vowel_indices: List[int] = []

    for ch in s:
        if ch in STRESS_MARKS:
            if clean_chars:
                stressed_vowel_indices.append(len(clean_chars) - 1)
        else:
            clean_chars.append(ch)

    clean = "".join(clean_chars)
    syl = syllable_count(clean)
    if syl == 0 or not stressed_vowel_indices:
        return None

    primary_vowel_idx = stressed_vowel_indices[-1]
    stress_syl = sum(1 for c in clean[: primary_vowel_idx + 1].lower() if c in VOWELS)
    if not (1 <= stress_syl <= syl):
        return None

    return clean, syl, stress_syl


def meter_pattern(meter: str, n_feet: int) -> List[int]:
    meter = meter.lower().strip()
    if meter not in {"iamb", "trochee"}:
        raise ValueError("meter must be 'iamb' or 'trochee'")
    base = [0, 1] if meter == "iamb" else [1, 0]
    return base * n_feet


def word_stress_vec(w: WordInfo) -> List[int]:
    return [1 if (i + 1) == w.stress_syllable else 0 for i in range(w.syllables)]


def safe_float(x: str) -> Optional[float]:
    try:
        if x is None:
            return None
        x = str(x).strip()
        if x == "":
            return None
        return float(x)
    except Exception:
        return None


def load_lexicon(csv_path: str) -> List[WordInfo]:
    """
    Robust loader:
    - handles BOM via utf-8-sig
    - trims headers (e.g. " name_stressed")
    - tries to build a frequency weight from common columns:
        corpus_count / chitanka_count / search_count
      falling back to 1.0
    """
    items: List[WordInfo] = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        if not r.fieldnames:
            raise RuntimeError("No header found in CSV.")
        field_map = {fn.strip(): fn for fn in r.fieldnames}

        if "name_stressed" not in field_map:
            raise RuntimeError(f"Couldn't find 'name_stressed'. Available: {list(field_map.keys())}")

        stressed_col = field_map["name_stressed"]
        deleted_col = field_map.get("deleted_at")

        # possible freq columns (use what exists)
        freq_cols = []
        for c in ("corpus_count", "chitanka_count", "search_count"):
            if c in field_map:
                freq_cols.append(field_map[c])

        for row in r:
            if deleted_col:
                deleted = (row.get(deleted_col) or "").strip()
                if deleted and deleted.lower() not in {"0", "0.0", "nan", "none", ""}:
                    continue

            ns = (row.get(stressed_col) or "").strip()
            parsed = parse_stress(ns)
            if parsed is None:
                continue

            clean, syl, stress_syl = parsed
            if syl > MAX_WORD_SYLL:
                continue

            # build weight from available frequency columns
            w = 1.0
            if freq_cols:
                vals = [safe_float(row.get(fc)) for fc in freq_cols]
                vals = [v for v in vals if v is not None and v > 0]
                if vals:
                    w = sum(vals)  # crude but effective for ranking

            items.append(WordInfo(clean=clean, stressed=ns, syllables=syl, stress_syllable=stress_syl, weight=w))

    return items


def build_sampler(words: List[WordInfo], rng: random.Random) -> Tuple[List[WordInfo], List[float]]:
    """
    Returns (words, cumulative weights) for fast weighted sampling.
    """
    weights = [max(1e-9, w.weight) for w in words]
    total = sum(weights)
    cum = []
    s = 0.0
    for w in weights:
        s += w / total
        cum.append(s)
    return words, cum


def weighted_sample(words: List[WordInfo], cum: List[float], rng: random.Random) -> WordInfo:
    x = rng.random()
    lo, hi = 0, len(cum) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if cum[mid] >= x:
            hi = mid
        else:
            lo = mid + 1
    return words[lo]


def next_word_cost(w: WordInfo, used: Dict[str, int]) -> float:
    # language cost: prefer frequent words => smaller cost
    # cost = -log(p) up to a constant; we approximate with -log(weight)
    lang = -math.log(max(1e-9, w.weight))

    # repetition penalty
    rep = used.get(w.clean, 0)
    rep_pen = WORD_REPEAT_PENALTY * rep

    # small penalty to discourage too many 1-syll words
    short_pen = SHORT_WORD_PENALTY if w.syllables == 1 else 0.0

    return lang + rep_pen + short_pen


def meter_mismatch_cost(pat: List[int], pos: int, w: WordInfo) -> Tuple[int, float]:
    """
    Compare word stress vec to the pattern slice at [pos : pos + w.syllables].
    If word would overshoot, return big penalty via mismatches.
    Returns (new_pos, cost)
    """
    L = w.syllables
    if pos + L > len(pat):
        # overshoot is strongly discouraged but not impossible
        # (we just forbid it for clean generation)
        return pos, 1e9

    slice_pat = pat[pos:pos + L]
    vec = word_stress_vec(w)
    mismatches = sum(1 for a, b in zip(vec, slice_pat) if a != b)
    return pos + L, METER_MISMATCH_WEIGHT * mismatches


def generate_line_soft(
    words: List[WordInfo],
    meter: str,
    n_feet: int,
    rng: random.Random,
) -> List[WordInfo]:
    pat = meter_pattern(meter, n_feet)
    target_len = len(pat)

    sample_words, cum = build_sampler(words, rng)

    # initial state
    init = State(words=[], pos=0, score=0.0, used={})
    beam: List[State] = [init]

    for _step in range(MAX_STEPS):
        new_beam: List[State] = []

        for st in beam:
            if st.pos == target_len:
                return st.words

            # expand this state by sampling candidate next words
            for _ in range(EXPAND_PER_STATE):
                w = weighted_sample(sample_words, cum, rng)

                new_pos, m_cost = meter_mismatch_cost(pat, st.pos, w)
                if m_cost >= 1e8:
                    continue

                c = next_word_cost(w, st.used) + m_cost

                new_used = dict(st.used)
                new_used[w.clean] = new_used.get(w.clean, 0) + 1

                new_state = State(
                    words=st.words + [w],
                    pos=new_pos,
                    score=st.score + c,
                    used=new_used,
                )
                new_beam.append(new_state)

        if not new_beam:
            # if we got stuck, restart beam from scratch
            beam = [init]
            continue

        # keep best BEAM_WIDTH states
        new_beam.sort(key=lambda s: s.score)
        beam = new_beam[:BEAM_WIDTH]

        # if best already completes, return
        if beam and beam[0].pos == target_len:
            return beam[0].words

    # if we didn't finish, return best partial (or raise)
    beam.sort(key=lambda s: (abs(target_len - s.pos), s.score))
    return beam[0].words

def word_stress_vec(w: WordInfo) -> List[int]:
    # 1 at stressed syllable, 0 elsewhere
    return [1 if (i + 1) == w.stress_syllable else 0 for i in range(w.syllables)]

def stress_vector(words: List[WordInfo]) -> List[int]:
    vec: List[int] = []
    for w in words:
        vec.extend(word_stress_vec(w))
    return vec

def main() -> None:
    rng = random.Random()

    print("Loading lexicon...")
    lex = load_lexicon(CSV_PATH)
    print(f"Loaded {len(lex):,} words with stress")
    if not lex:
        raise RuntimeError("Lexicon empty. Check CSV_PATH / headers / encoding.")

    line = generate_line_soft(lex, METER, N_FEET, rng)

    print("\nGenerated (soft) line:")
    print("  " + " ".join(w.clean for w in line))
    print("Syllables:", sum(w.syllables for w in line), "target:", 2 * N_FEET)
    print("Pattern:  ", meter_pattern(METER, N_FEET))
    print("Vector:   ", stress_vector(line))


if __name__ == "__main__":
    main()
