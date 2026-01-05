from __future__ import annotations

import csv
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from tqdm import tqdm


# -------------------- USER CONFIG --------------------
N_GRAM = 6          # 2=bigram, 3=trigram, ... (NOTE: very large N gets sparse; backoff helps but N=3..6 is practical)
SCHEME = "ABAB"     # "ABAB" or "AABB"

POEMS_DIR = Path("chitanka_poems_step1")
CSV_PATH = "hf_dataset/words_combined.csv"

METER = "iamb"      # "iamb" or "trochee"
N_FEET = 5          # 5 feet => 10 syllables
MAX_WORD_SYLL = 6   # ignore longer words from lexicon (keeps search feasible)
# -----------------------------------------------------


# -------------------- SEARCH / SCORING CONFIG --------------------
BEAM_WIDTH = 350
EXPAND_PER_STATE = 200
MAX_STEPS = 30

# Candidate proposal
TOP_NEXT_PER_CTX = 200        # take up to this many top next-words from context
EXPLORATION_SAMPLES = 10      # add a few random words for diversity/backoff

# Soft constraints weights
METER_MISMATCH_WEIGHT = 5.0   # larger -> stricter rhythm
REPEAT_PENALTY = 1.2
TOPIC_BONUS = 0.05            # reward for reusing a "topic word" from line 1
UNK_PENALTY = 10.0             # extra penalty if word not in LM vocab (rare here because we propose from LM)

# Rhyme (soft)
RHYME_SUFFIX_LEN = 3          # rhyme key = last N chars (simple but effective)
RHYME_BONUS = 4.0             # reward if last word matches target rhyme key

# Add-k smoothing (for LM logp)
ADD_K = 0.5

# Tokenization choice
WORDS_ONLY = True             # True -> only Cyrillic words; False -> keep punctuation tokens too
# -----------------------------------------------------


VOWELS = set("а ъ о у е и ю я ѝ".split())

# words-only: Bulgarian-ish Cyrillic tokens
WORD_RE_WORDS_ONLY = re.compile(r"[а-яА-ЯёЁѝЍ]+")
# keep punctuation too (optional)
WORD_RE_WITH_PUNCT = re.compile(r"[а-яА-ЯёЁѝЍ]+|[.,;:!?—-]")


@dataclass(frozen=True)
class WordInfo:
    clean: str
    stressed: str
    syllables: int
    stress_syllable: int  # 1-based


@dataclass
class State:
    words: List[str]
    pos: int
    score: float
    used: Dict[str, int]
    context: Tuple[str, ...]       # length <= N_GRAM-1 (we keep fixed length in beam)
    nll_sum: float                 # sum of -logp for diagnostics
    nll_count: int                 # how many LM steps contributed
    topic_hits: int                # how many topic words used (diagnostics)


def syllable_count(word: str) -> int:
    return sum(1 for c in word.lower() if c in VOWELS)


def parse_stress(name_stressed: str) -> Optional[WordInfo]:
    """
    Dataset stress mark:
      - ASCII backtick ` (U+0060) after stressed vowel
      - also supports combining grave U+0300 just in case
    If multiple marks exist, uses the last one as primary stress.
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

    clean = "".join(clean_chars).lower()
    syl = syllable_count(clean)
    if syl == 0 or not stressed_vowel_indices:
        return None

    primary_vowel_idx = stressed_vowel_indices[-1]
    stress_syl = sum(1 for c in clean[: primary_vowel_idx + 1] if c in VOWELS)
    if not (1 <= stress_syl <= syl):
        return None

    return WordInfo(clean=clean, stressed=s, syllables=syl, stress_syllable=stress_syl)


def meter_pattern(meter: str, n_feet: int) -> List[int]:
    meter = meter.lower().strip()
    if meter not in {"iamb", "trochee"}:
        raise ValueError("meter must be 'iamb' or 'trochee'")
    base = [0, 1] if meter == "iamb" else [1, 0]
    return base * n_feet


def word_stress_vec(w: WordInfo) -> List[int]:
    return [1 if (i + 1) == w.stress_syllable else 0 for i in range(w.syllables)]


def meter_mismatch_cost(pat: List[int], pos: int, wi: WordInfo) -> Tuple[int, float, int]:
    """
    Returns (new_pos, weighted_cost, mismatch_count).
    Overshoot => huge cost.
    """
    L = wi.syllables
    if pos + L > len(pat):
        return pos, 1e9, 999999
    slice_pat = pat[pos:pos + L]
    vec = word_stress_vec(wi)
    mismatches = sum(1 for a, b in zip(vec, slice_pat) if a != b)
    return pos + L, METER_MISMATCH_WEIGHT * mismatches, mismatches


def rhyme_key(word: str, n: int = RHYME_SUFFIX_LEN) -> str:
    w = word.lower()
    return w[-n:] if len(w) >= n else w


def tokenize(text: str) -> List[str]:
    if WORDS_ONLY:
        return [t.lower() for t in WORD_RE_WORDS_ONLY.findall(text)]
    return [t.lower() for t in WORD_RE_WITH_PUNCT.findall(text)]


def load_poems_sequences(poems_dir: Path, n: int) -> List[List[str]]:
    files = sorted(poems_dir.glob("*.txt"))
    if not files:
        return []

    bos = ["<s>"] * (n - 1)
    eos = ["</s>"]
    sequences: List[List[str]] = []

    for p in tqdm(files, desc="Reading poems", unit="file"):
        txt = p.read_text(encoding="utf-8", errors="ignore")
        for line in txt.splitlines():
            toks = tokenize(line)
            if toks:
                sequences.append(bos + toks + eos)
    return sequences


def load_lexicon(csv_path: str) -> Dict[str, List[WordInfo]]:
    """
    Returns mapping: clean_word -> [WordInfo variants]
    Robust to BOM and whitespace in headers.
    """
    by_word: Dict[str, List[WordInfo]] = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        if not r.fieldnames:
            raise RuntimeError("No header found in stress CSV.")
        field_map = {fn.strip(): fn for fn in r.fieldnames}
        if "name_stressed" not in field_map:
            raise RuntimeError(f"Missing name_stressed column. Columns: {list(field_map.keys())}")
        stressed_col = field_map["name_stressed"]

        rows = list(r)

    for row in tqdm(rows, desc="Loading stress lexicon", unit="row"):
        ns = (row.get(stressed_col) or "").strip()
        wi = parse_stress(ns)
        if wi is None:
            continue
        if wi.syllables > MAX_WORD_SYLL:
            continue
        by_word.setdefault(wi.clean, []).append(wi)

    return by_word


class BackoffNgramLM:
    """
    N-gram LM with:
    - add-k smoothing
    - backoff for logp(ctx,w) if ctx unseen (shorten context)
    - context->top next words lookup for candidate proposal
    """

    def __init__(self, n: int, add_k: float = 0.5):
        if n < 2:
            raise ValueError("n must be >= 2")
        self.n = n
        self.add_k = add_k
        self.max_ctx = n - 1

        # counts for all context lengths 0..max_ctx
        # context_counts[k][ctx] = count(ctx)
        # ngram_counts[k][ctx + (w,)] = count(ctx,w)   where ctx length = k
        self.context_counts: List[Dict[Tuple[str, ...], int]] = [dict() for _ in range(self.max_ctx + 1)]
        self.ngram_counts: List[Dict[Tuple[str, ...], int]] = [dict() for _ in range(self.max_ctx + 1)]

        self.vocab: Dict[str, int] = {}
        self.vocab_size = 0

        # proposal cache: for each ctx (any length), list of (word, count) sorted desc
        self.next_cache: Dict[Tuple[str, ...], List[Tuple[str, int]]] = {}

    def train(self, sequences: List[List[str]]) -> None:
        # Train for multiple epochs
        EPOCHS = 50  # Adjust as needed
        for epoch in range(EPOCHS):
            for seq in tqdm(sequences, desc=f"Training backoff {self.n}-gram (Epoch {epoch+1}/{EPOCHS})", unit="line"):
                for w in seq:
                    self.vocab[w] = self.vocab.get(w, 0) + 1

                # update all orders up to N
                for i in range(len(seq)):
                    # contexts up to max_ctx ending at i-1
                    for k in range(0, self.max_ctx + 1):
                        if i - k < 0:
                            continue
                        ctx = tuple(seq[i - k:i])  # length k (for predicting next token at position i)
                        self.context_counts[k][ctx] = self.context_counts[k].get(ctx, 0) + 1

                # update (ctx,w) pairs for all ctx lengths k=0..max_ctx
                for i in range(self.max_ctx, len(seq)):
                    for k in range(0, self.max_ctx + 1):
                        ctx = tuple(seq[i - k:i])        # length k
                        w = seq[i]
                        key = ctx + (w,)
                        self.ngram_counts[k][key] = self.ngram_counts[k].get(key, 0) + 1

        self.vocab_size = len(self.vocab)
        self._build_next_cache()

    def _build_next_cache(self) -> None:
        """
        Build ctx -> top next words using raw counts (no smoothing needed for proposals).
        We build for all ctx lengths present.
        """
        for k in range(0, self.max_ctx + 1):
            for key, c in self.ngram_counts[k].items():
                ctx = key[:-1]
                w = key[-1]
                self.next_cache.setdefault(ctx, []).append((w, c))

        # sort descending by count
        for ctx, lst in self.next_cache.items():
            lst.sort(key=lambda x: x[1], reverse=True)

    def logp(self, ctx: Tuple[str, ...], w: str) -> float:
        """
        Backoff: try longest suffix of ctx up to max_ctx, then shorten until found.
        Add-k smoothing at that order.
        """
        # ensure ctx length = max_ctx (beam uses fixed-length context)
        if len(ctx) != self.max_ctx:
            raise ValueError("Context length mismatch (beam should maintain fixed-length context).")

        for k in range(self.max_ctx, -1, -1):
            short_ctx = ctx[-k:] if k > 0 else tuple()
            c_ctx = self.context_counts[k].get(short_ctx, 0)
            c_ng = self.ngram_counts[k].get(short_ctx + (w,), 0)

            # if context unseen at this order, back off (unless k==0)
            if c_ctx == 0 and k > 0:
                continue

            num = c_ng + self.add_k
            den = c_ctx + self.add_k * max(1, self.vocab_size)
            return math.log(num / den)

        # should never happen
        return -math.log(max(1, self.vocab_size))

    def propose_next(self, ctx: Tuple[str, ...], k: int = TOP_NEXT_PER_CTX) -> List[str]:
        """
        Propose next words from best available context (backoff).
        Returns top-k next candidates by count.
        """
        if len(ctx) != self.max_ctx:
            raise ValueError("Context length mismatch.")

        for clen in range(self.max_ctx, -1, -1):
            short_ctx = ctx[-clen:] if clen > 0 else tuple()
            lst = self.next_cache.get(short_ctx)
            if lst:
                return [w for (w, _c) in lst[:k]]
        return []


def best_variant_for_meter(
    variants: List[WordInfo], pat: List[int], pos: int
) -> Optional[Tuple[WordInfo, int, float, int]]:
    """
    Pick the stress variant with minimum meter penalty at this position.
    Returns (variant, new_pos, meter_cost, mismatch_count) or None if overshoots.
    """
    best: Optional[Tuple[WordInfo, int, float, int]] = None
    for wi in variants:
        new_pos, cost, mism = meter_mismatch_cost(pat, pos, wi)
        if cost >= 1e8:
            continue
        if best is None or cost < best[2]:
            best = (wi, new_pos, cost, mism)
    return best


def pick_topic_words(line: List[str], max_topic: int = 8) -> List[str]:
    """
    Very simple: pick longer distinct words from line 1 as "topic".
    """
    seen = set()
    out = []
    for w in line:
        if len(w) < 4:
            continue
        if w in seen:
            continue
        seen.add(w)
        out.append(w)
        if len(out) >= max_topic:
            break
    return out


def generate_line_soft(
    lm: BackoffNgramLM,
    lex_by_word: Dict[str, List[WordInfo]],
    rng: random.Random,
    meter: str,
    n_feet: int,
    target_rhyme: Optional[str],
    topic_words: Optional[set[str]],
) -> Tuple[List[str], Dict[str, float]]:
    """
    Beam-search line generator:
    - candidates come from lm.propose_next(context) with exploration
    - score = -logp + meter_penalty + repeat_penalty - topic_bonus + (rhyme bonus at end)

    Returns (line_tokens, diagnostics dict).
    """
    pat = meter_pattern(meter, n_feet)
    target_len = len(pat)

    # Candidate universe for fallback/exploration
    # Use words present in BOTH LM vocab and lexicon
    global_cands = [w for w in lm.vocab.keys() if w not in {"<s>", "</s>"} and w in lex_by_word]
    if not global_cands:
        raise RuntimeError("No overlap between LM vocab and stress lexicon.")

    init_ctx = tuple(["<s>"] * lm.max_ctx)
    init = State(words=[], pos=0, score=0.0, used={}, context=init_ctx, nll_sum=0.0, nll_count=0, topic_hits=0)
    beam: List[State] = [init]

    for _step in range(MAX_STEPS):
        new_beam: List[State] = []

        for st in beam:
            # done?
            if st.pos == target_len:
                # apply rhyme bonus only when completed
                final_score = st.score
                if target_rhyme is not None and st.words:
                    if rhyme_key(st.words[-1]) == target_rhyme:
                        final_score -= RHYME_BONUS
                st.score = final_score
                new_beam.append(st)
                continue

            # propose candidates by context with backoff + exploration
            proposed = lm.propose_next(st.context, k=TOP_NEXT_PER_CTX)

            # filter to lexicon overlap
            proposed = [w for w in proposed if w in lex_by_word and w not in {"<s>", "</s>"}]
            # exploration samples
            for _ in range(EXPLORATION_SAMPLES):
                proposed.append(rng.choice(global_cands))

            # de-dup but keep order
            seen = set()
            cand_list = []
            for w in proposed:
                if w in seen:
                    continue
                seen.add(w)
                cand_list.append(w)

            # expand
            for w in cand_list[:EXPAND_PER_STATE]:
                # LM cost
                lp = lm.logp(st.context, w)
                nll = -lp

                # meter: choose best stress variant
                best = best_variant_for_meter(lex_by_word[w], pat, st.pos)
                if best is None:
                    continue
                _wi, new_pos, m_cost, _mism = best

                # repetition
                rep = st.used.get(w, 0)
                rep_cost = REPEAT_PENALTY * rep

                # topic coherence bonus
                topic_bonus = 0.0
                topic_hits = st.topic_hits
                if topic_words is not None and w in topic_words:
                    topic_bonus = TOPIC_BONUS
                    topic_hits += 1

                total = st.score + nll + m_cost + rep_cost - topic_bonus

                new_used = dict(st.used)
                new_used[w] = rep + 1

                new_ctx = st.context[1:] + (w,)

                new_beam.append(
                    State(
                        words=st.words + [w],
                        pos=new_pos,
                        score=total,
                        used=new_used,
                        context=new_ctx,
                        nll_sum=st.nll_sum + nll,
                        nll_count=st.nll_count + 1,
                        topic_hits=topic_hits,
                    )
                )

        if not new_beam:
            beam = [init]
            continue

        new_beam.sort(key=lambda s: s.score)
        beam = new_beam[:BEAM_WIDTH]

        # early exit if best finished
        if beam and beam[0].pos == target_len:
            best_state = beam[0]
            if target_rhyme is not None and best_state.words:
                if rhyme_key(best_state.words[-1]) == target_rhyme:
                    best_state.score -= RHYME_BONUS
            beam[0] = best_state
            break

    # pick best finished if any, else best closest
    finished = [s for s in beam if s.pos == target_len]
    if finished:
        finished.sort(key=lambda s: s.score)
        best = finished[0]
    else:
        beam.sort(key=lambda s: (abs(target_len - s.pos), s.score))
        best = beam[0]

    # diagnostics
    mismatches = count_meter_mismatches(best.words, lex_by_word, pat)
    syllables = sum(best_variant_for_meter(lex_by_word[w], pat, 0)[0].syllables for w in best.words) if best.words else 0
    avg_nll = best.nll_sum / max(1, best.nll_count)

    diag = {
        "syllables": float(syllables),
        "target_syllables": float(target_len),
        "meter_mismatches": float(mismatches),
        "avg_neglogp": float(avg_nll),
        "topic_hits": float(best.topic_hits),
        "last_rhyme_key": rhyme_key(best.words[-1]) if best.words else "",
        "score": float(best.score),
    }
    return best.words, diag


def count_meter_mismatches(line: List[str], lex_by_word: Dict[str, List[WordInfo]], pat: List[int]) -> int:
    """
    For diagnostics: pick best variant greedily left-to-right and count mismatches.
    """
    pos = 0
    mism = 0
    for w in line:
        best = best_variant_for_meter(lex_by_word[w], pat, pos)
        if best is None:
            break
        _wi, new_pos, _cost, mm = best
        mism += mm
        pos = new_pos
        if pos >= len(pat):
            break
    return mism


def generate_stanza(
    lm: BackoffNgramLM,
    lex_by_word: Dict[str, List[WordInfo]],
    rng: random.Random,
    meter: str,
    n_feet: int,
    scheme: str,
) -> List[Tuple[List[str], Dict[str, float]]]:
    scheme = scheme.upper().strip()
    if scheme not in {"ABAB", "AABB"}:
        raise ValueError("scheme must be ABAB or AABB")

    out: List[Tuple[List[str], Dict[str, float]]] = []

    # Line 1: no rhyme target; sets topic + rhyme A
    l1, d1 = generate_line_soft(lm, lex_by_word, rng, meter, n_feet, target_rhyme=None, topic_words=None)
    topic = set(pick_topic_words(l1))
    A = rhyme_key(l1[-1]) if l1 else None

    # Line 2: target A (soft)
    l2, d2 = generate_line_soft(lm, lex_by_word, rng, meter, n_feet, target_rhyme=A, topic_words=topic)

    # Line 3: if scheme is ABAB => set B from line 3; if AABB => line 3 targets B (and sets B)
    if scheme == "ABAB":
        l3, d3 = generate_line_soft(lm, lex_by_word, rng, meter, n_feet, target_rhyme=None, topic_words=topic)
        B = rhyme_key(l3[-1]) if l3 else None
        l4, d4 = generate_line_soft(lm, lex_by_word, rng, meter, n_feet, target_rhyme=B, topic_words=topic)
    else:  # AABB
        l3, d3 = generate_line_soft(lm, lex_by_word, rng, meter, n_feet, target_rhyme=None, topic_words=topic)
        B = rhyme_key(l3[-1]) if l3 else None
        l3, d3 = generate_line_soft(lm, lex_by_word, rng, meter, n_feet, target_rhyme=B, topic_words=topic)
        l4, d4 = generate_line_soft(lm, lex_by_word, rng, meter, n_feet, target_rhyme=B, topic_words=topic)

    out.append((l1, d1))
    out.append((l2, d2))
    out.append((l3, d3))
    out.append((l4, d4))
    return out


def main() -> None:
    if N_GRAM < 2:
        raise ValueError("N_GRAM must be >= 2")

    rng = random.Random()

    print(f"Loading poems: {POEMS_DIR} | N={N_GRAM} | words_only={WORDS_ONLY}")
    seqs = load_poems_sequences(POEMS_DIR, N_GRAM)
    if not seqs:
        raise RuntimeError("No poems found. Put .txt files into chitanka_poems_step1/")

    print("Training LM...")
    lm = BackoffNgramLM(n=N_GRAM, add_k=ADD_K)
    lm.train(seqs)
    print("LM vocab size:", lm.vocab_size)

    print("Loading stress lexicon...")
    lex_by_word = load_lexicon(CSV_PATH)
    print("Lexicon entries:", len(lex_by_word))

    print(f"\nGenerating stanza: {SCHEME} | meter={METER} | feet={N_FEET} | rhyme_suffix={RHYME_SUFFIX_LEN}")
    stanza = generate_stanza(lm, lex_by_word, rng, METER, N_FEET, SCHEME)

    print("\n--- STANZA ---")
    for i, (line, diag) in enumerate(stanza, 1):
        print(f"{i:>2}. " + " ".join(line))
        print(
            "    "
            f"syll={int(diag['syllables'])}/{int(diag['target_syllables'])} | "
            f"mism={int(diag['meter_mismatches'])} | "
            f"avgNLL={diag['avg_neglogp']:.3f} | "
            f"topicHits={int(diag['topic_hits'])} | "
            f"rhy={diag['last_rhyme_key']} | "
            f"score={diag['score']:.2f}"
        )


if __name__ == "__main__":
    main()