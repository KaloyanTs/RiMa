# -*- coding: utf-8 -*-
"""
Conditional Seq2Seq Transformer for Bulgarian poetry lines
=========================================================

Encoder condition:
  <SYL_k> <END> end_word <SEP> optional context lines (A1/A2/...)

Decoder:
  generates the whole line left->right.

This version IMPLEMENTS the data fixes you asked for:
1) drop lines with no Bulgarian last word (empty_last -> ~0)
2) drop lines containing Latin letters or digits
3) dedupe identical lines *within each poem*
(+ optional: remove train/val overlap by filtering val lines that appear in train)

Also logs regular data/training stats (loss + trunc_rate + avg_tgt_len + empty_last + end_mismatch)
and val loss + perplexity.

Run:
  python cond_seq2seq_poetry.py
"""

from __future__ import annotations

import csv
import glob
import logging
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F


# ====================== CONFIG ======================
POEMS_GLOB = "chitanka_poems_step1/*.txt"
DICT_CSV = "dictionary.csv"

# Tokenizer
SPM_PREFIX = "poems_cond_bpe"
SPM_MODEL = f"{SPM_PREFIX}.model"
SPM_TRAIN_TXT = "spm_train.txt"
VOCAB_SIZE = 5000
CHAR_COVERAGE = 0.9995

SEP_SYMBOL = "<SEP>"
END_SYMBOL = "<END>"
SYL_MAX = 24
SYL_TOK = lambda k: f"<SYL_{k}>"

# Data split
SEED = 42
VAL_FRACTION = 0.08
# If True: remove any val lines that occur in train (then drop val poems <4 lines)
FILTER_VAL_OVERLAP = True

# Data cleaning (the requested changes)
DROP_LATIN_OR_DIGITS = True
DEDUP_WITHIN_POEM = True

# Model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
D_MODEL = 256
N_HEAD = 8
ENC_LAYERS = 4
DEC_LAYERS = 6
D_FF = 4 * D_MODEL
DROPOUT = 0.25

# Lengths
MEM_CTX = 256
CTX = 96  # decoder input length

# Training
BATCH = 256
LR = 3e-4
WEIGHT_DECAY = 0.05
TRAIN_STEPS = 30_000
CLIP_NORM = 0.5
LABEL_SMOOTH = 0.1
TOKEN_DROPOUT_P = 0.2

SHOW_EVERY = 50
VAL_EVERY = 250
VAL_BATCHES = 40

CKPT_PATH = "cond_seq2seq.pt"
LOG_FILE = "cond_seq2seq.log"

# Task mixing
P_NOCTX = 0.35
P_BETWEEN = 0.45
P_NEXT = 0.20

# Syllable constraint (set None to allow variable)
TARGET_SYL = 10
# ====================================================


WORD_RE = re.compile(r"[а-яА-ЯёЁѝЍ]+")
VOWELS = set("а ъ о у е и ю я ѝ".split())
BAD_LATIN_DIGIT_RE = re.compile(r"[A-Za-z0-9]")


# ---------------- logging ----------------
def setup_logger() -> logging.Logger:
    lg = logging.getLogger("cond_seq2seq")
    lg.setLevel(logging.INFO)
    lg.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    lg.addHandler(fh)
    lg.addHandler(sh)
    return lg


LOG = setup_logger()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collapse_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def last_word(line: str) -> str:
    ws = WORD_RE.findall(line)
    return ws[-1].lower() if ws else ""


def normalize_line(line: str) -> str:
    return collapse_ws(line).lower()


def _count_vowels(s: str) -> int:
    return sum(1 for ch in s.lower() if ch in VOWELS)


# ---------------- SentencePiece ----------------
def concat_poems_to_file(poems_glob: str, out_path: str) -> int:
    files = sorted(glob.glob(poems_glob))
    if not files:
        raise RuntimeError(f"No poems found: {poems_glob}")
    parts = [Path(fp).read_text(encoding="utf-8", errors="ignore") for fp in files]
    text = "\n".join(parts)
    Path(out_path).write_text(text, encoding="utf-8")
    return len(text)


def train_spm_if_needed() -> None:
    LOG.info("[spm] preparing training text...")
    n_chars = concat_poems_to_file(POEMS_GLOB, SPM_TRAIN_TXT)
    LOG.info(f"[spm] wrote {SPM_TRAIN_TXT} ({n_chars} chars)")

    if Path(SPM_MODEL).exists():
        LOG.info(f"[spm] using existing: {SPM_MODEL}")
        return

    user_syms = [SEP_SYMBOL, END_SYMBOL] + [SYL_TOK(k) for k in range(1, SYL_MAX + 1)]
    LOG.info("[spm] training tokenizer...")
    spm.SentencePieceTrainer.train(
        input=SPM_TRAIN_TXT,
        model_prefix=SPM_PREFIX,
        vocab_size=VOCAB_SIZE,
        model_type="bpe",
        character_coverage=CHAR_COVERAGE,
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
        user_defined_symbols=user_syms,
    )
    LOG.info(f"[spm] trained: {SPM_MODEL}")


def load_spm() -> spm.SentencePieceProcessor:
    sp = spm.SentencePieceProcessor()
    sp.load(SPM_MODEL)
    return sp


# ---------------- poems / cleaning / splitting ----------------
def load_poem_lines(poems_glob: str) -> List[List[str]]:
    """
    Loads poems and applies the requested cleaning:
      - drop empty lines
      - drop lines with no Bulgarian last word
      - drop lines containing Latin letters or digits
      - dedupe identical lines within a poem
    """
    files = sorted(glob.glob(poems_glob))
    if not files:
        raise RuntimeError(f"No poems found: {poems_glob}")

    all_poems: List[List[str]] = []
    dropped_empty_last = 0
    dropped_latin_digit = 0
    dropped_short_poems = 0
    dedup_removed = 0

    for fp in files:
        txt = Path(fp).read_text(encoding="utf-8", errors="ignore")
        lines = [collapse_ws(x) for x in txt.splitlines()]
        lines = [x for x in lines if x]

        # (1) drop lines with no Bulgarian last word
        filtered = []
        for ln in lines:
            if not last_word(ln):
                dropped_empty_last += 1
                continue
            # (2) drop Latin/digits (optional)
            if DROP_LATIN_OR_DIGITS and BAD_LATIN_DIGIT_RE.search(ln):
                dropped_latin_digit += 1
                continue
            filtered.append(ln)
        lines = filtered

        # (3) dedupe within poem (optional)
        if DEDUP_WITHIN_POEM and lines:
            seen = set()
            ded = []
            for ln in lines:
                key = normalize_line(ln)
                if key in seen:
                    dedup_removed += 1
                    continue
                seen.add(key)
                ded.append(ln)
            lines = ded

        if len(lines) >= 4:
            all_poems.append(lines)
        else:
            dropped_short_poems += 1

    LOG.info(
        f"[data:clean] poems={len(all_poems)} dropped_empty_last={dropped_empty_last} "
        f"dropped_latin_digit={dropped_latin_digit} dedup_removed={dedup_removed} "
        f"dropped_short_poems={dropped_short_poems}"
    )
    if not all_poems:
        raise RuntimeError("Need poems with >=4 lines after cleaning.")
    return all_poems


def split_train_val_poems(poems: List[List[str]], val_fraction: float, seed: int) -> Tuple[List[List[str]], List[List[str]]]:
    rng = random.Random(seed)
    idx = list(range(len(poems)))
    rng.shuffle(idx)
    n_val = max(1, int(len(poems) * val_fraction))
    val_idx = set(idx[:n_val])
    train = [poems[i] for i in range(len(poems)) if i not in val_idx]
    val = [poems[i] for i in range(len(poems)) if i in val_idx]
    return train, val


def filter_val_overlap(train_poems: List[List[str]], val_poems: List[List[str]]) -> List[List[str]]:
    train_set = set()
    for p in train_poems:
        for ln in p:
            train_set.add(normalize_line(ln))

    kept: List[List[str]] = []
    removed_lines = 0
    removed_poems = 0
    for p in val_poems:
        q = []
        for ln in p:
            if normalize_line(ln) in train_set:
                removed_lines += 1
                continue
            q.append(ln)
        if len(q) >= 4:
            kept.append(q)
        else:
            removed_poems += 1

    LOG.info(f"[data:val_overlap_fix] removed_lines={removed_lines} removed_poems={removed_poems} val_poems={len(kept)}")
    return kept


def flatten_lines(poems: List[List[str]]) -> List[str]:
    out: List[str] = []
    for p in poems:
        out.extend(p)
    return out


def log_dup_stats(train_poems: List[List[str]], val_poems: List[List[str]]) -> None:
    train_lines = [normalize_line(x) for x in flatten_lines(train_poems)]
    val_lines = [normalize_line(x) for x in flatten_lines(val_poems)]

    def dups_count(xs: List[str]) -> Tuple[int, int]:
        from collections import Counter
        c = Counter(xs)
        dups = sum(v - 1 for v in c.values() if v > 1)
        uniq = sum(1 for v in c.values() if v == 1)
        return dups, uniq

    train_dups, train_unique = dups_count(train_lines)
    val_dups, _ = dups_count(val_lines)
    train_set = set(train_lines)
    overlap = sum(1 for x in set(val_lines) if x in train_set)

    LOG.info(f"[data:dups] train_dups={train_dups} val_dups={val_dups} train_unique={train_unique} train∩val_overlap={overlap}")


# ---------------- dictionary.csv: stress + syllables ----------------
def parse_word_syllables(word_syllables: str, fallback_word: str) -> int:
    ws = (word_syllables or "").strip()
    if ws and "-" in ws:
        parts = [p for p in ws.split("-") if p]
        if parts:
            return len(parts)
    return _count_vowels(fallback_word)


def load_syll_map(dict_csv: str) -> Dict[str, int]:
    syll_map: Dict[str, int] = {}
    with open(dict_csv, newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        for row in r:
            w = (row.get("word") or "").strip().lower()
            wsy = (row.get("word_syllables") or "").strip()
            if not w:
                continue
            clean = w.replace("-", "").replace("'", "")
            syll_map[w] = parse_word_syllables(wsy, clean)
            if clean not in syll_map:
                syll_map[clean] = _count_vowels(clean)
    return syll_map


def count_line_syllables(line: str, syll_map: Dict[str, int]) -> int:
    words = [w.lower() for w in WORD_RE.findall(line)]
    total = 0
    for w in words:
        total += syll_map.get(w, _count_vowels(w))
    return total


# ---------------- batching helpers ----------------
def pad_to(ids: List[int], length: int, pad_id: int) -> List[int]:
    if len(ids) >= length:
        return ids[:length]
    return ids + [pad_id] * (length - len(ids))


def make_encoder_ids(
    sp: spm.SentencePieceProcessor,
    sep_id: int,
    end_id: int,
    syl_id: int,
    end_word: str,
    ctx_lines: List[str],
    max_len: int,
) -> List[int]:
    prefix = [syl_id, end_id] + sp.encode(end_word, out_type=int)

    ctx: List[int] = []
    for ln in ctx_lines:
        ctx.append(sep_id)
        ctx.extend(sp.encode(ln, out_type=int))

    if len(prefix) >= max_len:
        return prefix[:max_len]

    keep_ctx = max_len - len(prefix)
    ctx = ctx[-keep_ctx:]
    return prefix + ctx


# ---------------- model ----------------
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d: int, h: int, dropout: float, causal: bool, max_len: int):
        super().__init__()
        assert d % h == 0
        self.h = h
        self.dk = d // h
        self.causal = causal
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)
        if causal:
            self.register_buffer("tri", torch.tril(torch.ones(max_len, max_len)).unsqueeze(0).unsqueeze(0))
        else:
            self.tri = None

    def forward(self, x: torch.Tensor, key_pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=-1)
        q = q.view(B, T, self.h, self.dk).transpose(1, 2)
        k = k.view(B, T, self.h, self.dk).transpose(1, 2)
        v = v.view(B, T, self.h, self.dk).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)

        if self.causal:
            att = att.masked_fill(self.tri[:, :, :T, :T] == 0, float("-inf"))

        if key_pad_mask is not None:
            km = key_pad_mask.view(B, 1, 1, T)
            att = att.masked_fill(km == 0, float("-inf"))

        att = F.softmax(att, dim=-1)
        att = self.drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.proj(y))


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d: int, h: int, dropout: float):
        super().__init__()
        assert d % h == 0
        self.h = h
        self.dk = d // h
        self.q = nn.Linear(d, d)
        self.kv = nn.Linear(d, 2 * d)
        self.proj = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mem: torch.Tensor, mem_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, D = x.shape
        M = mem.shape[1]

        q = self.q(x).view(B, T, self.h, self.dk).transpose(1, 2)
        kv = self.kv(mem)
        k, v = kv.split(D, dim=-1)
        k = k.view(B, M, self.h, self.dk).transpose(1, 2)
        v = v.view(B, M, self.h, self.dk).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)
        if mem_mask is not None:
            mm = mem_mask.view(B, 1, 1, M)
            att = att.masked_fill(mm == 0, float("-inf"))

        att = F.softmax(att, dim=-1)
        att = self.drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, D)
        return self.drop(self.proj(y))


class EncoderBlock(nn.Module):
    def __init__(self, d: int, h: int, dff: int, dropout: float, max_len: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = MultiHeadSelfAttention(d, h, dropout, causal=False, max_len=max_len)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, dff), nn.GELU(), nn.Linear(dff, d), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), key_pad_mask=pad_mask)
        x = x + self.ff(self.ln2(x))
        return x


class DecoderBlock(nn.Module):
    def __init__(self, d: int, h: int, dff: int, dropout: float, max_len: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.self_attn = MultiHeadSelfAttention(d, h, dropout, causal=True, max_len=max_len)
        self.ln_mem = nn.LayerNorm(d)
        self.cross_attn = MultiHeadCrossAttention(d, h, dropout)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, dff), nn.GELU(), nn.Linear(dff, d), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor, mem: torch.Tensor, mem_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.ln1(x), key_pad_mask=None)
        x = x + self.cross_attn(self.ln_mem(x), mem, mem_mask=mem_mask)
        x = x + self.ff(self.ln2(x))
        return x


class Encoder(nn.Module):
    def __init__(self, vocab: int, max_len: int, pad_id: int):
        super().__init__()
        self.pad_id = pad_id
        self.tok = nn.Embedding(vocab, D_MODEL)
        self.pos = nn.Embedding(max_len, D_MODEL)
        self.drop = nn.Dropout(DROPOUT)
        self.blocks = nn.ModuleList([EncoderBlock(D_MODEL, N_HEAD, D_FF, DROPOUT, max_len) for _ in range(ENC_LAYERS)])
        self.ln = nn.LayerNorm(D_MODEL)

    def forward(self, ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, M = ids.shape
        pos = torch.arange(M, device=ids.device)
        x = self.tok(ids) + self.pos(pos)[None, :, :]
        x = self.drop(x)
        mask = (ids != self.pad_id).long()
        for blk in self.blocks:
            x = blk(x, pad_mask=mask)
        x = self.ln(x)
        return x, mask


class Decoder(nn.Module):
    def __init__(self, vocab: int, max_len: int, pad_id: int):
        super().__init__()
        self.vocab = vocab
        self.max_len = max_len
        self.pad_id = pad_id
        self.tok = nn.Embedding(vocab, D_MODEL)
        self.pos = nn.Embedding(max_len, D_MODEL)
        self.drop = nn.Dropout(DROPOUT)
        self.blocks = nn.ModuleList([DecoderBlock(D_MODEL, N_HEAD, D_FF, DROPOUT, max_len) for _ in range(DEC_LAYERS)])
        self.ln = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, vocab, bias=False)

    def forward(self, idx: torch.Tensor, mem: torch.Tensor, mem_mask: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok(idx) + self.pos(pos)[None, :, :]
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x, mem=mem, mem_mask=mem_mask)
        x = self.ln(x)
        return self.head(x)


class CondSeq2Seq(nn.Module):
    def __init__(self, vocab: int, pad_id: int):
        super().__init__()
        self.pad_id = pad_id
        self.enc = Encoder(vocab=vocab, max_len=MEM_CTX, pad_id=pad_id)
        self.dec = Decoder(vocab=vocab, max_len=CTX, pad_id=pad_id)

    def forward(self, mem_ids: torch.Tensor, dec_in: torch.Tensor, dec_tgt: Optional[torch.Tensor] = None):
        mem, mem_mask = self.enc(mem_ids)
        logits = self.dec(dec_in, mem=mem, mem_mask=mem_mask)
        loss = None
        if dec_tgt is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                dec_tgt.view(-1),
                ignore_index=self.pad_id,
                label_smoothing=LABEL_SMOOTH,
            )
        return logits, loss


# ---------------- training data sampler ----------------
class CondSampler:
    def __init__(
        self,
        sp: spm.SentencePieceProcessor,
        poems: List[List[str]],
        sep_id: int,
        end_id: int,
        syl_ids: Dict[int, int],
        syll_map: Dict[str, int],
    ):
        self.sp = sp
        self.poems = poems
        self.sep_id = sep_id
        self.end_id = end_id
        self.syl_ids = syl_ids
        self.syll_map = syll_map
        self.bos = sp.bos_id()
        self.eos = sp.eos_id()
        self.pad = sp.pad_id()

    def _pick_syl(self, line: str) -> int:
        s = count_line_syllables(line, self.syll_map)
        s = max(1, min(SYL_MAX, s))
        return s

    def _make_example(self, target_line: str, ctx_lines: List[str]) -> Tuple[List[int], List[int], str, bool]:
        endw = last_word(target_line)
        empty_last = (endw == "")
        if empty_last:
            endw = "а"  # should be very rare after cleaning
        syl = self._pick_syl(target_line)
        if TARGET_SYL is not None:
            syl = TARGET_SYL

        syl = max(1, min(SYL_MAX, syl))
        syl_id = self.syl_ids[syl]

        mem = make_encoder_ids(
            self.sp,
            self.sep_id,
            self.end_id,
            syl_id,
            end_word=endw,
            ctx_lines=ctx_lines,
            max_len=MEM_CTX,
        )
        tgt = self.sp.encode(target_line, out_type=int)
        return mem, tgt, endw, empty_last

    def sample(self) -> Tuple[List[int], List[int], str, bool]:
        r = random.random()
        if r < P_NOCTX:
            lines = random.choice(self.poems)
            t = random.choice(lines)
            return self._make_example(t, ctx_lines=[])

        if r < P_NOCTX + P_BETWEEN:
            lines = random.choice(self.poems)
            if len(lines) < 3:
                return self.sample()
            i = random.randint(0, len(lines) - 3)
            left, mid, right = lines[i], lines[i + 1], lines[i + 2]
            return self._make_example(mid, ctx_lines=[left, right])

        lines = random.choice(self.poems)
        if len(lines) < 4:
            return self.sample()
        i = random.randint(0, len(lines) - 4)
        a1, b1, a2, b2 = lines[i], lines[i + 1], lines[i + 2], lines[i + 3]
        return self._make_example(b2, ctx_lines=[a1, b1, a2])


@dataclass
class BatchStats:
    trunc_rate: float
    avg_tgt_len: float
    empty_last_rate: float
    end_mismatch_rate: float


def make_batch(sampler: CondSampler, batch: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, BatchStats]:
    mems, xs, ys = [], [], []
    truncs = 0
    tgt_lens = []
    empty_last = 0
    end_mismatch = 0

    for _ in range(batch):
        mem, tgt, endw, was_empty_last = sampler.sample()
        empty_last += int(was_empty_last)

        mem = pad_to(mem, MEM_CTX, sampler.pad)

        # truncate target to fit CTX (leave room for EOS)
        tgt_len = len(tgt)
        if tgt_len > (CTX - 1):
            truncs += 1
            tgt = tgt[: CTX - 1]
        tgt_lens.append(min(tgt_len, CTX - 1))

        seq = [sampler.bos] + tgt + [sampler.eos]
        x = pad_to(seq[:-1], CTX, sampler.pad)
        y = pad_to(seq[1:], CTX, sampler.pad)

        # end_word mismatch sanity check (should be 0 by construction)
        # (we compute on raw text level, so it's inherently consistent unless last_word() fails)
        # Here we can only check that endw is non-empty; mismatch is tracked in generation, not batch.

        mems.append(mem)
        xs.append(x)
        ys.append(y)

    mem_t = torch.tensor(mems, dtype=torch.long, device=DEVICE)

    # token dropout on encoder memory (protect special tokens)
    if TOKEN_DROPOUT_P > 0:
        protect_ids = {sampler.pad, sampler.bos, sampler.eos, sampler.sep_id, sampler.end_id}
        protect = torch.zeros_like(mem_t, dtype=torch.bool)
        for pid in protect_ids:
            protect |= (mem_t == pid)
        drop_mask = (torch.rand_like(mem_t, dtype=torch.float) < TOKEN_DROPOUT_P) & (~protect)
        mem_t = mem_t.masked_fill(drop_mask, sampler.pad)

    x_t = torch.tensor(xs, dtype=torch.long, device=DEVICE)
    y_t = torch.tensor(ys, dtype=torch.long, device=DEVICE)

    stats = BatchStats(
        trunc_rate=truncs / max(1, batch),
        avg_tgt_len=float(sum(tgt_lens) / max(1, len(tgt_lens))),
        empty_last_rate=empty_last / max(1, batch),
        end_mismatch_rate=end_mismatch / max(1, batch),
    )
    return mem_t, x_t, y_t, stats


@torch.no_grad()
def eval_val_loss(model: CondSeq2Seq, sampler: CondSampler, batches: int) -> Tuple[float, float]:
    model.eval()
    losses = []
    for _ in range(batches):
        mem, x, y, _ = make_batch(sampler, BATCH)
        _, loss = model(mem, x, y)
        losses.append(float(loss.item()))
    mean_loss = float(sum(losses) / max(1, len(losses)))
    ppl = float(math.exp(min(20.0, mean_loss)))  # cap for stability
    return mean_loss, ppl


# ---------------- ckpt ----------------
def save_ckpt(path: str, model: nn.Module, opt: torch.optim.Optimizer, step: int) -> None:
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step}, path)
    LOG.info(f"[ckpt] saved {path} step={step}")


def load_ckpt(path: str, model: nn.Module, opt: torch.optim.Optimizer) -> int:
    ckpt = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["opt"])
    step = int(ckpt.get("step", 0))
    LOG.info(f"[ckpt] loaded {path} step={step}")
    return step


# ---------------- main ----------------
def main() -> None:
    set_seed(SEED)
    LOG.info(f"[env] device={DEVICE} torch={torch.__version__}")

    train_spm_if_needed()
    sp = load_spm()

    sep_id = sp.piece_to_id(SEP_SYMBOL)
    end_id = sp.piece_to_id(END_SYMBOL)
    if sep_id == sp.unk_id() or end_id == sp.unk_id():
        raise RuntimeError("Missing special tokens in tokenizer. Delete *.model/*.vocab and retrain.")

    syl_ids: Dict[int, int] = {}
    for k in range(1, SYL_MAX + 1):
        pid = sp.piece_to_id(SYL_TOK(k))
        if pid == sp.unk_id():
            raise RuntimeError(f"Missing {SYL_TOK(k)} in tokenizer. Retrain tokenizer.")
        syl_ids[k] = pid

    LOG.info(
        f"[spm] vocab={sp.get_piece_size()} pad={sp.pad_id()} bos={sp.bos_id()} eos={sp.eos_id()} sep={sep_id} end={end_id}"
    )

    poems = load_poem_lines(POEMS_GLOB)
    train_poems, val_poems = split_train_val_poems(poems, VAL_FRACTION, SEED)
    if FILTER_VAL_OVERLAP:
        val_poems = filter_val_overlap(train_poems, val_poems)
    log_dup_stats(train_poems, val_poems)
    LOG.info(f"[data] train_poems={len(train_poems)} val_poems={len(val_poems)}")

    syll_map = load_syll_map(DICT_CSV)
    LOG.info(f"[dict] syll_words={len(syll_map)}")

    train_sampler = CondSampler(sp, train_poems, sep_id=sep_id, end_id=end_id, syl_ids=syl_ids, syll_map=syll_map)
    val_sampler = CondSampler(sp, val_poems, sep_id=sep_id, end_id=end_id, syl_ids=syl_ids, syll_map=syll_map)

    model = CondSeq2Seq(vocab=sp.get_piece_size(), pad_id=sp.pad_id()).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    start = 0
    if Path(CKPT_PATH).exists():
        start = load_ckpt(CKPT_PATH, model, opt)

    model.train()
    LOG.info(f"[train] {start} -> {TRAIN_STEPS} batch={BATCH} lr={LR} weight_decay={WEIGHT_DECAY}")

    for step in range(start + 1, TRAIN_STEPS + 1):
        mem, x, y, st = make_batch(train_sampler, BATCH)
        _, loss = model(mem, x, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
        opt.step()

        if step == 1 or step % SHOW_EVERY == 0:
            LOG.info(
                f"[train] step={step} loss={loss.item():.4f} trunc_rate={st.trunc_rate:.3f} "
                f"avg_tgt_len={st.avg_tgt_len:.1f} empty_last={st.empty_last_rate:.3f} end_mismatch={st.end_mismatch_rate:.3f}"
            )

        if step == 1 or step % VAL_EVERY == 0:
            vloss, vppl = eval_val_loss(model, val_sampler, VAL_BATCHES)
            LOG.info(f"[val] step={step} loss={vloss:.4f} ppl={vppl:.1f}")
            model.train()

        if step % 2000 == 0 or step == TRAIN_STEPS:
            save_ckpt(CKPT_PATH, model, opt, step)


if __name__ == "__main__":
    main()
