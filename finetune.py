# bg_rhyme_pipeline.py
# One-file pipeline: (optional) prepare corpus -> build stress-aware rhyme dict -> make synthetic AA couplets
# -> LoRA-bf16 fine-tune with checkpoint resume (and safe fallback) -> (optional) merge -> demo generate
#
# Key feature you asked for:
# - If interrupted, it will try to resume from the latest checkpoint.
# - If your torch < 2.6 and Transformers refuses to load optimizer state (CVE guard),
#   it falls back to "weights-only resume": loads adapter weights from the latest checkpoint
#   and continues training with a fresh optimizer (no crash, but optimizer/scheduler reset).

from __future__ import annotations

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import re
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable, DefaultDict
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)
from peft import LoraConfig, get_peft_model, PeftModel, TaskType


# =========================
# CONSTANTS (edit here)
# =========================

MODEL_NAME = "INSAIT-Institute/BgGPT-Gemma-2-2.6B-IT-v1.0"

PROJECT_ROOT = Path(".").resolve()
POEMS_DIR = PROJECT_ROOT / "chitanka_poems_step1"
DICT_CSV = PROJECT_ROOT / "dictionary.csv"

POEMS_TXT = PROJECT_ROOT / "spm_train.txt"
SYNTH_TXT = PROJECT_ROOT / "synthetic_aa.txt"

OUT_STYLE_ADAPTERS = PROJECT_ROOT / "lora_style"
OUT_RHYME_ADAPTERS = PROJECT_ROOT / "lora_rhyme"

DROP_LINES_WITH_LATIN = True
MIN_LINE_SYLLABLES = 3

TARGET_SYLLABLES_PER_LINE = 12
N_SYNTH_COUPLETS = 20000

GLOBAL_SEED = 42

MAX_SEQ_LEN = 256
STYLE_EPOCHS = 1
RHYME_EPOCHS = 1

LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.0
WARMUP_RATIO = 0.03

PER_DEVICE_BATCH = 1
GRAD_ACCUM = 16
LOGGING_STEPS = 25

SAVE_STRATEGY = "steps"
SAVE_STEPS = 20
SAVE_TOTAL_LIMIT = 3

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

RUN_PREPARE_POEMS_TXT = False
RUN_BUILD_SYNTH = True
RUN_TRAIN_STYLE = True
RUN_TRAIN_RHYME = True

RUN_MERGE_FINAL = True
OUT_MERGED_MODEL = PROJECT_ROOT / "merged_model_final"

# Special token for stanza/line boundaries in synthetic data
SPECIAL_NL = "<NL>"

# Demo generation
RUN_DEMO_GENERATE = True
DEMO_USE_MERGED_IF_AVAILABLE = True  # if merged_model_final exists, use it
DEMO_ADAPTERS = "rhyme"  # "style" or "rhyme" (only used when not using merged)
DEMO_PROMPTS = [
    "Напиши поетична линия за морето и небето.",
    "Завърши ред, който завършва със слънцето.",
]
DEMO_MAX_NEW_TOKENS = 64
DEMO_TEMPERATURE = 0.8
DEMO_TOP_P = 0.9
DEMO_TOP_K = 50
DEMO_REPETITION_PENALTY = 1.1
DEMO_OUT_TXT = PROJECT_ROOT / "demo_generations.txt"


# =========================
# Resume helpers
# =========================

def latest_checkpoint(output_dir: Path) -> Optional[str]:
    """Return latest checkpoint-* directory path, else None."""
    if not output_dir.exists():
        return None
    candidates = []
    for p in output_dir.glob("checkpoint-*"):
        m = re.search(r"checkpoint-(\d+)$", p.name)
        if m and p.is_dir():
            candidates.append((int(m.group(1)), p))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return str(candidates[-1][1])


def read_trainer_state(ckpt_dir: str) -> Optional[dict]:
    """Read trainer_state.json if present (useful for debugging)."""
    p = Path(ckpt_dir) / "trainer_state.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def try_train_resume(trainer: Trainer, ckpt: Optional[str]) -> bool:
    """
    Try true resume (includes optimizer/scheduler state).
    Returns True if succeeded, False if we should fallback to weights-only resume.
    """
    if not ckpt:
        trainer.train()
        return True

    print(f"[TRAIN] Attempting true resume from: {ckpt}")
    st = read_trainer_state(ckpt)
    if st:
        gs = st.get("global_step", None)
        ep = st.get("epoch", None)
        print(f"[TRAIN] Checkpoint trainer_state: global_step={gs}, epoch={ep}")

    try:
        trainer.train(resume_from_checkpoint=ckpt)
        return True
    except ValueError as e:
        msg = str(e)
        # Transformers added a safety guard requiring torch>=2.6 to load optimizer state.
        if ("torch.load" in msg and "upgrade torch" in msg) or ("CVE-2025-32434" in msg):
            print("[WARN] True resume blocked by torch security guard (torch<2.6).")
            print("[WARN] Falling back to WEIGHTS-ONLY resume (optimizer/scheduler will be reset).")
            return False
        raise  # if it's another error, don't hide it


# =========================
# Bulgarian text utilities
# =========================

BG_VOWELS = set(list("аеиоуъюяѝАЕИОУЪЮЯЍ"))
LATIN_RE = re.compile(r"[A-Za-z]")
CYRILLIC_WORD_RE = re.compile(r"[А-Яа-яЁёЍѝ]+", re.UNICODE)

TRAIL_PUNCT_RE = re.compile(r"([,.;:!?…\"”)]+)$", re.UNICODE)
LEAD_PUNCT_RE = re.compile(r"^[\"“(„]+", re.UNICODE)

APOSTROPHE = "'"


def count_syllables_word_simple(word: str) -> int:
    return sum(1 for ch in word if ch in BG_VOWELS)


def normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def strip_edge_punct(token: str) -> str:
    token = LEAD_PUNCT_RE.sub("", token)
    token = TRAIL_PUNCT_RE.sub("", token)
    return token


def split_tokens(line: str) -> List[str]:
    return [t for t in line.strip().split() if t]


def line_has_latin(line: str) -> bool:
    return bool(LATIN_RE.search(line))


def extract_last_word_and_span(line: str) -> Optional[Tuple[str, Tuple[int, int], str]]:
    """
    Returns (last_word_clean, (start,end) span in original line for replacement, trailing_punct)
    last_word_clean contains only cyrillic letters (lowercased).
    """
    if not line.strip():
        return None
    tokens = split_tokens(line)
    if not tokens:
        return None

    idx = None
    for i in range(len(tokens) - 1, -1, -1):
        if CYRILLIC_WORD_RE.search(tokens[i]):
            idx = i
            break
    if idx is None:
        return None

    token = tokens[idx]
    m = TRAIL_PUNCT_RE.search(token)
    trailing = m.group(1) if m else ""
    token_core = strip_edge_punct(token)

    words = CYRILLIC_WORD_RE.findall(token_core)
    if not words:
        return None
    last_word = words[-1].lower()

    pattern = re.compile(re.escape(words[-1]) + r"(" + re.escape(trailing) + r")?\s*$", re.UNICODE)
    match = pattern.search(line)
    if not match:
        return last_word, (-1, -1), trailing

    start = match.start()
    end = match.start() + len(words[-1])
    return last_word, (start, end), trailing


def count_syllables_line(line: str) -> int:
    words = CYRILLIC_WORD_RE.findall(line)
    return sum(count_syllables_word_simple(w) for w in words)


def capitalize_like(src: str, dst: str) -> str:
    if src and src[0].isupper():
        return dst[:1].upper() + dst[1:]
    return dst


# =========================
# Dictionary structures
# =========================

@dataclass(frozen=True)
class DictEntry:
    word: str
    stressed_form: str
    syllables: int
    stress_vowel_index: int
    rhyme_key: str


def syllables_from_hyphenated(s: str) -> Optional[int]:
    if not isinstance(s, str) or not s.strip():
        return None
    parts = [p for p in s.split("-") if p]
    return len(parts) if parts else None


def build_dictionary(csv_path: Path) -> Tuple[Dict[str, DictEntry], Dict[Tuple[str, int], List[str]]]:
    """
    Uses:
      - word_stressed with apostrophe BEFORE the stressed vowel, exactly once.
      - rhyme key = substring from stressed vowel to end (lowercased).
      - syllables = hyphen count in word_syllables, fallback to vowel count.
    """
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    required_cols = {"word", "word_stressed", "word_syllables"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"dictionary.csv missing columns: {missing}. Found: {list(df.columns)}")

    entries: Dict[str, DictEntry] = {}
    rhyme_groups: Dict[Tuple[str, int], List[str]] = defaultdict(list)

    for _, row in df.iterrows():
        word = (row.get("word") or "").strip().lower()
        stressed = (row.get("word_stressed") or "").strip()

        if not word:
            continue
        if not stressed or APOSTROPHE not in stressed:
            continue

        apos_idx = stressed.find(APOSTROPHE)
        if apos_idx < 0 or apos_idx >= len(stressed) - 1:
            continue
        stressed_vowel = stressed[apos_idx + 1]
        if stressed_vowel.lower() not in BG_VOWELS:
            continue

        cleaned = (stressed[:apos_idx] + stressed[apos_idx + 1:]).strip()
        cleaned_lc = cleaned.lower()

        stress_vowel_index = apos_idx  # apostrophe removed; stressed vowel moved into this index
        if not (0 <= stress_vowel_index < len(cleaned_lc)):
            continue

        rhyme_key = cleaned_lc[stress_vowel_index:]
        syl = syllables_from_hyphenated(row.get("word_syllables", ""))
        if syl is None:
            syl = count_syllables_word_simple(cleaned_lc)

        if word not in entries:
            entry = DictEntry(
                word=word,
                stressed_form=stressed,
                syllables=int(syl),
                stress_vowel_index=int(stress_vowel_index),
                rhyme_key=rhyme_key,
            )
            entries[word] = entry
            rhyme_groups[(entry.rhyme_key, entry.syllables)].append(word)

    return entries, rhyme_groups


# =========================
# Corpus preparation
# =========================

def iter_poem_lines_from_dir(poems_dir: Path) -> Iterable[str]:
    for fp in sorted(poems_dir.glob("*.txt")):
        text = fp.read_text(encoding="utf-8", errors="replace")
        for raw in text.splitlines():
            yield raw


def prepare_poems_txt(poems_dir: Path, out_path: Path) -> int:
    kept = 0
    out_lines: List[str] = []
    for raw in iter_poem_lines_from_dir(poems_dir):
        line = normalize_spaces(raw)
        if not line:
            continue
        if DROP_LINES_WITH_LATIN and line_has_latin(line):
            continue
        if count_syllables_line(line) < MIN_LINE_SYLLABLES:
            continue
        out_lines.append(line)
        kept += 1
    out_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return kept


def build_word_frequency(poems_txt: Path) -> Dict[str, int]:
    freq: DefaultDict[str, int] = defaultdict(int)
    for ln in poems_txt.read_text(encoding="utf-8", errors="replace").splitlines():
        info = extract_last_word_and_span(ln)
        if not info:
            continue
        w, _, _ = info
        if w:
            freq[w] += 1
    return dict(freq)


# =========================
# Synthetic AA couplets
# =========================

def replace_last_word(line: str, new_word: str) -> str:
    info = extract_last_word_and_span(line)
    if not info:
        return line
    _, (start, end), _ = info

    if start != -1 and end != -1:
        old_sub = line[start:end]
        nw = capitalize_like(old_sub, new_word)
        return line[:start] + nw + line[end:]
    else:
        tokens = split_tokens(line)
        for i in range(len(tokens) - 1, -1, -1):
            if CYRILLIC_WORD_RE.search(tokens[i]):
                tok = tokens[i]
                m = TRAIL_PUNCT_RE.search(tok)
                tr = m.group(1) if m else ""
                core = strip_edge_punct(tok)
                words = CYRILLIC_WORD_RE.findall(core)
                if not words:
                    continue
                old = words[-1]
                nw = capitalize_like(old, new_word)
                new_core = re.sub(re.escape(old) + r"$", nw, core)
                lead = tok[: tok.find(core)] if core in tok else ""
                tokens[i] = f"{lead}{new_core}{tr}"
                break
        return " ".join(tokens)


def build_synthetic_aa(
    poems_txt: Path,
    dict_entries: Dict[str, DictEntry],
    rhyme_groups: Dict[Tuple[str, int], List[str]],
    out_path: Path,
    n_couplets: int,
    target_syllables: int,
    word_freq: Optional[Dict[str, int]] = None,
) -> int:
    lines = poems_txt.read_text(encoding="utf-8", errors="replace").splitlines()

    candidates: List[str] = []
    candidates_by_last_syl: DefaultDict[int, List[str]] = defaultdict(list)

    for ln in lines:
        if count_syllables_line(ln) != target_syllables:
            continue
        info = extract_last_word_and_span(ln)
        if not info:
            continue
        last_word, _, _ = info
        entry = dict_entries.get(last_word)
        if not entry:
            continue
        candidates.append(ln)
        candidates_by_last_syl[entry.syllables].append(ln)

    if not candidates:
        raise RuntimeError(
            f"No candidate lines found with exactly {target_syllables} syllables and last word in dictionary."
        )

    rng = random.Random(GLOBAL_SEED)
    synth_lines: List[str] = []
    made = 0
    attempts = 0
    max_attempts = n_couplets * 50

    pbar = tqdm(total=n_couplets, desc="Generating synthetic AA couplets")
    while made < n_couplets and attempts < max_attempts:
        attempts += 1

        line1 = rng.choice(candidates)
        info1 = extract_last_word_and_span(line1)
        if not info1:
            continue
        w1, _, _ = info1
        e1 = dict_entries.get(w1)
        if not e1:
            continue

        group = rhyme_groups.get((e1.rhyme_key, e1.syllables), [])
        if len(group) < 2:
            continue

        if word_freq:
            sorted_group = sorted(group, key=lambda w: (-(word_freq.get(w, 0)), rng.random()))
            top_k = min(10, len(sorted_group))
            w2 = rng.choice(sorted_group[:top_k])
        else:
            w2 = rng.choice(group)

        if w2 == w1:
            continue

        pool2 = candidates_by_last_syl.get(e1.syllables, [])
        if not pool2:
            continue
        line2 = rng.choice(pool2)

        line2_rhymed = replace_last_word(line2, w2)
        if count_syllables_line(line2_rhymed) != target_syllables:
            continue

        synth_lines.append(line1)
        synth_lines.append(line2_rhymed)
        synth_lines.append(SPECIAL_NL)
        made += 1
        pbar.update(1)

    pbar.close()
    if made < n_couplets:
        print(f"[WARN] Only generated {made}/{n_couplets} couplets after {attempts} attempts.")

    out_path.write_text("\n".join(synth_lines).rstrip() + "\n", encoding="utf-8")
    return made


# =========================
# Training helpers
# =========================

def load_text_dataset(txt_path: Path):
    return load_dataset("text", data_files=str(txt_path), split="train")


def tokenize_dataset(ds, tokenizer: AutoTokenizer, max_len: int):
    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_len, padding=False)
    return ds.map(tok, batched=True, remove_columns=["text"])


def make_trainer(model, tokenizer, tokenized_ds, out_dir: Path, epochs: int, run_name: str):
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        logging_steps=LOGGING_STEPS,
        save_strategy=SAVE_STRATEGY,
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        bf16=True,
        fp16=False,
        optim="adamw_torch",
        report_to="none",
        run_name=run_name,
        dataloader_num_workers=0,   # avoids fork warnings + simplest for SSH runs
        gradient_checkpointing=True,
        max_grad_norm=1.0,
    )

    return Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_ds,
        data_collator=collator,
    )


def build_lora_model(base_model):
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    )
    peft_model = get_peft_model(base_model, lora_cfg)
    peft_model.print_trainable_parameters()
    return peft_model


def load_base_model_trainable(model_name: str, device_map: str = "auto"):
    m = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )
    m.gradient_checkpointing_enable()
    m.config.use_cache = False
    return m


# =========================
# Demo generation helpers
# =========================

def load_inference_model_and_tokenizer(model_path_or_name: str, adapters_dir: Optional[Path]):
    tokenizer = AutoTokenizer.from_pretrained(model_path_or_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        model_path_or_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = base
    if adapters_dir and adapters_dir.exists():
        model = PeftModel.from_pretrained(base, str(adapters_dir))
    model.eval()
    model.config.use_cache = True
    return tokenizer, model


@torch.no_grad()
def generate_text(model, tokenizer, prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    device = model.device if hasattr(model, "device") else ("cuda" if torch.cuda.is_available() else "cpu")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    out = model.generate(
        **inputs,
        max_new_tokens=DEMO_MAX_NEW_TOKENS,
        do_sample=True,
        temperature=DEMO_TEMPERATURE,
        top_p=DEMO_TOP_P,
        top_k=DEMO_TOP_K,
        repetition_penalty=DEMO_REPETITION_PENALTY,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    text = text.replace(SPECIAL_NL, "\n")
    gen_part = text[len(prompt):] if text.startswith(prompt) else text
    first_line = re.split(r"\n|\r|\u2028|\u2029", gen_part, maxsplit=1)[0]
    return normalize_spaces(first_line)


# =========================
# Main
# =========================

def main():
    set_seed(GLOBAL_SEED)
    random.seed(GLOBAL_SEED)

    if not POEMS_DIR.exists() or not POEMS_DIR.is_dir():
        raise FileNotFoundError(f"Expected poems folder: {POEMS_DIR} (with many .txt files)")
    if not DICT_CSV.exists():
        raise FileNotFoundError(f"Expected dictionary file: {DICT_CSV}")

    # Step 1: Prepare poems
    if RUN_PREPARE_POEMS_TXT:
        print(f"[1/5] Preparing {POEMS_TXT} from {POEMS_DIR} ...")
        kept = prepare_poems_txt(POEMS_DIR, POEMS_TXT)
        print(f"      Kept {kept} lines -> {POEMS_TXT}")
    else:
        print(f"[1/5] Skipping poems preparation (RUN_PREPARE_POEMS_TXT=False).")

    # Step 2: Dictionary
    print(f"[2/5] Loading dictionary from {DICT_CSV} ...")
    dict_entries, rhyme_groups = build_dictionary(DICT_CSV)
    print(f"      Entries with stress: {len(dict_entries):,}")
    print(f"      Rhyme groups: {len(rhyme_groups):,}")

    # Step 3: Synthetic data
    if RUN_BUILD_SYNTH:
        print(f"[3/5] Building synthetic AA couplets -> {SYNTH_TXT}")
        word_freq = build_word_frequency(POEMS_TXT)
        made = build_synthetic_aa(
            poems_txt=POEMS_TXT,
            dict_entries=dict_entries,
            rhyme_groups=rhyme_groups,
            out_path=SYNTH_TXT,
            n_couplets=N_SYNTH_COUPLETS,
            target_syllables=TARGET_SYLLABLES_PER_LINE,
            word_freq=word_freq,
        )
        print(f"      Wrote {made} couplets -> {SYNTH_TXT}")
    else:
        print(f"[3/5] Skipping synthetic data generation (RUN_BUILD_SYNTH=False).")

    # Step 4: Tokenizer (+ add SPECIAL_NL as special token)
    print(f"[4/5] Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Add stanza separator token as an actual special token (better than splitting into sub-tokens)
    if SPECIAL_NL not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [SPECIAL_NL]})

    # --------------------------
    # Stage A: Style LoRA
    # --------------------------
    if RUN_TRAIN_STYLE:
        print("[TRAIN] Stage A: style LoRA")

        ckpt = latest_checkpoint(OUT_STYLE_ADAPTERS)
        if ckpt:
            print(f"[TRAIN] Found style checkpoint: {ckpt}")

        base_model = load_base_model_trainable(MODEL_NAME, device_map="auto")
        # Ensure embeddings resized if we added SPECIAL_NL
        base_model.resize_token_embeddings(len(tokenizer))

        ds_style = load_text_dataset(POEMS_TXT)
        tok_style = tokenize_dataset(ds_style, tokenizer, MAX_SEQ_LEN)

        # If checkpoint exists, prefer true resume; if blocked, do weights-only resume
        if ckpt:
            # Build a fresh LoRA structure (must match what checkpoint expects)
            style_model = build_lora_model(base_model)
            trainer = make_trainer(style_model, tokenizer, tok_style, OUT_STYLE_ADAPTERS, STYLE_EPOCHS, "bg_style_lora")
            ok = try_train_resume(trainer, ckpt)
            if not ok:
                # weights-only: load adapter weights from checkpoint, then train with fresh optimizer
                print(f"[TRAIN] WEIGHTS-ONLY resume style from: {ckpt}")
                style_model = PeftModel.from_pretrained(base_model, ckpt, is_trainable=True)
                trainer = make_trainer(style_model, tokenizer, tok_style, OUT_STYLE_ADAPTERS, STYLE_EPOCHS, "bg_style_lora")
                trainer.train()
        else:
            style_model = build_lora_model(base_model)
            trainer = make_trainer(style_model, tokenizer, tok_style, OUT_STYLE_ADAPTERS, STYLE_EPOCHS, "bg_style_lora")
            trainer.train()

        style_model.save_pretrained(str(OUT_STYLE_ADAPTERS))
        tokenizer.save_pretrained(str(OUT_STYLE_ADAPTERS))
        print(f"Saved style adapters to: {OUT_STYLE_ADAPTERS}")
    else:
        print("[TRAIN] Skipping style training")

    # --------------------------
    # Stage B: Rhyme LoRA
    # --------------------------
    if RUN_TRAIN_RHYME:
        print("[TRAIN] Stage B: rhyme LoRA")

        ckpt2 = latest_checkpoint(OUT_RHYME_ADAPTERS)
        if ckpt2:
            print(f"[TRAIN] Found rhyme checkpoint: {ckpt2}")

        rhyme_base = load_base_model_trainable(MODEL_NAME, device_map="auto")
        rhyme_base.resize_token_embeddings(len(tokenizer))

        ds_rhyme = load_text_dataset(SYNTH_TXT)
        tok_rhyme = tokenize_dataset(ds_rhyme, tokenizer, MAX_SEQ_LEN)

        # If we already have a rhyme checkpoint, it contains the full adapter state for this stage.
        if ckpt2:
            # Try true resume first (needs torch>=2.6), else weights-only resume.
            rhyme_model = build_lora_model(rhyme_base)
            trainer2 = make_trainer(rhyme_model, tokenizer, tok_rhyme, OUT_RHYME_ADAPTERS, RHYME_EPOCHS, "bg_rhyme_lora")
            ok2 = try_train_resume(trainer2, ckpt2)
            if not ok2:
                print(f"[TRAIN] WEIGHTS-ONLY resume rhyme from: {ckpt2}")
                rhyme_model = PeftModel.from_pretrained(rhyme_base, ckpt2, is_trainable=True)
                trainer2 = make_trainer(rhyme_model, tokenizer, tok_rhyme, OUT_RHYME_ADAPTERS, RHYME_EPOCHS, "bg_rhyme_lora")
                trainer2.train()
        else:
            # No rhyme checkpoint: start from style adapters if available, else fresh LoRA.
            if OUT_STYLE_ADAPTERS.exists():
                print(f"Loading style adapters from {OUT_STYLE_ADAPTERS} as starting point...")
                rhyme_model = PeftModel.from_pretrained(rhyme_base, str(OUT_STYLE_ADAPTERS), is_trainable=True)
            else:
                print("Style adapters not found; starting rhyme training from fresh LoRA on base model.")
                rhyme_model = build_lora_model(rhyme_base)

            trainer2 = make_trainer(rhyme_model, tokenizer, tok_rhyme, OUT_RHYME_ADAPTERS, RHYME_EPOCHS, "bg_rhyme_lora")
            trainer2.train()

        rhyme_model.save_pretrained(str(OUT_RHYME_ADAPTERS))
        tokenizer.save_pretrained(str(OUT_RHYME_ADAPTERS))
        print(f"Saved rhyme adapters to: {OUT_RHYME_ADAPTERS}")
    else:
        print("[TRAIN] Skipping rhyme training")

    # --------------------------
    # Merge final adapters
    # --------------------------
    if RUN_MERGE_FINAL:
        if not OUT_RHYME_ADAPTERS.exists():
            raise FileNotFoundError("Cannot merge: rhyme adapters folder not found.")
        print(f"[MERGE] Merging adapters into full model at: {OUT_MERGED_MODEL}")

        merge_base = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
        )
        # Ensure embeddings sized the same as tokenizer (SPECIAL_NL)
        merge_base.resize_token_embeddings(len(tokenizer))

        merged = PeftModel.from_pretrained(merge_base, str(OUT_RHYME_ADAPTERS))
        merged = merged.merge_and_unload()
        OUT_MERGED_MODEL.mkdir(parents=True, exist_ok=True)
        merged.save_pretrained(str(OUT_MERGED_MODEL))
        tokenizer.save_pretrained(str(OUT_MERGED_MODEL))
        print(f"Saved merged model to: {OUT_MERGED_MODEL}")

    print("[DONE]")

    # --------------------------
    # Demo generation
    # --------------------------
    if RUN_DEMO_GENERATE:
        print("[DEMO] Generating sample outputs...")

        if DEMO_USE_MERGED_IF_AVAILABLE and OUT_MERGED_MODEL.exists():
            demo_model_path = str(OUT_MERGED_MODEL)
            demo_adapters = None
            print(f"[DEMO] Using merged model: {OUT_MERGED_MODEL}")
        else:
            demo_model_path = MODEL_NAME
            adapters = OUT_RHYME_ADAPTERS if DEMO_ADAPTERS == "rhyme" else OUT_STYLE_ADAPTERS
            demo_adapters = adapters if adapters.exists() else None
            print(f"[DEMO] Using base model + adapters: {demo_adapters}")

        tok_demo, mdl_demo = load_inference_model_and_tokenizer(demo_model_path, demo_adapters)

        lines = []
        for i, p in enumerate(DEMO_PROMPTS, 1):
            print(f"[DEMO] Prompt {i}: {p}")
            gen = generate_text(mdl_demo, tok_demo, p)
            print(f"[DEMO] Output {i}: {gen}")
            lines.append(f"### Prompt {i}\n{p}\n### Output {i}\n{gen}\n")

        DEMO_OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
        print(f"[DEMO] Saved outputs to: {DEMO_OUT_TXT}")


if __name__ == "__main__":
    main()
