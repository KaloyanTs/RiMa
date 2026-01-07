# abab_pipeline_finetune_and_generate.py
#
# What this script does
# ---------------------
# Two modes (set MODE below):
#   - MODE="finetune":
#       1) Builds a large mixed dataset from:
#           a) all poem lines (T1)
#           b) consecutive lines inside stanzas (T2)
#           c) triples (T3) and quadruples (T4) inside stanzas
#           d) your clean quatrains file rhyming_quatrains_abab_aabb.txt (adds more T2/T3/T4)
#       2) Finetunes LoRA on BgGPT-Gemma-2-2.6B-IT
#       3) Saves adapter to OUTPUT_DIR/final
#       4) Runs a 4-step ABAB demo (pick rhyming x1,x2 and y1,y2; 4 calls)
#
#   - MODE="generate":
#       1) Loads the tuned adapter from OUTPUT_DIR/final
#       2) Runs the same 4-step ABAB demo (no training)
#
#   - MODE="test":
#       1) Loads the tuned adapter from OUTPUT_DIR/final
#       2) Evaluates N random ABAB generations and reports accuracy
#          (share of samples where ALL FOUR line endings match the required words)
#
# The ABAB generation recipe (4 calls)
# -----------------------------------
# 1) given ending <x1>  -> generate line1 ending exactly with x1
# 2) given line1 + <y1> -> generate line2 related to line1 ending exactly with y1
# 3) given line1+line2 + <x2> -> generate line3 related to line1+2 ending exactly with x2
# 4) given line1+2+3 + <y2> -> generate line4 related to line1+2+3 ending exactly with y2
#
# No CLI args: configure constants below.

import os
import re
import csv
import json
import time
import random
import logging
import inspect
from typing import Dict, List, Optional, Tuple, Any, Iterable

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, PeftModel


# =========================
# CONSTANTS
# =========================
MODE = "test"  # "finetune" | "generate" | "test"

# Data locations
POEMS_DIR = "chitanka_poems_step1"
DICT_FILE = "dictionary.csv"
QUATRAINS_FILE = "rhyming_quatrains_abab_aabb.txt"
DEMO_OUT_FILE = "abab_4step_bggpt_demo.txt"

# Model
MODEL_ID = "INSAIT-Institute/BgGPT-Gemma-2-2.6B-IT-v1.0"
ATTN_IMPLEMENTATION = "eager"

# Output
OUTPUT_DIR = "./bggpt-abab-4step-lora"
FINAL_ADAPTER_DIR = os.path.join(OUTPUT_DIR, "final")

# Repro
SEED = 42

# Dataset building
REQUIRE_ENDINGS_IN_DICT = True

# Caps (set to None for "use everything", but watch RAM/disk/time)
MAX_T1 = 250_000   # single-line (x1 -> line1)
MAX_T2 = 250_000   # pairs
MAX_T3 = 150_000   # triples
MAX_T4 = 120_000   # quadruples
MAX_FROM_QUATRAINS = None  # additional examples from QUATRAINS_FILE (None = all)

# Balancing: we oversample smaller tasks by duplicating them up to these target ratios
# (relative weights for final training set)
TASK_WEIGHTS = {"T1": 0.35, "T2": 0.30, "T3": 0.20, "T4": 0.15}
MAX_FINAL_TRAIN_EXAMPLES = None  # None = use full mixed set after balancing

# Training
VAL_RATIO = 0.03
MAX_SEQ_LEN = 256
EPOCHS = 2
LR = 2e-5
PER_DEVICE_BATCH = 2
GRAD_ACCUM = 8
WARMUP_RATIO = 0.05
MAX_GRAD_NORM = 0.3
WEIGHT_DECAY = 0.0

LOGGING_STEPS = 10
EVAL_STEPS = 200
SAVE_STEPS = 200
SAVE_TOTAL_LIMIT = 3

# LoRA
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

# Generation demo
DEMO_SAMPLES = 5
GEN_COUNT = 30
GEN_MAX_ATTEMPTS = 8
GEN_MAX_NEW_TOKENS = 72
GEN_DO_SAMPLE = True
GEN_TEMPERATURE = 0.9
GEN_TOP_P = 0.95
GEN_REPETITION_PENALTY = 1.12
GEN_NO_REPEAT_NGRAM = 3

# Test
TEST_SAMPLES = 500  # None -> use GEN_COUNT; else override number of test trials

# Rhyme picking
MIN_BUCKET = 4  # require buckets at least this big

# Chat template marker for assistant-only loss
ASSISTANT_MARKER = "<start_of_turn>model\n"

# Regex
WORD_RE = re.compile(r"[а-яА-ЯёЁ]+", re.UNICODE)
VOWELS = set("аеиоуъюяАЕИОУЪЮЯ")


# =========================
# LOGGING
# =========================
LOG = logging.getLogger("abab_4step")
LOG.setLevel(logging.INFO)
LOG.propagate = False


def setup_logging():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    fh = logging.FileHandler(os.path.join(OUTPUT_DIR, "run.log"), encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)

    if not any(isinstance(h, logging.StreamHandler) for h in LOG.handlers):
        LOG.addHandler(ch)
    if not any(isinstance(h, logging.FileHandler) for h in LOG.handlers):
        LOG.addHandler(fh)


# =========================
# COMPAT: TrainingArguments rename
# =========================
_TA_PARAMS = set(inspect.signature(TrainingArguments.__init__).parameters.keys())


def make_training_args(**kw) -> TrainingArguments:
    # HF renamed evaluation_strategy -> eval_strategy in some versions
    if "save_safetensors" in _TA_PARAMS and "save_safetensors" not in kw:
        kw["save_safetensors"] = True
    if "evaluation_strategy" in kw and "evaluation_strategy" not in _TA_PARAMS and "eval_strategy" in _TA_PARAMS:
        kw["eval_strategy"] = kw.pop("evaluation_strategy")
    filtered = {k: v for k, v in kw.items() if k in _TA_PARAMS}
    dropped = sorted(set(kw.keys()) - set(filtered.keys()))
    if dropped:
        LOG.info(f"[compat] Dropped unsupported TrainingArguments keys: {dropped}")
    return TrainingArguments(**filtered)


# =========================
# SMALL UTILS
# =========================
def set_seeds(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def gpu_supports_bf16() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 8  # Ampere+


def last_word(line: str) -> Optional[str]:
    ws = WORD_RE.findall(line or "")
    return ws[-1].lower() if ws else None


def first_line_only(text: str) -> str:
    t = (text or "").strip()
    if "\n" in t:
        t = t.split("\n", 1)[0].strip()
    return t


def ends_with_word(line: str, w: str) -> bool:
    lw = last_word(line)
    return lw == (w.lower() if w else w)


def sanitize_line(line: str) -> str:
    # keep as-is except strip trailing whitespace
    return (line or "").rstrip()


# Reservoir sampling for caps
def keep_with_cap(buf: List[Any], item: Any, cap: Optional[int], seen: int) -> int:
    """
    Maintains a uniform sample of size cap from a stream.
    Returns updated seen count.
    """
    seen += 1
    if cap is None:
        buf.append(item)
        return seen
    if len(buf) < cap:
        buf.append(item)
        return seen
    # replace with probability cap/seen
    j = random.randint(1, seen)
    if j <= cap:
        buf[j - 1] = item
    return seen


# =========================
# DICTIONARY + RHYME BUCKETS
# =========================
def rhyme_key_from_stressed(stressed_word: str) -> Optional[str]:
    # Stress marked by ' before the stressed letter: "ч'аса"
    if "'" not in stressed_word:
        return None
    idx = stressed_word.find("'")
    clean = stressed_word.replace("'", "")
    stressed_pos = idx  # after removing ', index matches

    # find the stressed vowel index
    stressed_vowel_idx = None
    for i in range(stressed_pos, len(clean)):
        if clean[i] in VOWELS:
            stressed_vowel_idx = i
            break
    if stressed_vowel_idx is None:
        return None

    # syllable onset: start right after previous vowel (or word start)
    start = 0
    for j in range(stressed_vowel_idx - 1, -1, -1):
        if clean[j] in VOWELS:
            start = j + 1
            break

    return clean[start:].lower()


def fallback_rhyme_key(word: str) -> Optional[str]:
    w = (word or "").lower()
    if len(w) >= 3:
        return w[-3:]
    if len(w) >= 2:
        return w[-2:]
    return None


def load_dictionary_and_buckets(dict_file: str) -> Tuple[Dict[str, str], Dict[str, List[str]], set]:
    """
    Returns:
      word_to_stressed: word -> stressed form (with ')
      rhyme_bucket: rhyme_key -> list of words
      dict_words: set of words in dictionary
    """
    LOG.info(f"[dict] Loading {dict_file}")
    word_to_stressed: Dict[str, str] = {}
    dict_words: set = set()

    with open(dict_file, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            w = (row.get("word") or "").strip().lower()
            sw = (row.get("word_stressed") or "").strip()
            if not w:
                continue
            dict_words.add(w)
            if sw:
                word_to_stressed[w] = sw

    rhyme_bucket: Dict[str, List[str]] = {}
    for w in dict_words:
        k = None
        if w in word_to_stressed:
            k = rhyme_key_from_stressed(word_to_stressed[w])
        if not k:
            k = fallback_rhyme_key(w)
        if not k:
            continue
        rhyme_bucket.setdefault(k, []).append(w)

    LOG.info(f"[dict] words={len(dict_words)} stressed={len(word_to_stressed)} rhyme_keys={len(rhyme_bucket)}")
    return word_to_stressed, rhyme_bucket, dict_words


def pick_two_distinct_from_bucket(words: List[str]) -> Optional[Tuple[str, str]]:
    ws = list(set(words))
    if len(ws) < 2:
        return None
    a, b = random.sample(ws, 2)
    return a, b


def pick_rhyme_pair(rhyme_bucket: Dict[str, List[str]], avoid: set) -> Optional[Tuple[str, str, str]]:
    keys = [k for k, ws in rhyme_bucket.items() if len(ws) >= MIN_BUCKET]
    if not keys:
        return None
    for _ in range(60):
        k = random.choice(keys)
        pair = pick_two_distinct_from_bucket(rhyme_bucket[k])
        if not pair:
            continue
        w1, w2 = pair
        if w1 in avoid or w2 in avoid:
            continue
        return k, w1, w2
    return None


def pick_x_y_pairs(rhyme_bucket: Dict[str, List[str]]) -> Optional[Tuple[str, str, str, str]]:
    # pick x1,x2 then y1,y2, all distinct if possible
    avoid: set = set()
    px = pick_rhyme_pair(rhyme_bucket, avoid)
    if not px:
        return None
    _, x1, x2 = px
    avoid.update([x1, x2])

    py = pick_rhyme_pair(rhyme_bucket, avoid)
    if not py:
        # fallback: allow overlap with x's but still ensure y1!=y2
        py = pick_rhyme_pair(rhyme_bucket, set())
        if not py:
            return None
        _, y1, y2 = py
        if y1 == y2:
            return None
        return x1, x2, y1, y2

    _, y1, y2 = py
    return x1, x2, y1, y2


# =========================
# PROMPTS (T1..T4)
# =========================
def prompt_T1(end_word: str) -> str:
    return (
        "Крайна дума:\n"
        f"{end_word}\n\n"
        "Напиши един поетичен стих (само един ред и нищо друго), който завършва ТОЧНО на крайната дума.\n"
    )


def prompt_Tk(context_lines: List[str], end_word: str) -> str:
    # For T2/T3/T4 (next line given context)
    ctx = "\n".join([f"{i+1}) {sanitize_line(l)}" for i, l in enumerate(context_lines)])
    return (
        "Контекст:\n"
        f"{ctx}\n\n"
        "Крайна дума:\n"
        f"{end_word}\n\n"
        "Напиши следващ поетичен стих (само един ред и нищо друго), който е смислово свързан с контекста "
        "и завършва ТОЧНО на крайната дума.\n"
    )


# =========================
# PARSING POEMS (stanza-based)
# =========================
def iter_poem_files(poems_dir: str) -> Iterable[str]:
    for root, _, files in os.walk(poems_dir):
        for fn in files:
            if fn.lower().endswith((".txt", ".md")):
                yield os.path.join(root, fn)


def split_stanzas_from_lines(lines: List[str]) -> List[List[str]]:
    stanzas: List[List[str]] = []
    cur: List[str] = []
    for raw in lines:
        s = raw.rstrip("\n")
        if s.strip() == "":
            if cur:
                stanzas.append(cur)
                cur = []
            continue
        cur.append(sanitize_line(s))
    if cur:
        stanzas.append(cur)
    return stanzas


def iter_stanza_lines_from_file(path: str) -> Iterable[List[str]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        with open(path, "r", encoding="utf-8-sig") as f:
            lines = f.readlines()
    for st in split_stanzas_from_lines(lines):
        # drop very short stanzas? keep all
        yield st


# =========================
# PARSING QUATRAINS FILE
# =========================
_QUA_HDR_RE = re.compile(r"^FILE:\s*(.*?)\s*\|\s*stanza:\s*(\d+)\s*\|\s*start_line:\s*(\d+)\s*\|\s*scheme:\s*(ABAB|AABB)\s*$")
_QUA_KEYS_RE = re.compile(r"^\s*\(rhyme keys\)\s*(.+)\s*$")
_QUA_LINE_RE = re.compile(r"^\s*\[\s*(\d+)\]\s*(.*\S)\s*$")


def iter_quatrains(quatrains_file: str) -> Iterable[Dict[str, Any]]:
    """
    Yields dicts: {file, stanza, start_line, scheme, keys[4], lines[4]}
    """
    if not os.path.exists(quatrains_file):
        LOG.info(f"[quatrains] Not found: {quatrains_file} (skipping)")
        return

    with open(quatrains_file, "r", encoding="utf-8") as f:
        cur: Dict[str, Any] = {}
        lines_acc: List[str] = []
        keys_acc: List[str] = []

        def flush():
            nonlocal cur, lines_acc, keys_acc
            if cur and len(lines_acc) == 4 and len(keys_acc) == 4:
                cur["lines"] = lines_acc
                cur["keys"] = keys_acc
                yield cur
            cur = {}
            lines_acc = []
            keys_acc = []

        for raw in f:
            s = raw.rstrip("\n")
            if s.strip() == "":
                # stanza block separator
                for item in flush():
                    yield item
                continue

            m = _QUA_HDR_RE.match(s)
            if m:
                # flush any previous
                for item in flush():
                    yield item
                cur = {
                    "file": m.group(1),
                    "stanza": int(m.group(2)),
                    "start_line": int(m.group(3)),
                    "scheme": m.group(4),
                }
                continue

            mk = _QUA_KEYS_RE.match(s.strip())
            if mk:
                ks = mk.group(1).strip().split()
                if len(ks) >= 4:
                    keys_acc = ks[:4]
                continue

            ml = _QUA_LINE_RE.match(s.strip())
            if ml:
                lines_acc.append(sanitize_line(ml.group(2)))
                continue

        # final flush
        for item in flush():
            yield item


# =========================
# TRAINING TEXT (chat template)
# =========================
def build_chat_training_text(tokenizer, prompt: str, answer_line: str) -> str:
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": sanitize_line(answer_line)},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


# =========================
# DATASET BUILD (T1..T4)
# =========================
def extract_T_examples_from_stanza(
    stanza_lines: List[str],
    dict_words: set,
    caps: Dict[str, Optional[int]],
    buffers: Dict[str, List[Dict[str, Any]]],
    seen: Dict[str, int],
):
    """
    From one stanza, create:
      T1 for each line
      T2 for each consecutive pair
      T3 for each consecutive triple
      T4 for each consecutive quadruple
    We store raw examples as dicts with: task, prompt, answer, end_word, context_len.
    """
    n = len(stanza_lines)
    if n == 0:
        return

    # T1
    for i in range(n):
        line = stanza_lines[i]
        w = last_word(line)
        if not w:
            continue
        if REQUIRE_ENDINGS_IN_DICT and w not in dict_words:
            continue
        ex = {"task": "T1", "prompt": prompt_T1(w), "answer": line, "end_word": w, "context_len": 0}
        seen["T1"] = keep_with_cap(buffers["T1"], ex, caps["T1"], seen["T1"])

    # T2..T4 as next-line prediction with forced ending
    for k, task in [(2, "T2"), (3, "T3"), (4, "T4")]:
        if n < k:
            continue
        for i in range(n - k + 1):
            ctx = stanza_lines[i:i + k - 1]
            target = stanza_lines[i + k - 1]
            w = last_word(target)
            if not w:
                continue
            if REQUIRE_ENDINGS_IN_DICT and w not in dict_words:
                continue
            # sanity: target must end with that word
            if not ends_with_word(target, w):
                continue
            ex = {"task": task, "prompt": prompt_Tk(ctx, w), "answer": target, "end_word": w, "context_len": k - 1}
            seen[task] = keep_with_cap(buffers[task], ex, caps[task], seen[task])


def build_mixed_raw_examples(
    poems_dir: str,
    quatrains_file: str,
    dict_words: set,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Returns buffers per task: {"T1":[...], "T2":[...], "T3":[...], "T4":[...]}
    """
    LOG.info("[data] Building raw examples from poems + quatrains")
    caps = {"T1": MAX_T1, "T2": MAX_T2, "T3": MAX_T3, "T4": MAX_T4}
    buffers = {"T1": [], "T2": [], "T3": [], "T4": []}
    seen = {"T1": 0, "T2": 0, "T3": 0, "T4": 0}

    # Poems
    files = list(iter_poem_files(poems_dir))
    LOG.info(f"[data] Found poem files: {len(files)}")
    for idx, path in enumerate(files, start=1):
        if idx % 200 == 0:
            LOG.info(f"[data] Poems progress: {idx}/{len(files)}")
        for stanza in iter_stanza_lines_from_file(path):
            extract_T_examples_from_stanza(stanza, dict_words, caps, buffers, seen)

    # Quatrains file (clean quads)
    if os.path.exists(quatrains_file):
        LOG.info(f"[data] Adding examples from quatrains file: {quatrains_file}")
        q_seen = 0
        for q in iter_quatrains(quatrains_file):
            q_seen += 1
            if MAX_FROM_QUATRAINS is not None and q_seen > MAX_FROM_QUATRAINS:
                break
            lines4 = q.get("lines", [])
            if len(lines4) != 4:
                continue
            extract_T_examples_from_stanza(lines4, dict_words, caps, buffers, seen)
        LOG.info(f"[data] Quatrains processed: {q_seen}")
    else:
        LOG.info("[data] Quatrains file missing; skipping")

    for t in ["T1", "T2", "T3", "T4"]:
        LOG.info(f"[data] Raw buffer {t}: {len(buffers[t])} (seen stream={seen[t]})")

    return buffers


def dedupe_examples(examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # dedupe by (task, end_word, answer) and also prompt hash
    seen = set()
    out = []
    for ex in examples:
        key = (ex["task"], ex["end_word"], ex["answer"])
        if key in seen:
            continue
        seen.add(key)
        out.append(ex)
    return out


def balance_and_merge(buffers: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Simple balancing by oversampling smaller tasks to approximate TASK_WEIGHTS.
    """
    # dedupe each task
    for t in buffers:
        before = len(buffers[t])
        buffers[t] = dedupe_examples(buffers[t])
        after = len(buffers[t])
        if after != before:
            LOG.info(f"[data] Dedupe {t}: {before} -> {after}")

    sizes = {t: len(buffers[t]) for t in buffers}
    LOG.info(f"[data] Sizes before balancing: {sizes}")

    # If any task is empty, we can't balance; just merge what we have
    if any(sizes[t] == 0 for t in ["T1", "T2", "T3", "T4"]):
        LOG.info("[data] Some tasks are empty; skipping balancing")
        merged = []
        for t in ["T1", "T2", "T3", "T4"]:
            merged.extend(buffers[t])
        random.shuffle(merged)
        return merged

    total = sum(sizes.values())
    # target total: keep as-is unless MAX_FINAL_TRAIN_EXAMPLES set
    target_total = total if MAX_FINAL_TRAIN_EXAMPLES is None else min(total, MAX_FINAL_TRAIN_EXAMPLES)

    # compute desired counts per task
    desired = {t: int(target_total * TASK_WEIGHTS[t]) for t in TASK_WEIGHTS}
    # adjust rounding
    diff = target_total - sum(desired.values())
    if diff != 0:
        desired["T1"] += diff

    LOG.info(f"[data] Target_total={target_total} desired={desired}")

    merged: List[Dict[str, Any]] = []
    for t in ["T1", "T2", "T3", "T4"]:
        pool = buffers[t]
        if not pool:
            continue
        need = desired[t]
        if need <= len(pool):
            sample = random.sample(pool, need)
        else:
            # oversample with replacement
            sample = [random.choice(pool) for _ in range(need)]
        merged.extend(sample)

    random.shuffle(merged)
    LOG.info(f"[data] Final merged examples: {len(merged)}")
    return merged


def to_hf_dataset(tokenizer, examples: List[Dict[str, Any]]) -> Dataset:
    rows = []
    skipped = 0
    for ex in examples:
        try:
            text = build_chat_training_text(tokenizer, ex["prompt"], ex["answer"])
        except Exception:
            skipped += 1
            continue
        rows.append({"task": ex["task"], "text": text})
    LOG.info(f"[data] Built chat-text rows={len(rows)} skipped={skipped}")
    return Dataset.from_list(rows)


# =========================
# TOKENIZATION + COLLATOR (assistant-only loss)
# =========================
def _find_sublist(haystack: List[int], needle: List[int]) -> int:
    n = len(needle)
    if n == 0:
        return -1
    for i in range(len(haystack) - n + 1):
        if haystack[i:i + n] == needle:
            return i
    return -1


class TokenizedAssistantOnlyCollator:
    def __init__(self, tokenizer, assistant_marker: str = ASSISTANT_MARKER):
        self.tokenizer = tokenizer
        self.marker_ids = tokenizer.encode(assistant_marker, add_special_tokens=False)

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        pad_id = self.tokenizer.pad_token_id
        input_ids = [torch.tensor(ex["input_ids"], dtype=torch.long) for ex in examples]
        attention_mask = [torch.tensor(ex["attention_mask"], dtype=torch.long) for ex in examples]

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)

        labels = input_ids.clone()
        for i in range(labels.size(0)):
            ids = labels[i].tolist()
            start = _find_sublist(ids, self.marker_ids)
            if start == -1:
                labels[i, :] = -100
                continue
            start = start + len(self.marker_ids)
            labels[i, :start] = -100
            labels[i, attention_mask[i] == 0] = -100

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def tokenize_dataset(ds: Dataset, tokenizer) -> Dataset:
    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_SEQ_LEN, padding=False)
    return ds.map(tok, batched=True, remove_columns=["text"])


def filter_missing_marker(ds: Dataset, tokenizer) -> Dataset:
    marker_ids = tokenizer.encode(ASSISTANT_MARKER, add_special_tokens=False)

    def has_marker(ex):
        return _find_sublist(ex["input_ids"], marker_ids) != -1

    before = len(ds)
    ds2 = ds.filter(has_marker)
    LOG.info(f"[data] Filtered missing-marker: {before} -> {len(ds2)}")
    return ds2


# =========================
# CHECKPOINT HELPERS
# =========================
def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    if not os.path.isdir(output_dir):
        return None
    ckpts = []
    for name in os.listdir(output_dir):
        if name.startswith("checkpoint-"):
            try:
                step = int(name.split("checkpoint-")[-1])
                ckpts.append((step, os.path.join(output_dir, name)))
            except ValueError:
                continue
    if not ckpts:
        return None
    ckpts.sort(key=lambda x: x[0])
    return ckpts[-1][1]


def checkpoint_lora_rank(ckpt_dir: str) -> Optional[int]:
    cfg_path = os.path.join(ckpt_dir, "adapter_config.json")
    if not os.path.exists(cfg_path):
        return None
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        if "r" in cfg:
            return int(cfg["r"])
        if "lora_r" in cfg:
            return int(cfg["lora_r"])
    except Exception:
        return None
    return None


# =========================
# MODEL LOADING
# =========================
def load_tokenizer() -> AutoTokenizer:
    LOG.info("[model] Loading tokenizer")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_default_system_prompt=False)
    # Many instruction models don't have a separate PAD token
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok


def load_base_model(dtype: torch.dtype):
    LOG.info(f"[model] Loading base model dtype={dtype} attn={ATTN_IMPLEMENTATION}")
    m = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=dtype,
        attn_implementation=ATTN_IMPLEMENTATION,
    )
    return m


def load_tuned_model(dtype: torch.dtype):
    if not os.path.isdir(FINAL_ADAPTER_DIR):
        raise RuntimeError(
            f"Adapter not found: {FINAL_ADAPTER_DIR}\n"
            f"Run MODE='finetune' first (or adjust FINAL_ADAPTER_DIR)."
        )
    LOG.info(f"[model] Loading tuned model from {FINAL_ADAPTER_DIR}")
    base = load_base_model(dtype)
    tuned = PeftModel.from_pretrained(base, FINAL_ADAPTER_DIR)
    return tuned


# =========================
# GENERATION (attention_mask-safe)
# =========================
@torch.inference_mode()
def gen_chat(model, tokenizer, user_prompt: str) -> str:
    messages = [{"role": "user", "content": user_prompt}]
    enc = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,  # provides attention_mask
    )
    input_ids = enc["input_ids"].to(model.device)
    attention_mask = enc["attention_mask"].to(model.device)
    input_len = input_ids.shape[1]

    eos_ids = [tokenizer.eos_token_id]
    eot = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    if eot is not None and eot != tokenizer.unk_token_id:
        eos_ids.append(eot)

    out = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,  # avoids pad==eos warning
        max_new_tokens=GEN_MAX_NEW_TOKENS,
        do_sample=GEN_DO_SAMPLE,
        temperature=GEN_TEMPERATURE if GEN_DO_SAMPLE else None,
        top_p=GEN_TOP_P if GEN_DO_SAMPLE else None,
        repetition_penalty=GEN_REPETITION_PENALTY,
        no_repeat_ngram_size=GEN_NO_REPEAT_NGRAM,
        eos_token_id=eos_ids,
        pad_token_id=tokenizer.eos_token_id,
    )
    new = out[0, input_len:]
    return tokenizer.decode(new, skip_special_tokens=True).strip()


def gen_line_with_retries(model, tokenizer, prompt: str, must_end: str) -> Tuple[str, bool]:
    best = ""
    for _ in range(GEN_MAX_ATTEMPTS):
        raw = gen_chat(model, tokenizer, prompt)
        line = first_line_only(raw)
        best = line
        if ends_with_word(line, must_end):
            return line, True
    return best, ends_with_word(best, must_end)


def generate_abab_4step(model, tokenizer, x1: str, x2: str, y1: str, y2: str) -> Tuple[List[str], List[bool]]:
    # 1) line1 ends x1
    l1, ok1 = gen_line_with_retries(model, tokenizer, prompt_T1(x1), x1)

    # 2) line2 related to line1 ends y1
    l2, ok2 = gen_line_with_retries(model, tokenizer, prompt_Tk([l1], y1), y1)

    # 3) line3 related to line1+2 ends x2
    l3, ok3 = gen_line_with_retries(model, tokenizer, prompt_Tk([l1, l2], x2), x2)

    # 4) line4 related to line1+2+3 ends y2
    l4, ok4 = gen_line_with_retries(model, tokenizer, prompt_Tk([l1, l2, l3], y2), y2)

    return [l1, l2, l3, l4], [ok1, ok2, ok3, ok4]


def run_generation_demo(model, tokenizer, rhyme_bucket: Dict[str, List[str]], title: str):
    print("\n" + "#" * 90)
    print(title)
    print("ABAB 4-step: pick (x1,x2) rhyming and (y1,y2) rhyming; then 4 model calls.")
    print("#" * 90)
    
    
    with open(DEMO_OUT_FILE, "a", encoding="utf-8") as f:

        model.eval()
        for i in range(DEMO_SAMPLES if MODE == "finetune" else GEN_COUNT):
            
            f.write("\n" + "#" * 90 + "\n\n")
                        
            picked = pick_x_y_pairs(rhyme_bucket)
            if not picked:
                print("Could not pick rhyme pairs (check MIN_BUCKET / dictionary).")
                return
            x1, x2, y1, y2 = picked
            lines, oks = generate_abab_4step(model, tokenizer, x1, x2, y1, y2)

            print("\n" + "=" * 80)
            print(f"DEMO {i+1}/{DEMO_SAMPLES if MODE == 'finetune' else GEN_COUNT}:")
            print(f"- x1={x1}  x2={x2}   (rhyming)")
            print(f"- y1={y1}  y2={y2}   (rhyming)")
            print(f"- OK endings: {oks}")
            for j, ln in enumerate(lines, start=1):
                f.write(f"{j}) {ln}\n")
                print(f"{j}) {ln}")
            print("=" * 80)
            f.write("=" * 80 + "\n")


def run_accuracy_test(base_model, tuned_model, tokenizer, rhyme_bucket: Dict[str, List[str]], samples: int) -> Dict[str, Any]:
    """
    Runs accuracy evaluation over `samples` trials.
    Accuracy is defined as the fraction of trials where all four line endings are correct.
    Also reports per-position success rates.
    """
    base_model.eval()
    tuned_model.eval()
    trials = 0
    base_allok = 0
    tuned_allok = 0
    base_pos_ok = [0, 0, 0, 0]
    tuned_pos_ok = [0, 0, 0, 0]

    for i in range(samples):
        picked = pick_x_y_pairs(rhyme_bucket)
        if not picked:
            LOG.warning("[test] Could not pick rhyme pairs (check MIN_BUCKET / dictionary). Stopping.")
            break
        x1, x2, y1, y2 = picked
        LOG.info(f"[test] sample {i+1}/{samples} picked x=({x1},{x2}) y=({y1},{y2})")

        lines, oks = generate_abab_4step(base_model, tokenizer, x1, x2, y1, y2)
        targets = [x1, y1, x2, y2]
        trials += 1
        for j, ok in enumerate(oks):
            if ok:
                base_pos_ok[j] += 1
        # per-step logs with expected endings
        for j, (ln, ok, tgt) in enumerate(zip(lines, oks, targets)):
            status = "OK" if ok else "FAIL"
            LOG.info(f"[base  test]   step {j+1}: {status} expected='{tgt}' line='{ln}'")

        if all(oks):
            base_allok += 1
            
        lines, oks = generate_abab_4step(tuned_model, tokenizer, x1, x2, y1, y2)
        targets = [x1, y1, x2, y2]
        for j, ok in enumerate(oks):
            if ok:
                tuned_pos_ok[j] += 1
        # per-step logs with expected endings
        for j, (ln, ok, tgt) in enumerate(zip(lines, oks, targets)):
            status = "OK" if ok else "FAIL"
            LOG.info(f"[tuned test]   step {j+1}: {status} expected='{tgt}' line='{ln}'")

        if all(oks):
            tuned_allok += 1

    base_acc = (base_allok / trials) if trials else 0.0
    base_per_pos = [(base_pos_ok[i] / trials) if trials else 0.0 for i in range(4)]

    tuned_acc = (tuned_allok / trials) if trials else 0.0
    tuned_per_pos = [(tuned_pos_ok[i] / trials) if trials else 0.0 for i in range(4)]
    metrics = {
        "samples": trials,
        "base_accuracy_all_ok": base_acc,
        "tuned_accuracy_all_ok": tuned_acc,
        "base_per_position_ok_rate": {
            "base_line1": base_per_pos[0],
            "base_line2": base_per_pos[1],
            "base_line3": base_per_pos[2],
            "base_line4": base_per_pos[3],
        },
        "tuned_per_position_ok_rate": {
            "tuned_line1": tuned_per_pos[0],
            "tuned_line2": tuned_per_pos[1],
            "tuned_line3": tuned_per_pos[2],
            "tuned_line4": tuned_per_pos[3],
        },
    }

    LOG.info(f"[test] samples={metrics['samples']} all-ok-acc={metrics['base_accuracy_all_ok']:.3f} "
             f"per-pos={[round(x, 3) for x in base_per_pos]}")
    print("\n" + "#" * 90)
    print("TEST RESULTS (ABAB all-endings accuracy)")
    print("#" * 90)
    print(f"Samples: {metrics['samples']}")
    print(f"All-four OK accuracy BASE: {metrics['base_accuracy_all_ok']:.3f}")
    print("Per-position OK rates:")
    print(f"  line1: {metrics['base_per_position_ok_rate']['base_line1']:.3f}")
    print(f"  line2: {metrics['base_per_position_ok_rate']['base_line2']:.3f}")
    print(f"  line3: {metrics['base_per_position_ok_rate']['base_line3']:.3f}")
    print(f"  line4: {metrics['base_per_position_ok_rate']['base_line4']:.3f}")
    print(f"All-four OK accuracy TUNED: {metrics['tuned_accuracy_all_ok']:.3f}")
    print("Per-position OK rates:")
    print(f"  line1: {metrics['tuned_per_position_ok_rate']['tuned_line1']:.3f}")
    print(f"  line2: {metrics['tuned_per_position_ok_rate']['tuned_line2']:.3f}")
    print(f"  line3: {metrics['tuned_per_position_ok_rate']['tuned_line3']:.3f}")
    print(f"  line4: {metrics['tuned_per_position_ok_rate']['tuned_line4']:.3f}")

    return metrics


# =========================
# MAIN
# =========================
def main():
    setup_logging()
    set_seeds(SEED)

    LOG.info("[init] Starting")
    LOG.info(f"[init] MODE={MODE}")
    LOG.info(f"[init] MODEL_ID={MODEL_ID}")
    LOG.info(f"[init] POEMS_DIR={POEMS_DIR}")
    LOG.info(f"[init] QUATRAINS_FILE={QUATRAINS_FILE}")
    LOG.info(f"[init] OUTPUT_DIR={OUTPUT_DIR}")

    # Dictionary and rhyme buckets
    _, rhyme_bucket, dict_words = load_dictionary_and_buckets(DICT_FILE)

    # Tokenizer
    tokenizer = load_tokenizer()

    # dtype
    use_bf16 = gpu_supports_bf16()
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    LOG.info(f"[init] dtype={dtype} (bf16_supported={use_bf16})")

    if MODE == "generate":
        model = load_tuned_model(dtype)
        run_generation_demo(model, tokenizer, rhyme_bucket, "GENERATION MODE (TUNED MODEL)")
        LOG.info("[done] generate mode finished")
        return

    if MODE == "test":
        base_model = load_base_model(dtype)
        tuned_model = load_tuned_model(dtype)
        n = TEST_SAMPLES if TEST_SAMPLES is not None else GEN_COUNT
        run_accuracy_test(base_model=base_model, tuned_model=tuned_model, tokenizer=tokenizer, rhyme_bucket=rhyme_bucket, samples=n)
        LOG.info("[done] test mode finished")
        return

    if MODE != "finetune":
        raise ValueError("MODE must be 'finetune' or 'generate'")

    # -------------------------
    # FINETUNE MODE
    # -------------------------
    base_model = load_base_model(dtype)

    # Baseline demo (optional but useful)
    run_generation_demo(base_model, tokenizer, rhyme_bucket, "BASELINE DEMO (BEFORE FINETUNING)")

    # Build dataset
    t0 = time.time()
    buffers = build_mixed_raw_examples(POEMS_DIR, QUATRAINS_FILE, dict_words)
    merged = balance_and_merge(buffers)
    LOG.info(f"[data] Dataset build time: {time.time() - t0:.1f}s")

    ds = to_hf_dataset(tokenizer, merged)
    if len(ds) < 2000:
        raise RuntimeError(f"Too few training examples ({len(ds)}). Check paths / dictionary coverage.")

    split = ds.train_test_split(test_size=VAL_RATIO, seed=SEED)
    train_raw, eval_raw = split["train"], split["test"]
    LOG.info(f"[data] train={len(train_raw)} val={len(eval_raw)}")

    LOG.info("[data] Tokenizing")
    train_tok = filter_missing_marker(tokenize_dataset(train_raw, tokenizer), tokenizer)
    eval_tok = filter_missing_marker(tokenize_dataset(eval_raw, tokenizer), tokenizer)

    collator = TokenizedAssistantOnlyCollator(tokenizer)

    # Attach LoRA
    LOG.info("[ft] Attaching LoRA")
    base_model.config.use_cache = False
    base_model.train()

    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    ft_model = get_peft_model(base_model, lora_cfg)

    # Training args
    LOG.info("[train] Building TrainingArguments")
    args = make_training_args(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        per_device_eval_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        max_grad_norm=MAX_GRAD_NORM,
        logging_steps=LOGGING_STEPS,
        evaluation_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        report_to=["tensorboard"],
        bf16=use_bf16,
        fp16=not use_bf16,
        optim="adamw_torch",
        seed=SEED,
        remove_unused_columns=False,
    )

    # Save run meta
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    meta = {
        "MODE": MODE,
        "MODEL_ID": MODEL_ID,
        "POEMS_DIR": POEMS_DIR,
        "DICT_FILE": DICT_FILE,
        "QUATRAINS_FILE": QUATRAINS_FILE,
        "TASK_WEIGHTS": TASK_WEIGHTS,
        "CAPS": {"MAX_T1": MAX_T1, "MAX_T2": MAX_T2, "MAX_T3": MAX_T3, "MAX_T4": MAX_T4},
        "MAX_SEQ_LEN": MAX_SEQ_LEN,
        "EPOCHS": EPOCHS,
        "LR": LR,
        "BATCH": PER_DEVICE_BATCH,
        "GRAD_ACCUM": GRAD_ACCUM,
        "LORA": {"r": LORA_R, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT},
        "ATTN_IMPLEMENTATION": ATTN_IMPLEMENTATION,
    }
    with open(os.path.join(OUTPUT_DIR, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    trainer = Trainer(
        model=ft_model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        data_collator=collator,
    )

    # Resume if compatible checkpoint exists
    latest = find_latest_checkpoint(OUTPUT_DIR)
    resume_ckpt = None
    if latest:
        ckpt_r = checkpoint_lora_rank(latest)
        if ckpt_r == LORA_R:
            resume_ckpt = latest
            LOG.info(f"[ckpt] Resuming from {latest}")
        else:
            LOG.info(f"[ckpt] Found {latest} but LoRA r mismatch ({ckpt_r} != {LORA_R}); not resuming")
    else:
        LOG.info("[ckpt] No checkpoint found; training from scratch")

    # Train
    LOG.info("[train] Starting training")
    t1 = time.time()
    trainer.train(resume_from_checkpoint=resume_ckpt)
    LOG.info(f"[train] Finished in {time.time() - t1:.1f}s")

    # Save final adapter
    os.makedirs(FINAL_ADAPTER_DIR, exist_ok=True)
    LOG.info(f"[save] Saving final adapter to {FINAL_ADAPTER_DIR}")
    trainer.save_model(FINAL_ADAPTER_DIR)
    tokenizer.save_pretrained(FINAL_ADAPTER_DIR)

    ft_model.eval()

    # Post-finetune demo
    run_generation_demo(ft_model, tokenizer, rhyme_bucket, "POST-FINETUNE DEMO (TUNED MODEL)")
    LOG.info("[done] finetune mode finished")


if __name__ == "__main__":
    main()
