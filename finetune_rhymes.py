# finetune_two_step_couplet.py
#
# Two-step couplet generation:
#   Call 1: given ending word <w1>, write ONE poetic line ending EXACTLY with <w1>
#   Call 2: given ending word <w2> and line1 context, write ONE poetic line RELATED to line1,
#           ending EXACTLY with <w2>
#
# Modes:
#   - MODE = "finetune": runs baseline demo with base model, then finetunes LoRA, then demo with tuned
#   - MODE = "generate": loads tuned adapter from OUTPUT_DIR/final and runs demo only (no training)
#
# No CLI args: edit constants.

import os
import re
import csv
import json
import time
import random
import logging
import inspect
from typing import Dict, List, Optional, Tuple, Any

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, PeftModel


# =========================
# CONSTANTS (edit these)
# =========================
MODE = "finetune"  # "finetune" or "generate"

MODEL_ID = "INSAIT-Institute/BgGPT-Gemma-2-2.6B-IT-v1.0"
PAIRS_FILE = "rhyming_pairs_from_quatrains.txt"
DICT_FILE = "dictionary.csv"

OUTPUT_DIR = "./bggpt-two-step-couplet-lora-fp16"
FINAL_ADAPTER_DIR = os.path.join(OUTPUT_DIR, "final")

SEED = 42
VAL_RATIO = 0.05
MAX_SAMPLES = None
BIDIRECTIONAL = True  # add swapped direction

# Training
MAX_SEQ_LEN = 256
EPOCHS = 2
LR = 2e-5
PER_DEVICE_BATCH = 2
GRAD_ACCUM = 8
WARMUP_RATIO = 0.05
MAX_GRAD_NORM = 0.3
WEIGHT_DECAY = 0.0

# Logging / eval / checkpoints
LOGGING_STEPS = 10
EVAL_STEPS = 100
SAVE_STEPS = 100
SAVE_TOTAL_LIMIT = 3

# Demo
DEMO_SAMPLES = 4

# Generation
GEN_MAX_NEW_TOKENS = 56
GEN_DO_SAMPLE = True
GEN_TEMPERATURE = 0.9
GEN_TOP_P = 0.95
GEN_REPETITION_PENALTY = 1.12
GEN_NO_REPEAT_NGRAM = 3
GEN_MAX_ATTEMPTS = 5
GEN_MAX_ATTEMPTS_LINE = 4

ATTN_IMPLEMENTATION = "eager"

# LoRA
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

# Dictionary selection
MIN_BUCKET = 30
REQUIRE_ENDINGS_IN_DICT_FOR_TRAIN = True

ASSISTANT_MARKER = "<start_of_turn>model\n"


# =========================
# LOGGING
# =========================
LOG = logging.getLogger("two_step_couplet")
LOG.setLevel(logging.INFO)
LOG.propagate = False


def setup_logging():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    fh = logging.FileHandler(os.path.join(OUTPUT_DIR, "train.log"), encoding="utf-8")
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
    if "evaluation_strategy" in kw and "evaluation_strategy" not in _TA_PARAMS and "eval_strategy" in _TA_PARAMS:
        kw["eval_strategy"] = kw.pop("evaluation_strategy")
    filtered = {k: v for k, v in kw.items() if k in _TA_PARAMS}
    dropped = sorted(set(kw.keys()) - set(filtered.keys()))
    if dropped:
        LOG.info(f"[compat] Dropped unsupported TrainingArguments keys: {dropped}")
    return TrainingArguments(**filtered)


# =========================
# TEXT / RHYME UTILS
# =========================
PAIR_LINE_RE = re.compile(r"^\s*\[\s*\d+\]\s*(.*\S)\s*$")
WORD_RE = re.compile(r"[а-яА-ЯёЁ]+", re.UNICODE)
VOWELS = set("аеиоуъюяАЕИОУЪЮЯ")


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
    lw = last_word(line or "")
    return lw == (w.lower() if w else w)


def rhyme_key_from_stressed(stressed_word: str) -> Optional[str]:
    if "'" not in stressed_word:
        return None
    idx = stressed_word.find("'")
    clean = stressed_word.replace("'", "")
    stressed_pos = idx
    for i in range(stressed_pos, len(clean)):
        if clean[i] in VOWELS:
            return clean[i:].lower()
    return None


def fallback_rhyme_key(word: str) -> Optional[str]:
    for i in range(len(word) - 1, -1, -1):
        if word[i] in VOWELS:
            return word[i:].lower()
    return None


def build_rhyme_key(word: str, word_to_stressed: Dict[str, str]) -> Optional[str]:
    sw = word_to_stressed.get(word)
    if sw:
        k = rhyme_key_from_stressed(sw)
        if k:
            return k
    return fallback_rhyme_key(word)


# =========================
# CORPUS PARSING
# =========================
def parse_pairs_file(path: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    buf: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            s = raw.rstrip("\n")
            if s.strip() == "":
                if len(buf) >= 2:
                    m1 = PAIR_LINE_RE.match(buf[0])
                    m2 = PAIR_LINE_RE.match(buf[1])
                    if m1 and m2:
                        pairs.append((m1.group(1).strip(), m2.group(1).strip()))
                buf = []
                continue
            if PAIR_LINE_RE.match(s):
                buf.append(s)

    if len(buf) >= 2:
        m1 = PAIR_LINE_RE.match(buf[0])
        m2 = PAIR_LINE_RE.match(buf[1])
        if m1 and m2:
            pairs.append((m1.group(1).strip(), m2.group(1).strip()))
    return pairs


def dedupe_pairs(pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    seen = set()
    out = []
    for a, b in pairs:
        key = (a.strip(), b.strip())
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


# =========================
# DICTIONARY + RHYME WORDPAIR PICKING
# =========================
def load_dictionary(dict_file: str) -> Tuple[Dict[str, str], Dict[str, List[str]], set]:
    LOG.info(f"[dict] Loading {dict_file}")
    word_to_stressed: Dict[str, str] = {}
    all_words: set = set()

    with open(dict_file, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            w = (row.get("word") or "").strip().lower()
            sw = (row.get("word_stressed") or "").strip()
            if not w:
                continue
            all_words.add(w)
            if sw:
                word_to_stressed[w] = sw

    rhyme_bucket: Dict[str, List[str]] = {}
    for w in all_words:
        k = build_rhyme_key(w, word_to_stressed)
        if not k:
            continue
        rhyme_bucket.setdefault(k, []).append(w)

    LOG.info(f"[dict] words={len(all_words)} stressed={len(word_to_stressed)} rhyme_keys={len(rhyme_bucket)}")
    return word_to_stressed, rhyme_bucket, all_words


def pick_rhyming_word_pair(rhyme_bucket: Dict[str, List[str]]) -> Optional[Tuple[str, str, str]]:
    # Collect keys with enough words and at least two words of length >= 4
    eligible_keys = []
    long_words_by_key: Dict[str, List[str]] = {}

    for k, ws in rhyme_bucket.items():
        if len(ws) < MIN_BUCKET:
            continue
        long_ws = [w.strip() for w in set(ws) if len(w.strip()) >= 4]
        if len(long_ws) >= 2:
            eligible_keys.append(k)
            long_words_by_key[k] = long_ws

    if not eligible_keys:
        return None

    k = random.choice(eligible_keys)
    long_ws = long_words_by_key[k]
    w1, w2 = random.sample(long_ws, 2)
    return k, w1, w2


# =========================
# PROMPTS (two separate calls)
# =========================
def prompt_line1_from_w1(w1: str) -> str:
    return (
        "Крайна дума:\n"
        f"{w1}\n\n"
        "Напиши един поетичен стих (само един ред и нищо друго), който ЗАВЪРШВА ТОЧНО на крайната дума.\n"
    )


def prompt_line2_from_w2_and_line1(w2: str, line1: str) -> str:
    return (
        "Първи стих:\n"
        f"{line1}\n\n"
        "Крайна дума за втория стих:\n"
        f"{w2}\n\n"
        "Напиши втори поетичен стих (само един ред и нищо друго), който е СМИСЛОВО СВЪРЗАН с първия стих "
        "и ЗАВЪРШВА ТОЧНО на крайната дума.\n"
    )


# =========================
# GENERATION (with attention_mask)
# =========================
@torch.inference_mode()
def gen_chat(model, tokenizer, user_prompt: str) -> str:
    messages = [{"role": "user", "content": user_prompt}]
    enc = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,  # <-- ensures attention_mask
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
        attention_mask=attention_mask,  # <-- pass it to avoid warning
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


def gen_line_with_retries(model, tokenizer, prompt: str, must_end: str) -> str:
    best = ""
    for _ in range(GEN_MAX_ATTEMPTS_LINE):
        print(f"[gen] Generating attempt {_+1}/{GEN_MAX_ATTEMPTS_LINE} for line ending with '{must_end}'")
        raw = gen_chat(model, tokenizer, prompt)
        line = first_line_only(raw)
        best = line
        print(line)
        if ends_with_word(line, must_end):
            print()
            return line
    return best


def generate_two_step_couplet(model, tokenizer, w1: str, w2: str) -> Tuple[str, str, bool, bool]:
    line1 = gen_line_with_retries(model, tokenizer, prompt_line1_from_w1(w1), w1)
    ok1 = ends_with_word(line1, w1)

    line2 = gen_line_with_retries(model, tokenizer, prompt_line2_from_w2_and_line1(w2, line1), w2)
    ok2 = ends_with_word(line2, w2)

    return line1, line2, ok1, ok2


# =========================
# TRAINING DATA (two tasks)
# =========================
def build_chat_training_text_line1(tokenizer, w1: str, line1: str) -> str:
    messages = [
        {"role": "user", "content": prompt_line1_from_w1(w1)},
        {"role": "assistant", "content": line1.strip()},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def build_chat_training_text_line2(tokenizer, w2: str, line1: str, line2: str) -> str:
    messages = [
        {"role": "user", "content": prompt_line2_from_w2_and_line1(w2, line1)},
        {"role": "assistant", "content": line2.strip()},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


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


def build_train_dataset(
    tokenizer,
    pairs: List[Tuple[str, str]],
    dict_words: set,
) -> Dataset:
    rows = []
    kept = skipped = 0

    for line1, line2 in pairs:
        w1 = last_word(line1)
        w2 = last_word(line2)
        if not w1 or not w2:
            skipped += 1
            continue
        if not ends_with_word(line1, w1) or not ends_with_word(line2, w2):
            skipped += 1
            continue
        if w1 == w2:
            skipped += 1
            continue

        if REQUIRE_ENDINGS_IN_DICT_FOR_TRAIN:
            if w1 not in dict_words or w2 not in dict_words:
                skipped += 1
                continue

        rows.append({"task": "line1", "text": build_chat_training_text_line1(tokenizer, w1, line1)})
        rows.append({"task": "line2", "text": build_chat_training_text_line2(tokenizer, w2, line1, line2)})
        kept += 2

        if BIDIRECTIONAL:
            rows.append({"task": "line1", "text": build_chat_training_text_line1(tokenizer, w2, line2)})
            rows.append({"task": "line2", "text": build_chat_training_text_line2(tokenizer, w1, line2, line1)})
            kept += 2

    if MAX_SAMPLES is not None and len(rows) > MAX_SAMPLES:
        random.shuffle(rows)
        rows = rows[:MAX_SAMPLES]

    LOG.info(f"[data] Train rows kept={kept} skipped={skipped} final={len(rows)}")
    return Dataset.from_list(rows)


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
# MODEL LOADING HELPERS (modes)
# =========================
def load_base_tokenizer() -> AutoTokenizer:
    LOG.info("[model] Loading tokenizer")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_default_system_prompt=False)
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
            f"FINAL_ADAPTER_DIR not found: {FINAL_ADAPTER_DIR}\n"
            f"Run MODE='finetune' first, or point FINAL_ADAPTER_DIR to your adapter folder."
        )
    LOG.info(f"[model] Loading base model + adapter from {FINAL_ADAPTER_DIR}")
    base = load_base_model(dtype)
    tuned = PeftModel.from_pretrained(base, FINAL_ADAPTER_DIR)
    return tuned


# =========================
# DEMO RUNNER
# =========================
def run_demo(model, tokenizer, rhyme_bucket: Dict[str, List[str]], title: str):
    print("\n" + "#" * 90)
    print(title)
    print("Two calls: Call1(w1)->line1 ; Call2(line1,w2)->line2")
    print("#" * 90)

    model.eval()
    for i in range(DEMO_SAMPLES):
        picked = pick_rhyming_word_pair(rhyme_bucket)
        if not picked:
            print("Could not pick a rhyme pair (check MIN_BUCKET / dictionary).")
            break
        rk, w1, w2 = picked

        attempts = 0
        while True:
            print(f"[demo] Generating attempt {attempts+1}/{GEN_MAX_ATTEMPTS} couplet for rhyme key '{rk}' with words '{w1}' and '{w2}'", end="\r")
            l1, l2, ok1, ok2 = generate_two_step_couplet(model, tokenizer, w1, w2)

            if ok1 and ok2:
                print()
                print("\n" + "=" * 80)
                print(f"DEMO {i+1}/{DEMO_SAMPLES}")
                print(f"- Rhyme key: {rk}")
                print(f"- w1: {w1}")
                print(f"- w2: {w2}")
                print(f"- Call1 OK: {ok1} | Call2 OK: {ok2}")
                print("- Output:")
                print(l1)
                print(l2)
                print("=" * 80)
                break
            attempts += 1
            if attempts >= GEN_MAX_ATTEMPTS:
                break

# =========================
# MAIN
# =========================
def main():
    setup_logging()
    set_seeds(SEED)

    LOG.info("[init] Starting")
    LOG.info(f"[init] MODE={MODE}")
    LOG.info(f"[init] MODEL_ID={MODEL_ID}")
    LOG.info(f"[init] OUTPUT_DIR={OUTPUT_DIR}")

    # Dictionary (needed for both modes)
    _, rhyme_bucket, dict_words = load_dictionary(DICT_FILE)

    # Tokenizer
    tokenizer = load_base_tokenizer()

    # dtype
    use_bf16 = gpu_supports_bf16()
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    if MODE == "generate":
        # Load tuned model and run demo only
        model = load_tuned_model(dtype)
        run_demo(model, tokenizer, rhyme_bucket, "GENERATION MODE (TUNED MODEL)")
        LOG.info("[done] generation mode finished")
        return

    if MODE != "finetune":
        raise ValueError("MODE must be 'finetune' or 'generate'")

    # =========================
    # FINETUNE MODE
    # =========================
    # Load base model
    base_model = load_base_model(dtype)

    # Load training corpus
    LOG.info(f"[data] Reading pairs from {PAIRS_FILE}")
    pairs = dedupe_pairs(parse_pairs_file(PAIRS_FILE))
    if not pairs:
        raise RuntimeError(f"No pairs parsed from {PAIRS_FILE}.")
    LOG.info(f"[data] pairs={len(pairs)}")

    # Build dataset
    LOG.info("[data] Building training dataset (two tasks)")
    full_ds = build_train_dataset(tokenizer, pairs, dict_words)
    if len(full_ds) < 1200:
        raise RuntimeError(
            f"Too few training examples ({len(full_ds)}). "
            f"Try REQUIRE_ENDINGS_IN_DICT_FOR_TRAIN=False or check dictionary coverage."
        )

    split = full_ds.train_test_split(test_size=VAL_RATIO, seed=SEED)
    train_raw, eval_raw = split["train"], split["test"]
    LOG.info(f"[data] train={len(train_raw)} val={len(eval_raw)}")

    LOG.info("[data] Tokenizing")
    train_tok = filter_missing_marker(tokenize_dataset(train_raw, tokenizer), tokenizer)
    eval_tok = filter_missing_marker(tokenize_dataset(eval_raw, tokenizer), tokenizer)
    collator = TokenizedAssistantOnlyCollator(tokenizer)

    # LoRA
    LOG.info("[ft] Attaching LoRA adapters (no quant)")
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

    meta_path = os.path.join(OUTPUT_DIR, "run_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "MODE": MODE,
                "MODEL_ID": MODEL_ID,
                "PAIRS_FILE": PAIRS_FILE,
                "DICT_FILE": DICT_FILE,
                "objective": "two_step_generation: (w1)->line1 and (line1,w2)->line2",
                "MAX_SEQ_LEN": MAX_SEQ_LEN,
                "EPOCHS": EPOCHS,
                "LR": LR,
                "BIDIRECTIONAL": BIDIRECTIONAL,
                "REQUIRE_ENDINGS_IN_DICT_FOR_TRAIN": REQUIRE_ENDINGS_IN_DICT_FOR_TRAIN,
                "LORA": {"r": LORA_R, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT},
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    trainer = Trainer(
        model=ft_model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        data_collator=collator,
    )

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

    LOG.info("[train] Starting training")
    t0 = time.time()
    trainer.train(resume_from_checkpoint=resume_ckpt)
    LOG.info(f"[train] Finished in {time.time() - t0:.1f}s")

    os.makedirs(FINAL_ADAPTER_DIR, exist_ok=True)
    LOG.info(f"[save] Saving final adapter to {FINAL_ADAPTER_DIR}")
    trainer.save_model(FINAL_ADAPTER_DIR)
    tokenizer.save_pretrained(FINAL_ADAPTER_DIR)

    # Post-finetune demo (same process, tuned model)
    ft_model.eval()
    run_demo(ft_model, tokenizer, rhyme_bucket, "POST-FINETUNE DEMO (TUNED MODEL)")

    LOG.info("[done] finetune mode finished")


if __name__ == "__main__":
    main()
