"""
Backward LSTM Language Model for Poetic Line Completion
==============================================================

Goal
----
- Train a decoder-only LSTM on reversed token sequences of poem lines,
  so that at inference time, given a suffix (the line ending), the model
  generates the missing beginning of the line.

Key ideas
---------
- Tokenization via existing SentencePiece model `poems_cond_bpe.model`.
- Reverse tokens within each line and append EOS. Train causal LM with BOS.
- At inference, given suffix S: reverse its tokens, prepend BOS, then generate
  until EOS using beam search. The generated tokens (reversed domain) are
  reversed back to produce the missing prefix. Finally decode the full line
  (prefix + suffix) with SentencePiece.
- Regularization to avoid overfitting: dropout, token dropout, weight decay, validation split,
  early stopping, gradient clipping, and small model defaults.

Usage
-----
- Configure constants in the CONFIG section below (no argparse).
- Set `ACTION` to one of: "train", "eval", "complete".
- For completion, set `SUFFIX`, `MAX_NEW_TOKENS`, and `BEAM_SIZE`.
"""

from __future__ import annotations

import glob
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional
import csv
import re
import unicodedata
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
	import sentencepiece as spm
except Exception as e:  # pragma: no cover
	spm = None  # defer error until actually used


# ------------------------- Defaults / Config -------------------------
SPM_MODEL = "poems_cond_bpe.model"  # expected to exist in repo
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model capacity (smaller for memory constraints)
D_MODEL = 384
N_HEAD = 8
N_LAYER = 8
D_FF = 4 * D_MODEL
DROPOUT = 0.3

CTX_LEN = 160  # max sequence length for training/decoding (reversed domain)

LR = 3e-4
WEIGHT_DECAY = 0.01
CLIP_NORM = 0.5
LABEL_SMOOTH = 0.05
BATCH_SIZE = 512
EPOCHS = 100
EARLY_STOP_PATIENCE = 5

CKPT_PATH = "backwards_transformer.pt"

# Data / run config (no argparse)
DATA_DIR = "chitanka_poems_step1"
VAL_FRACTION = 0.10

# Choose one of: "train" | "eval" | "complete"
ACTION = "train"

# Completion settings
SUFFIX = ""  # set to the ending of a line you want to complete
MAX_NEW_TOKENS = 60
BEAM_SIZE = 5

# Preview generation settings (example at start of each epoch)
PREVIEW_EACH_EPOCH = True
PREVIEW_SUFFIX = "слънцето"
PREVIEW_MAX_NEW = 60
PREVIEW_BEAM = 5

# Rhyme search config
RHYME_CANDIDATES = 10

# Logging config
LOG_FILE = "backwards_model.log"
LOG_LEVEL = logging.INFO
# SentencePiece training config
SPM_PREFIX = "poems_cond_bpe"
SPM_TRAIN_TXT = "spm_train.txt"
SPM_VOCAB_SIZE = 8000
SPM_CHAR_COVERAGE = 0.9995
TRAIN_SPM_IF_NEEDED = True

# Regularization: token dropout
TOKEN_DROPOUT_P = 0.2
DEDUPE_GLOBAL = True


# Optional LR scheduler
USE_SCHEDULER = True
WARMUP_STEPS = 500


def setup_logger() -> logging.Logger:
	logger = logging.getLogger("backwards_model")
	if logger.handlers:
		return logger
	logger.setLevel(LOG_LEVEL)
	fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
	fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
	fh.setFormatter(fmt)
	sh = logging.StreamHandler()
	sh.setFormatter(fmt)
	logger.addHandler(fh)
	logger.addHandler(sh)
	return logger


LOG = setup_logger()


def torch_load_safe(path: str, map_location=DEVICE):
	"""Secure wrapper for torch.load using weights_only=True when available.
	Falls back to default torch.load if the installed version does not support
	weights_only, with a log note.
	"""
	try:
		obj = torch.load(path, map_location=map_location, weights_only=True)
		LOG.info(f"[ckpt] loaded safely (weights_only=True): {path}")
		return obj
	except TypeError:
		LOG.info("[ckpt] torch.load does not support weights_only; falling back (less safe)")
		return torch.load(path, map_location=map_location)


def make_scheduler(opt: torch.optim.Optimizer, warmup_steps: int, total_steps: int):
	"""Linear warmup followed by cosine decay to 0."""
	from math import pi, cos

	warmup_steps = max(1, warmup_steps)
	total_steps = max(warmup_steps + 1, total_steps)

	def lr_lambda(step: int):
		if step < warmup_steps:
			return float(step) / float(warmup_steps)
		progress = float(step - warmup_steps) / float(total_steps - warmup_steps)
		return 0.5 * (1.0 + cos(pi * min(1.0, progress)))

	# Always start fresh (last_epoch=-1) so scheduler injects initial_lr in param groups
	return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda, last_epoch=-1)


# ------------------------- SentencePiece training -------------------------
def concat_poems_to_file(data_dir: str, out_path: str) -> int:
	"""Concatenate all poem files under data_dir into a single text file."""
	LOG.info(f"[spm] collecting text from: {data_dir}")
	files = sorted(glob.glob(os.path.join(data_dir, "*.txt")))
	if not files:
		raise RuntimeError(f"[spm] no poem files found in {data_dir}")
	parts = []
	for fp in files:
		parts.append(Path(fp).read_text(encoding="utf-8", errors="ignore"))
	text = "\n".join(parts)
	Path(out_path).write_text(text, encoding="utf-8")
	LOG.info(f"[spm] wrote {out_path} chars={len(text)} files={len(files)}")
	return len(text)


def train_spm_if_needed(model_path: str, data_dir: str):
	"""Train SentencePiece BPE tokenizer if model_path does not exist."""
	if spm is None:
		raise RuntimeError("sentencepiece is required. Install: pip install sentencepiece")
	if Path(model_path).exists():
		LOG.info(f"[spm] tokenizer present: {model_path}")
		return
	LOG.info("[spm] training tokenizer...")
	concat_poems_to_file(data_dir, SPM_TRAIN_TXT)
	spm.SentencePieceTrainer.train(
		input=SPM_TRAIN_TXT,
		model_prefix=SPM_PREFIX,
		vocab_size=SPM_VOCAB_SIZE,
		model_type="bpe",
		character_coverage=SPM_CHAR_COVERAGE,
		pad_id=0,
		bos_id=1,
		eos_id=2,
		unk_id=3,
	)
	LOG.info(f"[spm] trained tokenizer: {model_path}")


# ------------------------- Data utilities -------------------------
def _collapse_ws(s: str) -> str:
	import re
	return re.sub(r"\s+", " ", s).strip()


def _bad_latin_or_digits(s: str) -> bool:
	import re
	return re.search(r"[A-Za-z0-9]", s) is not None


def load_lines_from_dir(data_dir: str) -> List[str]:
	"""
	Load and lightly clean poem lines from a directory of .txt files.
	- drop empty lines
	- drop lines containing Latin letters or digits (to keep Bulgarian only)
	"""
	LOG.info(f"[data] scanning directory: {data_dir}")
	files = sorted(glob.glob(os.path.join(data_dir, "*.txt")))
	if not files:
		raise FileNotFoundError(f"No .txt files found under {data_dir}")

	LOG.info(f"[data] found {len(files)} files")
	lines: List[str] = []
	kept = 0
	dropped_empty = 0
	dropped_latin = 0
	for fp in files:
		txt = Path(fp).read_text(encoding="utf-8", errors="ignore")
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
	if DEDUPE_GLOBAL and lines:
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
	rng = random.Random(seed)
	idx = list(range(len(items)))
	rng.shuffle(idx)
	n_val = max(1, int(len(items) * val_fraction))
	val_idx = set(idx[:n_val])
	train = [items[i] for i in range(len(items)) if i not in val_idx]
	val = [items[i] for i in range(len(items)) if i in val_idx]
	LOG.info(f"[split] train={len(train)} val={len(val)}")
	return train, val


# ------------------------- Tokenization -------------------------
@dataclass
class Tokenizer:
	sp: "spm.SentencePieceProcessor"
	pad_id: int
	bos_id: int
	eos_id: int
	unk_id: int

	@classmethod
	def from_spm(cls, model_path: str) -> "Tokenizer":
		if spm is None:
			raise RuntimeError("sentencepiece is required. Install: pip install sentencepiece")
		sp = spm.SentencePieceProcessor()
		if not Path(model_path).exists():
			raise FileNotFoundError(f"SentencePiece model not found: {model_path}")
		sp.load(model_path)
		return cls(sp=sp, pad_id=sp.pad_id(), bos_id=sp.bos_id(), eos_id=sp.eos_id(), unk_id=sp.unk_id())

	def encode(self, text: str) -> List[int]:
		return self.sp.encode(text, out_type=int)

	def decode(self, ids: List[int]) -> str:
		return self.sp.decode(ids)


# ------------------------- Dataset -------------------------
class ReversedLineDataset(torch.utils.data.Dataset):
	"""
	Provides (input_ids, target_ids) pairs for causal LM training in reversed domain.
	For a tokenized line t1..tN, build reversed = [tN, ..., t1, EOS].
	We feed BOS + reversed[:-1] as input and reversed as target.
	Sequences are padded/truncated to CTX_LEN.
	"""

	def __init__(self, lines: List[str], tok: Tokenizer, ctx_len: int, is_train: bool):
		self.tok = tok
		self.ctx_len = ctx_len
		self.is_train = is_train
		# Pre-encode for speed
		self.samples: List[Tuple[List[int], List[int]]] = []
		LOG.info(f"[dataset] building reversed samples: n_lines={len(lines)} ctx_len={ctx_len}")
		total_len = 0
		for ln in lines:
			ids = tok.encode(ln)
			if not ids:
				continue
			rev = list(reversed(ids)) + [tok.eos_id]
			inp = [tok.bos_id] + rev[:-1]
			tgt = rev
			self.samples.append((inp, tgt))
			total_len += len(rev)
		if self.samples:
			avg_len = total_len / len(self.samples)
		else:
			avg_len = 0
		LOG.info(f"[dataset] samples={len(self.samples)} avg_rev_len={avg_len:.1f}")

	def __len__(self) -> int:
		return len(self.samples)

	def __getitem__(self, idx: int):
		inp, tgt = self.samples[idx]
		inp = inp[: self.ctx_len]
		tgt = tgt[: self.ctx_len]
		# pad
		pad = self.tok.pad_id
		if len(inp) < self.ctx_len:
			inp = inp + [pad] * (self.ctx_len - len(inp))
		if len(tgt) < self.ctx_len:
			tgt = tgt + [pad] * (self.ctx_len - len(tgt))
		# token dropout on inputs (replace random non-pad, non-BOS tokens with UNK)
		if self.is_train and TOKEN_DROPOUT_P > 0:
			unk = self.tok.unk_id
			bos = self.tok.bos_id
			for i in range(len(inp)):
				if inp[i] != pad and inp[i] != bos:
					if random.random() < TOKEN_DROPOUT_P:
						inp[i] = unk
		attn_mask = [0 if x == pad else 1 for x in inp]
		return (
			torch.tensor(inp, dtype=torch.long),
			torch.tensor(tgt, dtype=torch.long),
			torch.tensor(attn_mask, dtype=torch.bool),
		)


def make_loaders(train_lines: List[str], val_lines: List[str], tok: Tokenizer, ctx_len: int, batch_size: int):
	LOG.info(f"[loader] creating loaders batch_size={batch_size}")
	train_ds = ReversedLineDataset(train_lines, tok, ctx_len, is_train=True)
	val_ds = ReversedLineDataset(val_lines, tok, ctx_len, is_train=False)
	train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
	val_dl = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
	LOG.info(f"[loader] train_batches=~{max(1, len(train_ds)//max(1,batch_size))} val_batches=~{max(1, len(val_ds)//max(1,batch_size))}")
	return train_dl, val_dl


# ------------------------- Model (LSTM) -------------------------
class LSTMDecoderLM(nn.Module):
	def __init__(self, vocab_size: int, d_model: int, n_layer: int, dropout: float):
		super().__init__()
		self.tok_emb = nn.Embedding(vocab_size, d_model)
		self.drop = nn.Dropout(dropout)
		# LSTM with batch_first; dropout applies between layers when n_layer > 1
		self.rnn = nn.LSTM(
			input_size=d_model,
			hidden_size=d_model,
			num_layers=n_layer,
			batch_first=True,
			dropout=dropout if n_layer > 1 else 0.0,
		)
		self.ln_f = nn.LayerNorm(d_model)
		self.head = nn.Linear(d_model, vocab_size, bias=False)
		# weight tying for regularization
		self.head.weight = self.tok_emb.weight

	def forward(self, idx: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
		# attn_mask is ignored for LSTM; loss function handles pad masking
		x = self.tok_emb(idx)
		x = self.drop(x)
		x, _ = self.rnn(x)
		x = self.ln_f(x)
		logits = self.head(x)
		return logits


# ------------------------- Training / Evaluation -------------------------
def loss_fn(logits: torch.Tensor, targets: torch.Tensor, pad_id: int, label_smooth: float = 0.0) -> torch.Tensor:
	vocab_size = logits.size(-1)
	logits = logits.view(-1, vocab_size)
	targets = targets.view(-1)
	ignore = targets == pad_id
	if label_smooth > 0:
		# Label smoothing cross-entropy
		with torch.no_grad():
			true_dist = torch.zeros_like(logits)
			true_dist.fill_(label_smooth / (vocab_size - 1))
			true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - label_smooth)
			true_dist[ignore] = 0
		log_probs = F.log_softmax(logits, dim=-1)
		loss = -(true_dist * log_probs).sum(dim=-1)
		loss = loss.masked_select(~ignore).mean()
		return loss
	else:
		loss = F.cross_entropy(logits, targets, ignore_index=pad_id)
		return loss


@torch.no_grad()
def evaluate(model: GPTDecoderLM, loader, pad_id: int) -> Tuple[float, float]:
	model.eval()
	total_loss = 0.0
	total_tokens = 0
	step = 0
	for inp, tgt, att in loader:
		inp = inp.to(DEVICE)
		tgt = tgt.to(DEVICE)
		att = att.to(DEVICE)
		logits = model(inp, att)
		loss = loss_fn(logits, tgt, pad_id, label_smooth=0.0)
		# Count non-pad target tokens to aggregate loss per token
		nonpad = (tgt != pad_id).sum().item()
		step += 1
		LOG.info(f"[eval] step={step} loss={loss.item():.4f} nonpad={nonpad}")
		total_loss += loss.item() * max(1, nonpad)
		total_tokens += max(1, nonpad)
	avg = total_loss / max(1, total_tokens)
	ppl = math.exp(min(20.0, avg))
	LOG.info(f"[eval] summary val_loss={avg:.4f} val_ppl={ppl:.2f}")
	return avg, ppl


def train_model(
	train_dl,
	val_dl,
	tok: Tokenizer,
	d_model: int = D_MODEL,
	n_head: int = N_HEAD,
	n_layer: int = N_LAYER,
	d_ff: int = D_FF,
	dropout: float = DROPOUT,
	epochs: int = EPOCHS,
	lr: float = LR,
	weight_decay: float = WEIGHT_DECAY,
	label_smooth: float = LABEL_SMOOTH,
	clip_norm: float = CLIP_NORM,
	ckpt_path: str = CKPT_PATH,
) -> GPTDecoderLM:
	vocab = tok.sp.vocab_size()
	LOG.info(
		f"[model:init] device={DEVICE} vocab={vocab} d_model={d_model} n_head={n_head} n_layer={n_layer} d_ff={d_ff} dropout={dropout} ctx_len={CTX_LEN}"
	)
	model = LSTMDecoderLM(vocab, d_model, n_layer, dropout).to(DEVICE)
	opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
	LOG.info(f"[optim] AdamW lr={lr} weight_decay={weight_decay} label_smooth={label_smooth} clip_norm={clip_norm}")

	# Scheduler (initialize with global step for proper resume)
	scheduler = None
	steps_per_epoch = max(1, len(train_dl))
	global_step_start = 0
	if USE_SCHEDULER:
		total_steps = max(1, epochs * steps_per_epoch)
		# will create after potential resume, then advance to global_step_start

	# Try resume from checkpoint if available
	start_epoch = 1
	best_val = float("inf")
	patience = 0
	if Path(ckpt_path).exists():
		LOG.info(f"[resume] attempting to resume from {ckpt_path}")
		try:
			ckpt = torch_load_safe(ckpt_path, map_location=DEVICE)
			cfg = ckpt.get("cfg", {})
			if cfg:
				LOG.info(f"[resume] ckpt cfg: {cfg}")
			# Prefer last model state for resume
			if "model" in ckpt:
				model.load_state_dict(ckpt["model"]) 
				LOG.info("[resume] loaded model state (last)")
			elif "best_model" in ckpt:
				model.load_state_dict(ckpt["best_model"]) 
				LOG.info("[resume] loaded best_model state (no last model present)")
			if "optim" in ckpt:
				opt.load_state_dict(ckpt["optim"]) 
				LOG.info("[resume] loaded optimizer state")
			start_epoch = int(ckpt.get("epoch", 1))
			best_val = float(ckpt.get("best_val", best_val))
			patience = int(ckpt.get("patience", patience))
			LOG.info(f"[resume] start_epoch={start_epoch} best_val={best_val:.4f} patience={patience}")
		except Exception as e:
			LOG.info(f"[resume] failed to load checkpoint: {e}")

	# Now that we know the start_epoch, set up scheduler with correct global step
	if USE_SCHEDULER:
		global_step_start = (start_epoch - 1) * steps_per_epoch
		total_steps = max(1, epochs * steps_per_epoch)
		scheduler = make_scheduler(opt, WARMUP_STEPS, total_steps)
		# Advance scheduler to resumed global step so LR matches prior training
		if global_step_start > 0:
			try:
				scheduler.step(global_step_start)
			except Exception as e:
				LOG.info(f"[sched] step({global_step_start}) failed: {e}; setting last_epoch directly")
				scheduler.last_epoch = global_step_start
		LOG.info(f"[sched] enabled warmup_steps={WARMUP_STEPS} total_steps={total_steps} start_step={global_step_start}")

	for ep in range(start_epoch, epochs + 1):
		LOG.info(f"[train] epoch={ep}/{epochs} start")

		# Generate a preview completion before training each epoch
		if PREVIEW_EACH_EPOCH and PREVIEW_SUFFIX:
			try:
				model.eval()
				preview_text = complete_suffix(
					tok, model, PREVIEW_SUFFIX, max_new=PREVIEW_MAX_NEW, beam=PREVIEW_BEAM
				)
				LOG.info(f"[preview] suffix='{PREVIEW_SUFFIX}' -> {preview_text}")

				# Also suggest a rhyme and generate a line ending with it
				try:
					candidates = get_rhymes(PREVIEW_SUFFIX, topn=RHYME_CANDIDATES)
					if candidates:
						rhyme_word = candidates[0]
						preview_rhyme_text = complete_suffix(
							tok, model, rhyme_word, max_new=PREVIEW_MAX_NEW, beam=PREVIEW_BEAM
						)
						LOG.info(f"[preview-rhyme] rhyme='{rhyme_word}' -> {preview_rhyme_text}")
					else:
						LOG.info("[preview-rhyme] no rhyme candidates found")
				except Exception as e:
					LOG.info(f"[preview-rhyme] failed: {e}")
			except Exception as e:
				LOG.info(f"[preview] generation failed: {e}")
			finally:
				model.train()
		model.train()
		total_train_loss = 0.0
		train_tokens = 0
		steps = 0
		for inp, tgt, att in train_dl:
			inp = inp.to(DEVICE)
			tgt = tgt.to(DEVICE)
			att = att.to(DEVICE)
			logits = model(inp, att)
			loss = loss_fn(logits, tgt, tok.pad_id, label_smooth=label_smooth)

			opt.zero_grad(set_to_none=True)
			loss.backward()
			nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
			opt.step()
			if scheduler is not None:
				scheduler.step()
				if steps % 100 == 0:
					LOG.info(f"[sched] lr={opt.param_groups[0]['lr']:.6f}")

			nonpad = (tgt != tok.pad_id).sum().item()
			total_train_loss += loss.item() * max(1, nonpad)
			train_tokens += max(1, nonpad)
			steps += 1
			if steps % 10 == 0:
				LOG.info(f"[train] epoch={ep} step={steps} loss={loss.item():.4f}")

		val_loss, val_ppl = evaluate(model, val_dl, tok.pad_id)
		avg_tr = total_train_loss / max(1, train_tokens)
		gap = val_loss - avg_tr
		LOG.info(f"[train] epoch={ep} summary train_loss={avg_tr:.4f} val_loss={val_loss:.4f} val_ppl={val_ppl:.2f} gen_gap={gap:.4f}")

		# Save resume checkpoint (last state)
		torch.save({
			"model": model.state_dict(),
			"optim": opt.state_dict(),
			"epoch": ep + 1,
			"best_val": best_val,
			"patience": patience,
			"cfg": {
				"vocab": tok.sp.vocab_size(),
				"d_model": d_model,
				"n_layer": n_layer,
				"dropout": dropout,
			},
		}, ckpt_path)
		LOG.info(f"[ckpt] saved resume state to {ckpt_path}")

		if val_loss + 1e-6 < best_val:
			best_val = val_loss
			patience = 0
			# also store best_model in same file
			try:
				ckpt = torch_load_safe(ckpt_path, map_location=DEVICE)
			except Exception:
				ckpt = {}
			ckpt["best_model"] = model.state_dict()
			ckpt["best_val"] = best_val
			# keep cfg
			ckpt["cfg"] = ckpt.get("cfg", {
				"vocab": tok.sp.vocab_size(),
				"d_model": d_model,
				"n_layer": n_layer,
				"dropout": dropout,
			})
			torch.save(ckpt, ckpt_path)
			LOG.info(f"[ckpt] updated best_model in {ckpt_path}")
		else:
			patience += 1
			if patience >= EARLY_STOP_PATIENCE:
				LOG.info("[early_stop] patience reached; stopping training")
				break

	# Load best
	if Path(ckpt_path).exists():
		LOG.info(f"[ckpt] loading best from {ckpt_path}")
		ckpt = torch_load_safe(ckpt_path, map_location=DEVICE)
		cfg = ckpt["cfg"]
		best = LSTMDecoderLM(
			cfg["vocab"], cfg["d_model"], cfg["n_layer"], cfg["dropout"]
		)
		state = ckpt.get("best_model", ckpt.get("model"))
		best.load_state_dict(state)
		best.to(DEVICE)
		return best
	return model


def load_model(ckpt_path: str) -> LSTMDecoderLM:
	LOG.info(f"[ckpt] load_model path={ckpt_path}")
	ckpt = torch_load_safe(ckpt_path, map_location=DEVICE)
	cfg = ckpt["cfg"]
	model = LSTMDecoderLM(
		cfg["vocab"], cfg["d_model"], cfg["n_layer"], cfg["dropout"]
	)
	state = ckpt.get("best_model", ckpt.get("model"))
	model.load_state_dict(state)
	model.to(DEVICE)
	model.eval()
	return model


# ------------------------- Decoding (beam search) -------------------------
@torch.no_grad()
def beam_search_generate(
	model: LSTMDecoderLM,
	prompt: List[int],
	pad_id: int,
	eos_id: int,
	max_new_tokens: int = 60,
	beam_size: int = 5,
	length_penalty: float = 0.8,
) -> List[int]:
	"""
	Beam search over the reversed-domain sequence, starting from prompt tokens.
	Returns the sequence including the prompt and the generated tokens.
	Stops when EOS is generated (first beam that produces EOS with best score).
	"""
	LOG.info(f"[decode] beam_search start max_new={max_new_tokens} beam={beam_size} len_pen={length_penalty}")
	model.eval()
	device = next(model.parameters()).device
	prompt = prompt[-CTX_LEN:]
	beams = [(prompt, 0.0, False)]  # (tokens, logprob, ended)

	for _ in range(max_new_tokens):
		new_beams = []
		for tokens, score, ended in beams:
			if ended:
				new_beams.append((tokens, score, True))
				continue
			inp = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
			att = torch.tensor([1] * len(tokens), dtype=torch.bool, device=device).unsqueeze(0)
			logits = model(inp, att)[:, -1, :]
			logp = F.log_softmax(logits, dim=-1).squeeze(0)
			topk = torch.topk(logp, k=min(beam_size, logp.size(-1)))
			for next_id, lp in zip(topk.indices.tolist(), topk.values.tolist()):
				ntoks = tokens + [next_id]
				ended_flag = next_id == eos_id
				# length penalty on score to avoid bias towards short outputs
				new_score = (score + lp) / ((len(ntoks) ** length_penalty))
				new_beams.append((ntoks, new_score, ended_flag))
		# keep best beams
		new_beams.sort(key=lambda x: x[1], reverse=True)
		beams = new_beams[:beam_size]
		LOG.info(f"[decode] step beams_top_scores={[round(b[1],3) for b in beams]}")
		# If best beam ended, optionally early stop if multiple ended
		if any(b[2] for b in beams) and beams[0][2]:
			LOG.info("[decode] early stop: best beam ended")
			break

	# choose best finished if available else best partial
	finished = [b for b in beams if b[2]]
	best = max(finished, key=lambda x: x[1]) if finished else max(beams, key=lambda x: x[1])
	LOG.info(f"[decode] finished ended={best[2]} length={len(best[0])}")
	return best[0]


def complete_suffix(tok: Tokenizer, model: LSTMDecoderLM, suffix: str, max_new: int = 60, beam: int = 5) -> str:
	# Encode suffix, reverse to reversed-domain prefix
	suf_ids = tok.encode(suffix)
	if not suf_ids:
		return suffix
	rev_prefix = list(reversed(suf_ids))
	prompt = [tok.bos_id] + rev_prefix
	LOG.info(f"[complete] suffix='{suffix}' tokens={len(suf_ids)} prompt_len={len(prompt)}")
	out = beam_search_generate(model, prompt, tok.pad_id, tok.eos_id, max_new_tokens=max_new, beam_size=beam)
	gen = out[len(prompt):]
	# Cut at EOS if present
	if tok.eos_id in gen:
		cut = gen.index(tok.eos_id)
		gen = gen[:cut]
	# Reverse back to original order = the missing prefix tokens
	prefix_tokens = list(reversed(gen))
	full_tokens = prefix_tokens + suf_ids
	text = tok.decode(full_tokens)
	LOG.info(f"[complete] generated prefix_tokens={len(prefix_tokens)} full_len={len(full_tokens)}")
	LOG.info(f"[complete] text={text}")
	return text


def main():
	# All settings are taken from constants at the top.
	LOG.info(
		f"[main] action={ACTION} device={DEVICE} data_dir={DATA_DIR} spm={SPM_MODEL} ckpt={CKPT_PATH}"
	)
	if TRAIN_SPM_IF_NEEDED:
		train_spm_if_needed(SPM_MODEL, DATA_DIR)
	tok = Tokenizer.from_spm(SPM_MODEL)

	if ACTION == "train":
		all_lines = load_lines_from_dir(DATA_DIR)
		train_lines, val_lines = split_train_val(all_lines, val_fraction=VAL_FRACTION, seed=42)
		train_dl, val_dl = make_loaders(train_lines, val_lines, tok, CTX_LEN, BATCH_SIZE)
		_ = train_model(
			train_dl,
			val_dl,
			tok,
			d_model=D_MODEL,
			n_head=N_HEAD,
			n_layer=N_LAYER,
			d_ff=D_FF,
			dropout=DROPOUT,
			epochs=EPOCHS,
			lr=LR,
			weight_decay=WEIGHT_DECAY,
			label_smooth=LABEL_SMOOTH,
			clip_norm=CLIP_NORM,
			ckpt_path=CKPT_PATH,
		)
		LOG.info(f"[main] training done; best checkpoint at {CKPT_PATH}")

	elif ACTION == "eval":
		if not Path(CKPT_PATH).exists():
			raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}. Train first.")
		all_lines = load_lines_from_dir(DATA_DIR)
		_, val_lines = split_train_val(all_lines, val_fraction=VAL_FRACTION, seed=42)
		_, val_dl = make_loaders([], val_lines, tok, CTX_LEN, BATCH_SIZE)
		model = load_model(CKPT_PATH)
		val_loss, val_ppl = evaluate(model, val_dl, tok.pad_id)
		LOG.info(f"[main] eval done val_loss={val_loss:.4f} val_ppl={val_ppl:.2f}")

	elif ACTION == "complete":
		if not Path(CKPT_PATH).exists():
			raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}. Train first.")
		if SUFFIX is None:
			LOG.info("Please set SUFFIX constant to the desired line ending.")
			return
		model = load_model(CKPT_PATH)
		out = complete_suffix(tok, model, SUFFIX, max_new=MAX_NEW_TOKENS, beam=BEAM_SIZE)
		LOG.info(f"[main] completion output: {out}")
	else:
		raise ValueError("ACTION must be one of: 'train', 'eval', 'complete'")


# ------------------------- Rhymes -------------------------
_RHYME_INDEX_BUILT = False
_RHYME_KEY_TO_WORDS: dict = {}
_ALL_WORDS: List[str] = []
_TAIL3_TO_WORDS: dict = {}
_TAIL2_TO_WORDS: dict = {}

_BG_VOWELS = set("аеиоуъюяѝ")


def _strip_diacritics(s: str) -> str:
	n = unicodedata.normalize("NFD", s)
	n = "".join(ch for ch in n if unicodedata.category(ch) != "Mn")
	return unicodedata.normalize("NFC", n)


def _only_letters(s: str) -> str:
	return "".join(ch for ch in s if ch.isalpha())


def _canon_word(s: str) -> str:
	return _only_letters(_strip_diacritics(s)).lower()


def _stressed_tail_from_row(row: dict) -> Optional[str]:
	w_stress = (row.get("word_stressed") or "").strip()
	if w_stress and "'" in w_stress:
		t = w_stress.replace("-", "")
		t = t.split("'", 1)[1]
		t = _canon_word(t)
		return t or None
	w_acc = (row.get("word_stressed_accent") or "").strip()
	if w_acc:
		s = _strip_diacritics(w_acc)
		s = s.replace("-", "")
		# approximate: from first vowel in accented form to end
		s_low = s.lower()
		pos = None
		for i, ch in enumerate(s_low):
			if ch in _BG_VOWELS:
				pos = i
				break
		if pos is not None:
			return _only_letters(s_low[pos:]) or None
	w_syl = (row.get("word_syllables") or "").strip()
	if w_syl and "-" in w_syl:
		parts = [p for p in w_syl.split("-") if p]
		if parts:
			last = _canon_word(parts[-1])
			return last or None
	w = (row.get("word") or "").strip()
	w_c = _canon_word(w)
	if not w_c:
		return None
	pos = None
	for i in range(len(w_c) - 1, -1, -1):
		if w_c[i] in _BG_VOWELS:
			pos = i
			break
	if pos is None:
		return None
	return w_c[pos:]


def _build_rhyme_index(dict_csv_path: Optional[str] = None) -> None:
	global _RHYME_INDEX_BUILT, _RHYME_KEY_TO_WORDS, _ALL_WORDS, _TAIL3_TO_WORDS, _TAIL2_TO_WORDS
	if _RHYME_INDEX_BUILT:
		return
	if dict_csv_path is None:
		dict_csv_path = str(Path(__file__).with_name("dictionary.csv"))
	mp_key = {}
	all_words = []
	tail3 = {}
	tail2 = {}
	LOG.info(f"[rhyme] building index from {dict_csv_path}")
	with open(dict_csv_path, newline="", encoding="utf-8-sig") as f:
		r = csv.DictReader(f)
		for row in r:
			w = (row.get("word") or "").strip()
			if not w:
				continue
			w_norm = _canon_word(w)
			if not w_norm:
				continue
			key = _stressed_tail_from_row(row)
			if not key:
				continue
			mp_key.setdefault(key, [])
			if w_norm not in (x[0] for x in mp_key[key]):
				mp_key[key].append((w_norm, w))
			all_words.append(w)
			t3 = w_norm[-3:] if len(w_norm) >= 3 else w_norm
			t2 = w_norm[-2:] if len(w_norm) >= 2 else w_norm
			tail3.setdefault(t3, []).append(w)
			tail2.setdefault(t2, []).append(w)
	# finalize
	_RHYME_KEY_TO_WORDS = {k: [disp for _, disp in v] for k, v in mp_key.items()}
	_ALL_WORDS = all_words
	_TAIL3_TO_WORDS = tail3
	_TAIL2_TO_WORDS = tail2
	LOG.info(f"[rhyme] keys={len(_RHYME_KEY_TO_WORDS)} words={len(_ALL_WORDS)} tail3={len(_TAIL3_TO_WORDS)} tail2={len(_TAIL2_TO_WORDS)}")
	_RHYME_INDEX_BUILT = True


def _rhyme_key_for_word(w: str, row_lookup: Optional[dict] = None) -> Optional[str]:
	w_c = _canon_word(w)
	if not w_c:
		return None
	# best-effort: derive key by locating last vowel if we lack row
	pos = None
	for i in range(len(w_c) - 1, -1, -1):
		if w_c[i] in _BG_VOWELS:
			pos = i
			break
	if pos is None:
		return None
	return w_c[pos:]


def get_rhymes(word: str, topn: int = RHYME_CANDIDATES) -> List[str]:
	LOG.info(f"[rhyme] query word='{word}' topn={topn}")
	_build_rhyme_index(None)
	key = _rhyme_key_for_word(word)
	res: List[str] = []
	seen = set()
	w_norm = _canon_word(word)
	if key and key in _RHYME_KEY_TO_WORDS:
		for w in _RHYME_KEY_TO_WORDS[key]:
			if _canon_word(w) == w_norm:
				continue
			if w not in seen:
				res.append(w)
				seen.add(w)
			if len(res) >= topn:
				LOG.info(f"[rhyme] found exact key matches={len(res)}")
				return res
	# fallback: tail3
	t3 = w_norm[-3:] if len(w_norm) >= 3 else w_norm
	if t3 in _TAIL3_TO_WORDS:
		for w in _TAIL3_TO_WORDS[t3]:
			if _canon_word(w) == w_norm:
				continue
			if w not in seen:
				res.append(w)
				seen.add(w)
			if len(res) >= topn:
				LOG.info(f"[rhyme] used tail3 fallback; total={len(res)}")
				return res
	# fallback: tail2
	t2 = w_norm[-2:] if len(w_norm) >= 2 else w_norm
	if t2 in _TAIL2_TO_WORDS:
		for w in _TAIL2_TO_WORDS[t2]:
			if _canon_word(w) == w_norm:
				continue
			if w not in seen:
				res.append(w)
				seen.add(w)
			if len(res) >= topn:
				LOG.info(f"[rhyme] used tail2 fallback; total={len(res)}")
				return res
	LOG.info(f"[rhyme] total_returned={len(res[:topn])}")
	return res[:topn]


if __name__ == "__main__":
	main()

