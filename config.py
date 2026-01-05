# -*- coding: utf-8 -*-
"""
Configuration file for BPE + Transformer poem generation
All hyperparameters are defined here for easy tuning
"""

import torch

# ====================== DATA PATHS ======================
POEMS_GLOB = "chitanka_poems_step1/*.txt"
LEXICON_CSV = "hf_dataset/words_combined.csv"
DICT_CSV = "dictionary.csv"

# ====================== TOKENIZER ======================
SPM_PREFIX = "poems_bpe"
SPM_MODEL = f"{SPM_PREFIX}.model"
SPM_TRAIN_TXT = "spm_train.txt"
VOCAB_SIZE = 5000
CHAR_COVERAGE = 0.9995

# ====================== CHECKPOINTS/LOGS ======================
LOG_FILE = "bpe_rhyme_rl.log"
CKPT_FWD = "lm_fwd.pt"
CKPT_BWD = "lm_bwd.pt"

# ====================== GENERAL ======================
SEED = 7
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ====================== MODEL ARCHITECTURE ======================
CTX = 128
D_MODEL = 128
N_HEAD = 4
N_LAYERS = 6
D_FF = 4 * D_MODEL
DROPOUT = 0.3

# ====================== TRAINING ======================
BATCH = 64
TRAIN_STEPS = 100000
LR = 2e-3
CLIP_NORM = 1.0
SAVE_EVERY = 1000
SHOW_EVERY = 50

# ====================== GENERATION ======================
PROMPT_TEXT = ""
MAX_TOKENS_PER_LINE = 48
TEMPERATURE = 0.7
TOP_P = 0.9

# Candidate generation counts
CAND_A = 40
CAND_B = 60

# ====================== RHYME (STRESS-AWARE) ======================
RHYME_TRIES_PER_LINE = 5
# attempt these tail suffix lengths (from stressed vowel -> end), in order
RHYME_TAIL_KS = [999, 5, 4, 3, 2]    # 999 = full tail
# fallback if stress missing
RHYME_SUFFIX_LEN = 3
ALLOW_SHORTER_SUFFIX = True

# ====================== METER SCORING ======================
USE_METER = True
METER = "iamb"
N_FEET = 10
OOV_PENALTY = 0.2
INCOMPLETE_PENALTY = 0.2
MAX_WORD_SYLL = 6

# ====================== OUTPUT ======================
OUT_FILE = "generated_poems.txt"
GENERATIONS = 100

# ====================== MODEL-SPECIFIC CONFIG ======================
# Hyperparameters specific to this INFILL model architecture
SEP_SYMBOL = "<sep>"
LOG_FILE = "bpe_infill_rhyme.log"

CKPT_FWD_LM = "lm_fwd.pt"
CKPT_BWD_LM = "lm_bwd.pt"
CKPT_FWD_INFILL = "infill_fwd.pt"
CKPT_BWD_INFILL = "infill_bwd.pt"

MEM_CTX = 192
TRAIN_STEPS_LM = 100000
TRAIN_STEPS_INFILL = 100000
ENC_LAYERS = 2

CAND_A1 = 40
CAND_A2 = 80
CAND_B1 = 80
CAND_B2 = 80

RHYME_TRIES = 6
RHYME_SUFFIX_FALLBACK = 3

TRAIN_FWD_LM = True
TRAIN_BWD_LM = True
TRAIN_FWD_INFILL = True
TRAIN_BWD_INFILL = True