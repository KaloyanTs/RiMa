# RiMa
## Generator of Rhymed and Metered poetry in Bulgarian
**Project Overview**
- **Goal:** Train LoRA adapters on BgGPT-Gemma-2-2.6B-IT to generate rhymed Bulgarian poetry  either two-step couplets or four-step ABAB quatrains.
- **Pipelines:**
  - Two-step couplets: [finetune_rhymes.py](finetune_rhymes.py)
  - Stanza-based ABAB (4 calls): [finetune_stanza.py](finetune_stanza.py)
- **Model:** Uses Hugging Face model `INSAIT-Institute/BgGPT-Gemma-2-2.6B-IT-v1.0` with chat templates and assistant-only loss.

**Data & Assets**
- **Rhyme Dictionary:**
  - Primary: [dictionary.csv](dictionary.csv)  CSV with stressed forms used to build rhyme buckets.
  - Legacy: [dictionary_old.csv](dictionary_old.csv)  older format kept for reference.
- **Poem Corpus (not included in repo):**
  - Preprocessed poems: [chitanka_poems_step1/](chitanka_poems_step1/) directory with `.txt` files.
  - Additional curated files (not included):
    - [rhyming_pairs_from_quatrains.txt](rhyming_pairs_from_quatrains.txt) pairs of rhyming words for couplets.
    - [rhyming_quatrains_abab_aabb.txt](rhyming_quatrains_abab_aabb.txt) stanzas following abab or aabb patterns
- **Important note:** Text corpora and large CSV/TXT assets are not shipped. Place them in the workspace with the expected names/paths or adjust constants in the scripts.

**Dependencies**
- **Python:** 3.10+ recommended.
- **Packages:** `torch`, `transformers`, `datasets`, `peft`, `accelerate`, `sentencepiece`.
- **Setup:**
```bash
pip install -U torch transformers datasets peft accelerate sentencepiece
huggingface-cli login    # if the model requires access
```
- **GPU:** CUDA-capable GPU strongly recommended. Scripts auto-detect BF16 support (Ampere+); otherwise use FP16. If memory is tight, lower `PER_DEVICE_BATCH` and increase `GRAD_ACCUM` in the scripts.

**Two-Step Couplet Pipeline**
- **Script:** [finetune_rhymes.py](finetune_rhymes.py)
- **What it does:**
  - Builds a dataset of rhyming word pairs from [rhyming_pairs_from_quatrains.txt](rhyming_pairs_from_quatrains.txt), optionally filtering by [dictionary.csv](dictionary.csv).
  - Trains LoRA with assistant-only loss using the models chat template.
  - Generates couplets in two calls: line1 constrained to end on `w1`, line2 constrained to end on `w2` and remain semantically related.
- **Key constants to review:**
  - `MODE`  "finetune" to train, "generate" to only run demo.
  - `MODEL_ID`, `PAIRS_FILE`, `DICT_FILE`, `OUTPUT_DIR`.
  - Training: `MAX_SEQ_LEN`, `EPOCHS`, `LR`, `PER_DEVICE_BATCH`, `GRAD_ACCUM`, etc.
  - Generation: `GEN_MAX_NEW_TOKENS`, `GEN_TEMPERATURE`, `GEN_TOP_P`, etc.
  - Dictionary gating: `MIN_BUCKET`, `REQUIRE_ENDINGS_IN_DICT_FOR_TRAIN`.
- **Outputs:**
  - LoRA adapter checkpoints in [bggpt-two-step-couplet-lora-fp16/](bggpt-two-step-couplet-lora-fp16/) and final adapter in `final/`.
  - Logs: `train.log` inside the output folder.
- **Run (Windows):**
```bash
python finetune_rhymes.py            # uses constants; MODE controls behavior
```
- **Demo behavior:** Picks rhyming pairs and attempts generation with retries to ensure exact end-word matches.

**Stanza-Based ABAB Pipeline**
- **Script:** [finetune_stanza.py](finetune_stanza.py)
- **What it does:**
  - Builds a mixed dataset from stanzas and quatrains:
    - T1: single line ending with target word.
    - T2: consecutive lines inside stanzas.
    - T3: triples inside stanzas.
    - T4: quadruples (stanza fragments).
    - Adds extra pairs/triples/quads from [rhyming_quatrains_abab_aabb.txt](rhyming_quatrains_abab_aabb.txt).
  - Balances tasks via `TASK_WEIGHTS` and caps (`MAX_T1..MAX_T4`).
  - Trains LoRA and runs a 4-step ABAB demo:
    - 1) Given `<x1>`  generate line1 ending exactly with `x1`.
    - 2) Given `line1` + `<y1>`  generate line2 ending exactly with `y1`.
    - 3) Given `line1+line2` + `<x2>`  generate line3 ending exactly with `x2`.
    - 4) Given `line1+line2+line3` + `<y2>`  generate line4 ending exactly with `y2`.
- **Key constants to review:**
  - `MODE`  "finetune" or "generate".
  - `POEMS_DIR`, `DICT_FILE`, `QUATRAINS_FILE`, `OUTPUT_DIR`.
  - Caps and balancing: `MAX_T1..MAX_T4`, `MAX_FROM_QUATRAINS`, `TASK_WEIGHTS`, `MAX_FINAL_TRAIN_EXAMPLES`.
  - Training: `EPOCHS`, `LR`, `PER_DEVICE_BATCH`, `GRAD_ACCUM`, etc.
  - Generation: `GEN_*` parameters.
  - Dictionary gating: `REQUIRE_ENDINGS_IN_DICT`.
- **Outputs:**
  - LoRA adapter checkpoints in [bggpt-abab-4step-lora/](bggpt-abab-4step-lora/) and final adapter in `final/`.
  - Logs: `run.log` inside the output folder.
- **Run (Windows):**
```bash
python finetune_stanza.py            # uses constants; MODE controls behavior
```

**Data Placement**
- **Expected paths:**
  - Place your preprocessed poem `.txt` files under [chitanka_poems_step1/](chitanka_poems_step1/).
  - Place rhyme assets (must be genrated via [rhyme_extractor.py](rhyme_extractor.py)) alongside the scripts: [rhyming_pairs_from_quatrains.txt](rhyming_pairs_from_quatrains.txt), [rhyming_quatrains_abab_aabb.txt](rhyming_quatrains_abab_aabb.txt).
  - Ensure [dictionary.csv](dictionary.csv) is present; adjust paths via constants if needed.
- **Alternative:** If you use different folders or filenames, update the constants at the top of each script.

**Tuning Notes**
- **Precision:** Scripts detect BF16-capable GPUs; otherwise they use FP16. If you run into instability, try lowering `LR` or switching precision.
- **Memory:** Reduce `PER_DEVICE_BATCH` and increase `GRAD_ACCUM` to fit memory; optionally shorten `MAX_SEQ_LEN`.
- **Dictionary gating:** If generation fails due to strict end-word filters, relax `REQUIRE_ENDINGS_IN_DICT` (ABAB) or `REQUIRE_ENDINGS_IN_DICT_FOR_TRAIN` (couplets).
- **Checkpoints:** Latest checkpoints auto-detected for resume; final adapters saved under `final/` for loading in `MODE="generate"`.

**Quick Start**
- **Train couplets:**
  - Edit `MODE = "finetune"` in [finetune_rhymes.py](finetune_rhymes.py), verify paths.
  - Run:
```bash
python finetune_rhymes.py
```
- **Generate couplets (adapter loaded):**
  - Set `MODE = "generate"`.
```bash
python finetune_rhymes.py
```
- **Train ABAB quatrains:**
  - Edit `MODE = "finetune"` in [finetune_stanza.py](finetune_stanza.py), verify paths.
```bash
python finetune_stanza.py
```
- **Generate ABAB quatrains (adapter loaded):**
  - Set `MODE = "generate"`.
```bash
python finetune_stanza.py
```

**Logging & Outputs**
- **Couplets:** Logs in [bggpt-two-step-couplet-lora-fp16/train.log](bggpt-two-step-couplet-lora-fp16/train.log); checkpoints and `final/` adapter saved under [bggpt-two-step-couplet-lora-fp16/](bggpt-two-step-couplet-lora-fp16/).
- **ABAB:** Logs in [bggpt-abab-4step-lora/run.log](bggpt-abab-4step-lora/run.log); checkpoints and `final/` adapter saved under [bggpt-abab-4step-lora/](bggpt-abab-4step-lora/).

**Troubleshooting**
- **Hugging Face access errors:** Log in with `huggingface-cli login` and accept model terms.
- **Strategy rename warnings:** Scripts include compatibility shims for `TrainingArguments` changes.
- **Exact end-word not satisfied:** The generation has retries; if it still fails, increase `GEN_MAX_ATTEMPTS` or reduce `GEN_TEMPERATURE`.
- **OOM during training:** Lower batch size, increase gradient accumulation, or reduce sequence length.

**Notes**
- This repository purposefully excludes large corpora and datasets. You must provide them locally or customize the constants to point to your data.

**Requirements**
- **File:** [requirements.txt](requirements.txt)  pinned versions known to work.
- **Install:**
```bash
pip install -r requirements.txt
```

**Try It**
- **Create virtual environment (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
- **Install packages:**
```powershell
pip install -r requirements.txt
```
- **Couplets demo:**
  - Set `MODE = "generate"` in [finetune_rhymes.py](finetune_rhymes.py) if you already have an adapter in [bggpt-two-step-couplet-lora-fp16/final](bggpt-two-step-couplet-lora-fp16/final). Otherwise set `MODE = "finetune"` to train first.
```powershell
python finetune_rhymes.py
```
- **ABAB demo:**
  - Set `MODE = "generate"` in [finetune_stanza.py](finetune_stanza.py) if you already have an adapter in [bggpt-abab-4step-lora/final](bggpt-abab-4step-lora/final). Otherwise set `MODE = "finetune"`.
```powershell
python finetune_stanza.py
```

