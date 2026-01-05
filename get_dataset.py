#!/usr/bin/env python3
"""
Download ALL files from the Hugging Face dataset repo:
  vislupus/bulgarian-dictionary-raw-data
and save them into a local folder next to this script: ./hf_dataset/

Install:
  pip install -U huggingface_hub
Run:
  python download_all_hf_dataset_files.py
"""

from pathlib import Path
import csv
import unicodedata
from typing import Dict, Optional

REPO_ID = "vislupus/bulgarian-dictionary-raw-data"
OUT_DIR = Path(__file__).resolve().parent / "hf_dataset"
DICT_CSV = Path(__file__).resolve().parent / "dictionary.csv"

# Cyrillic vowels (upper + lower)
CYR_VOWELS = set("аеиоуюяъАЕИОУЮЯЪ")

def _stressed_vowel_from_accented(word_stressed_accent: str) -> Optional[str]:
  """Extract the stressed Cyrillic vowel from a word where stress is marked
  with a combining accent (e.g., \u0300 grave or \u0301 acute) as seen in
  the `word_stressed_accent` column.

  Returns the vowel as a single Cyrillic character if found, else None.
  """
  if not word_stressed_accent:
    return None

  # Normalize to NFD so accents become separate combining chars
  s = unicodedata.normalize("NFD", word_stressed_accent)
  for i, ch in enumerate(s):
    # Combining Acute (\u0301) or Combining Grave (\u0300)
    if ch in ("\u0301", "\u0300"):
      # Find the base letter preceding this accent
      # Walk backwards to the previous non-combining character
      j = i - 1
      while j >= 0 and unicodedata.combining(s[j]) != 0:
        j -= 1
      if j >= 0 and s[j] in CYR_VOWELS:
        # Return normalized NFC single character vowel
        return unicodedata.normalize("NFC", s[j])
  return None

def _stressed_vowel_from_apostrophe(word_stressed: str) -> Optional[str]:
  """Extract the stressed vowel from the `word_stressed` column where stress
  is indicated with an apostrophe ('). The stressed syllable usually starts
  at/after the apostrophe; return the first Cyrillic vowel found after it.
  """
  if not word_stressed:
    return None
  try:
    pos = word_stressed.index("'")
  except ValueError:
    return None

  for ch in word_stressed[pos+1:]:
    if ch in CYR_VOWELS:
      return ch
  return None

def load_stress_map() -> Dict[str, str]:
  """Load `dictionary.csv` and build a map: lowercase `word` -> stressed vowel.

  Priority of sources per row:
    1) `vowel_stress` (if non-empty)
    2) derived from `word_stressed_accent`
    3) derived from `word_stressed`
  Returns only entries where a vowel could be resolved.
  """
  stress_map: Dict[str, str] = {}
  if not DICT_CSV.exists():
    return stress_map

  with DICT_CSV.open("r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
      word = (row.get("word") or "").strip()
      if not word:
        continue

      # 1) If `vowel_stress` is present, prefer it (convert to Cyrillic if needed)
      vs = (row.get("vowel_stress") or "").strip()
      stressed_vowel: Optional[str] = None
      if vs:
        # Map Latin vowels to Cyrillic where obvious
        latin_to_cyr = {
          "a": "а", "e": "е", "i": "и", "o": "о", "u": "у",
          # Bulgarian vowels beyond Latin mapping
          # If column already contains Cyrillic, keep as-is
          "A": "а", "E": "е", "I": "и", "O": "о", "U": "у",
        }
        stressed_vowel = latin_to_cyr.get(vs, vs) if vs else None

      # 2) Try `word_stressed_accent`
      if not stressed_vowel:
        stressed_vowel = _stressed_vowel_from_accented(row.get("word_stressed_accent") or "")

      # 3) Try `word_stressed` with apostrophe
      if not stressed_vowel:
        stressed_vowel = _stressed_vowel_from_apostrophe(row.get("word_stressed") or "")

      if stressed_vowel and stressed_vowel in CYR_VOWELS:
        stress_map[word.lower()] = stressed_vowel

  return stress_map

def get_stressed_vowel(word: str) -> Optional[str]:
  """Return the stressed Cyrillic vowel for `word` using `dictionary.csv`.
  Case-insensitive lookup. Returns None if not found.
  """
  if not word:
    return None
  mapping = load_stress_map()
  return mapping.get(word.lower())

def main() -> None:
  """Read dictionary.csv and print stressed vowels for sample words."""
  if not DICT_CSV.exists():
    print(f"Error: {DICT_CSV} not found!")
    return

  # Load the stress map
  print("Loading stress map from dictionary.csv...")
  stress_map = load_stress_map()
  print(f"Loaded {len(stress_map)} words with stress information.\n")

  # Test with some sample words
  test_words = ["Андора", "А", "Австрия", "Албания", "Аляска", "Америка", "Анкара"]
  
  print("Sample stressed vowels:")
  print("-" * 40)
  for word in test_words:
    stressed = get_stressed_vowel(word)
    if stressed:
      print(f"{word:15} -> {stressed}")
    else:
      print(f"{word:15} -> (no stress info)")
  
  print("\nYou can use get_stressed_vowel(word) to look up any word.")

if __name__ == "__main__":
    main()
