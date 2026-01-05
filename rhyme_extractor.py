import os
import re
import csv
import logging
from dataclasses import dataclass

POEMS_DIR = "chitanka_poems_step1"
DICT_FILE = "dictionary.csv"
OUT_FILE = "rhyming_quatrains_abab_aabb.txt"
PAIRS_FILE = "rhyming_pairs_from_quatrains.txt"

WORD_RE = re.compile(r"[а-яА-ЯёЁ]+", re.UNICODE)
VOWELS = set("аеиоуъюяАЕИОУЪЮЯ")

LOG = logging.getLogger("rhyme_extractor")
LOG.setLevel(logging.INFO)

_console = logging.StreamHandler()
_console.setLevel(logging.INFO)
_console.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))

if not LOG.handlers:
    LOG.addHandler(_console)
LOG.propagate = False


@dataclass(frozen=True)
class KeyInfo:
    key: str | None          # rhyme key
    source: str              # "dict" or "fallback" or "none"


def load_dictionary(path: str) -> dict[str, str]:
    d: dict[str, str] = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            w = (row.get("word") or "").strip().lower()
            stressed = (row.get("word_stressed") or "").strip()
            if w and stressed:
                d[w] = stressed
    return d


def last_word(line: str) -> str | None:
    words = WORD_RE.findall(line)
    return words[-1].lower() if words else None


def rhyme_key_from_stressed(stressed_word: str) -> str | None:
    # Stress marked by ' before the stressed letter: "ч'аса"
    if "'" not in stressed_word:
        return None

    idx = stressed_word.find("'")
    clean = stressed_word.replace("'", "")
    stressed_pos = idx  # after removing ', index matches

    # find the stressed vowel index first
    stressed_vowel_idx = None
    for i in range(stressed_pos, len(clean)):
        if clean[i] in VOWELS:
            stressed_vowel_idx = i
            break
    if stressed_vowel_idx is None:
        return None

    # syllable onset: start right after the previous vowel (or at word start)
    start = 0
    for j in range(stressed_vowel_idx - 1, -1, -1):
        if clean[j] in VOWELS:
            start = j + 1
            break

    return clean[start:].lower()


def fallback_rhyme_key(word: str) -> str | None:
    # tighter fallback than "last vowel to end":
    # take last 3 letters if possible, else last 2; avoids tons of accidental matches
    w = word.lower()
    if len(w) >= 3:
        return w[-3:]
    if len(w) >= 2:
        return w[-2:]
    return None


def rhyme_key_info(word: str, dictionary: dict[str, str]) -> KeyInfo:
    if word in dictionary:
        k = rhyme_key_from_stressed(dictionary[word])
        if k:
            return KeyInfo(k, "dict")
    k2 = fallback_rhyme_key(word)
    if k2:
        return KeyInfo(k2, "fallback")
    return KeyInfo(None, "none")


def split_stanzas(lines: list[str]) -> list[list[tuple[int, str]]]:
    stanzas = []
    cur: list[tuple[int, str]] = []
    for idx, raw in enumerate(lines, start=1):
        s = raw.strip("\n")
        if s.strip() == "":
            if cur:
                stanzas.append(cur)
                cur = []
            continue
        cur.append((idx, s.rstrip()))
    if cur:
        stanzas.append(cur)
    return stanzas


def detect_scheme(keys: list[str]) -> str | None:
    a, b, c, d = keys
    if a == c and b == d and a != b:
        return "ABAB"
    if a == b and c == d and a != c:
        return "AABB"
    return None


def pair_indices_for_scheme(scheme: str) -> list[tuple[int, int]]:
    # indices in 0..3
    if scheme == "ABAB":
        return [(0, 2), (1, 3)]
    return [(0, 1), (2, 3)]  # AABB


def process_file(filepath: str, dictionary: dict[str, str], out_f, pairs_f, stats):
    LOG.info(f"[process] {filepath}")
    with open(filepath, encoding="utf-8") as f:
        lines = f.readlines()

    stanzas = split_stanzas(lines)
    base = os.path.basename(filepath)

    for si, stanza in enumerate(stanzas, start=1):
        if len(stanza) < 4:
            continue

        for i in range(len(stanza) - 3):
            window = stanza[i:i+4]  # [(lineno, text), ...]

            # last words (and dict miss logging)
            words: list[tuple[int, str]] = []
            ok = True
            for ln, txt in window:
                lw = last_word(txt)
                if lw is None:
                    stats["skipped_rows"].add((filepath, ln))
                    ok = False
                    break
                w = lw.lower()
                if w not in dictionary:
                    row_key = (filepath, ln)
                    if row_key not in stats["logged_missing"]:
                        LOG.warning(f"[dict-miss] {base}:{ln} word '{w}' not in dictionary")
                        stats["logged_missing"].add(row_key)
                    stats["missing_dict_rows"].add(row_key)
                words.append((ln, w))
            if not ok:
                continue

            # keys
            infos: list[KeyInfo] = []
            keys: list[str] = []
            for ln, w in words:
                info = rhyme_key_info(w, dictionary)
                if info.key is None:
                    stats["skipped_rows"].add((filepath, ln))
                    infos = []
                    keys = []
                    break
                infos.append(info)
                keys.append(info.key)
            if not keys:
                continue

            scheme = detect_scheme(keys)
            if not scheme:
                continue

            # Write quatrain as before (optional dedupe if you want)
            out_f.write(f"FILE: {base} | stanza: {si} | start_line: {window[0][0]} | scheme: {scheme}\n")
            out_f.write(f"  (rhyme keys) {keys[0]} {keys[1]} {keys[2]} {keys[3]}\n")
            for ln, txt in window:
                out_f.write(f"  [{ln:>4}] {txt}\n")
            out_f.write("\n")

            # Write PAIRS, but:
            # - only if both lines in the pair have DICT-based keys (reduces false positives)
            # - dedupe pairs globally
            for a, b in pair_indices_for_scheme(scheme):
                (ln1, t1) = window[a]
                (ln2, t2) = window[b]
                info1, info2 = infos[a], infos[b]

                # strict: require dictionary stress for both sides of the pair
                if not (info1.source == "dict" and info2.source == "dict"):
                    stats["pairs_skipped_weak"] += 1
                    continue

                # final sanity: keys must match for the pair (should already hold via scheme)
                if info1.key != info2.key:
                    stats["pairs_skipped_mismatch"] += 1
                    continue

                pair_id = (filepath, min(ln1, ln2), max(ln1, ln2))
                if pair_id in stats["seen_pairs"]:
                    stats["pairs_deduped"] += 1
                    continue
                stats["seen_pairs"].add(pair_id)

                pairs_f.write(
                    f"FILE: {base} | stanza: {si} | scheme: {scheme} | rhyme: {info1.key}\n"
                )
                pairs_f.write(f"  [{ln1:>4}] {t1}\n")
                pairs_f.write(f"  [{ln2:>4}] {t2}\n\n")
                stats["pairs_written"] += 1


def main():
    dictionary = load_dictionary(DICT_FILE)

    stats = {
        "skipped_rows": set(),              # set[(filepath, ln)]
        "missing_dict_rows": set(),         # set[(filepath, ln)]
        "logged_missing": set(),            # set[(filepath, ln)]
        "seen_pairs": set(),                # set[(filepath, ln_min, ln_max)]
        "pairs_written": 0,
        "pairs_deduped": 0,
        "pairs_skipped_weak": 0,
        "pairs_skipped_mismatch": 0,
    }

    with open(OUT_FILE, "w", encoding="utf-8") as out_f, open(PAIRS_FILE, "w", encoding="utf-8") as pairs_f:
        for root, _, files in os.walk(POEMS_DIR):
            for fn in files:
                if fn.lower().endswith((".txt", ".md")):
                    process_file(os.path.join(root, fn), dictionary, out_f, pairs_f, stats)

    LOG.info(f"Skipped rows: {len(stats['skipped_rows'])}")
    LOG.info(f"Rows with words not in dictionary: {len(stats['missing_dict_rows'])}")
    LOG.info(f"Pairs written: {stats['pairs_written']}")
    LOG.info(f"Pairs deduped (skipped): {stats['pairs_deduped']}")
    LOG.info(f"Pairs skipped (not dict-based on both lines): {stats['pairs_skipped_weak']}")
    LOG.info(f"Pairs skipped (key mismatch): {stats['pairs_skipped_mismatch']}")

    print(
        f"Done. Output written to {OUT_FILE} and {PAIRS_FILE}. "
        f"Pairs: {stats['pairs_written']}, deduped: {stats['pairs_deduped']}."
    )


if __name__ == "__main__":
    main()
