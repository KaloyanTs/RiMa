# -*- coding: utf-8 -*-
"""
GUI annotator for dictionary.csv using spm_train.txt

NEW: auto-skip "obvious" words with exactly 1 vowel:
- word_syllables = word (no dashes)
- word_stressed = apostrophe before that vowel
- written to dictionary.csv automatically (no GUI shown for them)

Also:
- defines _count_vowels
- Prev/Next buttons
- centered window
"""

from __future__ import annotations

import csv
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import tkinter as tk
from tkinter import ttk, messagebox


# ----------------- CONFIG -----------------
SPM_TRAIN_TXT = "spm_train.txt"
DICT_CSV = "dictionary.csv"

WORD_RE = re.compile(r"[а-яА-ЯёЁѝЍ]+")
VOWELS = set("а ъ о у е и ю я ѝ".split())


# ----------------- helpers -----------------
def _count_vowels(s: str) -> int:
    return sum(1 for ch in s.lower() if ch in VOWELS)


def vowel_positions(word: str) -> List[int]:
    return [i for i, ch in enumerate(word) if ch.lower() in VOWELS]


def has_vowel(w: str) -> bool:
    return _count_vowels(w) > 0


def extract_words_with_freq(txt_path: str) -> Counter:
    text = Path(txt_path).read_text(encoding="utf-8", errors="ignore").lower()
    words = WORD_RE.findall(text)
    return Counter(w.lower() for w in words if has_vowel(w))


def read_dictionary(csv_path: str) -> Tuple[List[str], List[Dict[str, str]], Dict[str, int]]:
    p = Path(csv_path)
    if not p.exists():
        fieldnames = [
            "word",
            "word_stressed",
            "word_stressed_accent",
            "word_syllables",
            "transcription",
            "vowel_stress",
            "original_transcription",
        ]
        return fieldnames, [], {}

    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        fieldnames = list(r.fieldnames or [])
        rows = [dict(row) for row in r]

    if "word" not in fieldnames:
        raise RuntimeError(f"dictionary.csv missing 'word' column. Found: {fieldnames}")

    idx: Dict[str, int] = {}
    for i, row in enumerate(rows):
        w = (row.get("word") or "").strip().lower()
        if w and w not in idx:
            idx[w] = i
    return fieldnames, rows, idx


def write_dictionary(csv_path: str, fieldnames: List[str], rows: List[Dict[str, str]]) -> None:
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            out = {k: (row.get(k) or "") for k in fieldnames}
            w.writerow(out)


def ensure_backup_once(csv_path: str) -> None:
    p = Path(csv_path)
    if not p.exists():
        return
    bak = p.with_suffix(p.suffix + ".bak")
    if not bak.exists():
        shutil.copy2(p, bak)


def upsert_entry_fill_missing(
    csv_path: str,
    fieldnames: List[str],
    rows: List[Dict[str, str]],
    idx: Dict[str, int],
    word: str,
    word_stressed: str,
    word_syllables: str,
    write_now: bool = True,
) -> None:
    key = word.strip().lower()
    if not key:
        return

    if key in idx:
        r = rows[idx[key]]
    else:
        r = {fn: "" for fn in fieldnames}
        rows.append(r)
        idx[key] = len(rows) - 1

    r["word"] = word

    if "word_stressed" in fieldnames and not (r.get("word_stressed") or "").strip():
        r["word_stressed"] = word_stressed
    if "word_syllables" in fieldnames and not (r.get("word_syllables") or "").strip():
        r["word_syllables"] = word_syllables

    if write_now:
        write_dictionary(csv_path, fieldnames, rows)


def build_syllables_from_boundaries(word: str, boundaries: List[bool]) -> str:
    letters = list(word)
    out = []
    for i, ch in enumerate(letters):
        out.append(ch)
        if i < len(letters) - 1 and boundaries[i]:
            out.append("-")
    return "".join(out)


def build_word_stressed(word: str, stress_letter_idx: int) -> str:
    letters = list(word)
    if not (0 <= stress_letter_idx < len(letters)):
        return word
    letters.insert(stress_letter_idx, "'")  # apostrophe BEFORE stressed vowel
    return "".join(letters)


def center_window(root: tk.Tk, w: int, h: int) -> None:
    root.update_idletasks()
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    x = (sw - w) // 2
    y = (sh - h) // 2
    root.geometry(f"{w}x{h}+{x}+{y}")


# ----------------- GUI -----------------
@dataclass
class Item:
    word: str
    freq: int


class AnnotatorGUI:
    def __init__(
        self,
        items: List[Item],
        csv_path: str,
        fieldnames: List[str],
        rows: List[Dict[str, str]],
        idx: Dict[str, int],
        annotated_start: int = 0,
    ):
        self.items = items
        self.csv_path = csv_path
        self.fieldnames = fieldnames
        self.rows = rows
        self.idx = idx

        self.i = 0
        self.annotated = annotated_start

        self.root = tk.Tk()
        self.root.title("dictionary.csv annotator")
        center_window(self.root, 900, 300)

        self.counter_var = tk.StringVar()
        self.word_var = tk.StringVar()
        self.preview_syl_var = tk.StringVar()
        self.preview_stress_var = tk.StringVar()
        self.msg_var = tk.StringVar()

        top = ttk.Frame(self.root, padding=10)
        top.pack(fill="x")
        ttk.Label(top, textvariable=self.counter_var, font=("Arial", 12, "bold")).pack(anchor="center")
        ttk.Label(top, textvariable=self.word_var, font=("Arial", 22, "bold")).pack(anchor="center", pady=(6, 10))

        mid = ttk.Frame(self.root, padding=(10, 0, 10, 0))
        mid.pack(fill="x")
        self.boundary_frame = ttk.Frame(mid)
        self.boundary_frame.pack(fill="x", pady=(0, 10))
        self.vowel_frame = ttk.Frame(mid)
        self.vowel_frame.pack(fill="x", pady=(0, 10))

        prev = ttk.Frame(self.root, padding=(10, 0, 10, 0))
        prev.pack(fill="x")
        prev.columnconfigure(0, weight=1)
        prev.columnconfigure(1, weight=3)

        ttk.Label(prev, text="word_syllables:", anchor="e").grid(row=0, column=0, sticky="e")
        ttk.Label(prev, textvariable=self.preview_syl_var, font=("Consolas", 12)).grid(row=0, column=1, sticky="w", padx=8)
        ttk.Label(prev, text="word_stressed:", anchor="e").grid(row=1, column=0, sticky="e")
        ttk.Label(prev, textvariable=self.preview_stress_var, font=("Consolas", 12)).grid(row=1, column=1, sticky="w", padx=8)
        # --- Left-side statistics: frequency buckets of remaining words ---
        # Shift existing preview columns to the right to free column 0 for stats
        for child in prev.grid_slaves():
            info = child.grid_info()
            r = int(info.get("row", 0))
            c = int(info.get("column", 0))
            if r in (0, 1) and c in (0, 1):
                child.grid_configure(column=c + 1)

        # Stats UI on the left (column 0)
        self.stats_var = tk.StringVar()
        ttk.Label(prev, text="Buckets (freq: count):", anchor="w").grid(row=0, column=0, sticky="nw", padx=(0, 8))
        ttk.Label(prev, textvariable=self.stats_var, font=("Consolas", 11), justify="left").grid(row=1, column=0, sticky="nw")

        # Periodically recompute buckets for remaining items (sorted by freq desc)
        def _update_stats():
            try:
                remaining = self.items[self.i :] if 0 <= self.i < len(self.items) else []
                freq_counts = Counter(it.freq for it in remaining)
                if freq_counts:
                    lines = [f"{freq}: {count}" for freq, count in sorted(freq_counts.items(), key=lambda t: -t[0])]
                    self.stats_var.set("\n".join(lines))
                else:
                    self.stats_var.set("(none)")
            except Exception:
                pass  # keep GUI resilient

            if self.root.winfo_exists():
                self.root.after(300, _update_stats)

        _update_stats()
        bottom = ttk.Frame(self.root, padding=10)
        bottom.pack(fill="x")
        ttk.Label(bottom, textvariable=self.msg_var, foreground="#a00").pack(anchor="center")

        btns = ttk.Frame(bottom)
        btns.pack(anchor="center", pady=(8, 0))
        ttk.Button(btns, text="Prev", command=self.prev).grid(row=0, column=0, padx=6)
        ttk.Button(btns, text="Skip → Next", command=self.skip).grid(row=0, column=1, padx=6)
        ttk.Button(btns, text="Save + Next (Space)", command=self.save_next).grid(row=0, column=2, padx=6)
        ttk.Button(btns, text="Next", command=self.next_only).grid(row=0, column=3, padx=6)

        self.root.bind("<space>", lambda e: self.save_next())
        self.root.bind("<Left>", lambda e: self.prev())
        self.root.bind("<Right>", lambda e: self.next_only())
        self.root.bind("<Escape>", lambda e: self.root.destroy())

        self.boundary_vars: List[tk.IntVar] = []
        self.selected_stress_idx: Optional[int] = None

        self.load_current()

    def load_current(self) -> None:
        if self.i < 0:
            self.i = 0

        if self.i >= len(self.items):
            self.counter_var.set(f"Done! annotated={self.annotated}")
            self.word_var.set("")
            self.preview_syl_var.set("")
            self.preview_stress_var.set("")
            self.msg_var.set("All words annotated.")
            messagebox.showinfo("Done", "All missing words are annotated.")
            self.root.destroy()
            return

        it = self.items[self.i]
        w = it.word

        remaining_occurrences = sum(item.freq for item in self.items[self.i:])
        self.counter_var.set(
            f"Annotated: {self.annotated}    Remaining (occurrences): {remaining_occurrences}    (freq={it.freq})"
        )
        self.word_var.set(w)

        for child in self.boundary_frame.winfo_children():
            # clear previous UI
            child.destroy()
            # enforce lowercase for current word and update display
            it.word = it.word.lower()
            w = it.word
            self.word_var.set(w)
        for child in self.vowel_frame.winfo_children():
            child.destroy()

        self.boundary_vars = []
        self.selected_stress_idx = None

        letters = list(w)
        n = len(letters)

        row = ttk.Frame(self.boundary_frame)
        row.pack(anchor="center")

        col = 0
        for j, ch in enumerate(letters):
            ttk.Label(row, text=ch, font=("Arial", 16)).grid(row=0, column=col, padx=1)
            col += 1
            if j < n - 1:
                v = tk.IntVar(value=0)
                self.boundary_vars.append(v)
                cb = ttk.Checkbutton(row, variable=v, command=self.update_previews)
                cb.grid(row=0, column=col, padx=1)
                col += 1

        row2 = ttk.Frame(self.vowel_frame)
        row2.pack(anchor="center")
        ttk.Label(row2, text="Click stressed vowel:", font=("Arial", 11)).pack(side="left", padx=(0, 10))

        for j, ch in enumerate(letters):
            if ch.lower() in VOWELS:
                ttk.Button(row2, text=ch, width=3, command=lambda jj=j: self.set_stress(jj)).pack(side="left", padx=2)

        self.msg_var.set("")
        self.update_previews()

    def set_stress(self, idx: int) -> None:
        self.selected_stress_idx = idx
        self.update_previews()

    def update_previews(self) -> None:
        w = self.items[self.i].word
        boundaries = [bool(v.get()) for v in self.boundary_vars]
        self.preview_syl_var.set(build_syllables_from_boundaries(w, boundaries))

        if self.selected_stress_idx is None:
            self.preview_stress_var.set("(click a vowel)")
        else:
            self.preview_stress_var.set(build_word_stressed(w, self.selected_stress_idx))

    def validate(self) -> bool:
        w = self.items[self.i].word

        if self.selected_stress_idx is None:
            self.msg_var.set("Pick the stressed vowel.")
            return False

        ch = list(w)[self.selected_stress_idx]
        if ch.lower() not in VOWELS:
            self.msg_var.set("Stress must be on a vowel.")
            return False

        # still require boundaries for multi-vowel words (you can relax this if you want)
        if _count_vowels(w) >= 2 and "-" not in self.preview_syl_var.get():
            self.msg_var.set("Add syllable boundaries (checkboxes) for multi-vowel words.")
            return False

        self.msg_var.set("")
        return True

    def save_next(self) -> None:
        if not self.validate():
            return

        w = self.items[self.i].word
        word_syllables = self.preview_syl_var.get()
        word_stressed = build_word_stressed(w, self.selected_stress_idx or 0)

        try:
            upsert_entry_fill_missing(
                self.csv_path, self.fieldnames, self.rows, self.idx,
                w, word_stressed, word_syllables,
                write_now=True,  # after each manual annotation
            )
        except Exception as e:
            messagebox.showerror("Write error", f"Failed to update dictionary.csv:\n{e}")
            return

        self.annotated += 1
        self.i += 1
        self.load_current()

    def skip(self) -> None:
        self.i += 1
        self.load_current()

    def next_only(self) -> None:
        self.i += 1
        self.load_current()

    def prev(self) -> None:
        self.i -= 1
        self.load_current()

    def run(self) -> None:
        self.root.mainloop()


# ----------------- main -----------------
def main() -> None:
    if not Path(SPM_TRAIN_TXT).exists():
        raise RuntimeError(f"Missing {SPM_TRAIN_TXT}")

    ensure_backup_once(DICT_CSV)
    fieldnames, rows, idx = read_dictionary(DICT_CSV)

    if "word_stressed" not in fieldnames or "word_syllables" not in fieldnames:
        raise RuntimeError(
            f"dictionary.csv must have columns word_stressed and word_syllables. Found: {fieldnames}"
        )

    freq = extract_words_with_freq(SPM_TRAIN_TXT)

    # Build todo list, but AUTO-ANNOTATE words with exactly 1 vowel
    items: List[Item] = []
    auto_done = 0

    # candidate set: missing row OR missing either field
    candidates: List[Tuple[str, int]] = []
    for w, f in freq.items():
        key = w.lower()
        if key not in idx:
            candidates.append((w, f))
            continue
        r = rows[idx[key]]
        ws = (r.get("word_stressed") or "").strip()
        wsy = (r.get("word_syllables") or "").strip()
        if (not ws) or (not wsy):
            candidates.append((w, f))

    # process in descending freq for nicer UX
    candidates.sort(key=lambda x: (-x[1], x[0].lower()))

    for w, f in candidates:
        vpos = vowel_positions(w)
        if len(vpos) == 1:
            stress_idx = vpos[0]
            ws = build_word_stressed(w, stress_idx)
            wsy = w  # one syllable -> no dashes
            upsert_entry_fill_missing(
                DICT_CSV, fieldnames, rows, idx,
                w, ws, wsy,
                write_now=False,  # batch write for speed
            )
            auto_done += 1
        else:
            items.append(Item(word=w, freq=f))

    if auto_done > 0:
        write_dictionary(DICT_CSV, fieldnames, rows)

    if not items:
        print(f"Nothing to annotate in GUI. Auto-annotated {auto_done} one-vowel words.")
        return

    print(f"Auto-annotated (1 vowel): {auto_done}")
    print(f"To annotate in GUI: {len(items)}")

    AnnotatorGUI(items, DICT_CSV, fieldnames, rows, idx, annotated_start=auto_done).run()


if __name__ == "__main__":
    main()
