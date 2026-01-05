from __future__ import annotations

import re
from pathlib import Path

IN_DIR = Path("bagryana")
OUT_DIR = Path("bagryana_step1")

CUT_MARKERS = re.compile(
    r"(?m)^(?:\s*\$id\s*=|\s*Източник:|\s*----\s*$|\s*__Издание:__)\s*.*$"
)

def extract_work(raw: str) -> str:
    s = raw.replace("\r\n", "\n").replace("\r", "\n")

    # Cut metadata block (if present)
    m = CUT_MARKERS.search(s)
    if m:
        s = s[:m.start()]

    # Trim outer whitespace/newlines
    s = s.strip("\n\t ")

    # Remove author + title = first 2 non-empty lines
    lines = s.splitlines()
    out_lines = []
    nonempty_seen = 0
    for line in lines:
        if nonempty_seen < 2:
            if line.strip():
                nonempty_seen += 1
            continue
        out_lines.append(line)

    return "\n".join(out_lines).strip()

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # If you want recursive: replace with IN_DIR.rglob("*")
    files = [p for p in IN_DIR.iterdir() if p.is_file()]

    processed = 0
    skipped_empty = 0

    for p in files:
        try:
            raw = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # Fallback, in case some files are cp1251
            raw = p.read_text(encoding="cp1251")

        cleaned = extract_work(raw)

        if not cleaned.strip():
            skipped_empty += 1
            continue

        out_path = OUT_DIR / p.name
        out_path.write_text(cleaned + "\n", encoding="utf-8")
        processed += 1

    print(f"Processed: {processed}")
    print(f"Skipped (empty after cleaning): {skipped_empty}")
    print(f"Output dir: {OUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
