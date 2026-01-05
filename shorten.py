#!/usr/bin/env python3
# keep_short_lines.py
import argparse
from pathlib import Path
import re

def keep_short_lines(input_path: Path, output_path: Path, max_len: int, strip_newline: bool, remove_empty: bool = False, remove_numbered: bool = False) -> None:
        with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8", newline="\n") as fout:
            for line in fin:
                s = line.rstrip("\n") if strip_newline else line
                
                # Skip if line is too long
                if len(s) > max_len:
                    continue
                
                # Skip empty lines if requested
                if remove_empty and not line.strip():
                    continue
                
                # Skip numbered lines if requested (e.g., "  (123)  ")
                if remove_numbered and re.match(r'^\s*\(\d+\)\s*$', line.rstrip("\n")):
                    continue
                
                
                # Skip lines containing more than three consecutive digits
                if re.search(r'\d{4,}', line):
                    continue
                
                # Skip lines ending with '*'
                if line.rstrip("\n").rstrip().endswith("*"):
                    continue
                
                # Skip blocks of 1–2 non-empty lines surrounded by empty lines
                if not line.strip():
                    def emit_if_passes(l: str) -> None:
                        t = l.rstrip("\n") if strip_newline else l
                        if len(t) > max_len:
                            return
                        if remove_empty and not l.strip():
                            return
                        if remove_numbered and re.match(r'^\s*\(\d+\)\s*$', l.rstrip("\n")):
                            return
                        if re.search(r'\d{4,}', l):
                            return
                        fout.write(l)

                    try:
                        l1 = next(fin)
                    except StopIteration:
                        pass
                    else:
                        if l1.strip():
                            # have at least one non-empty after the empty
                            try:
                                l2 = next(fin)
                            except StopIteration:
                                # not a complete pattern; emit current empty then l1
                                emit_if_passes(line)
                                emit_if_passes(l1)
                                continue
                            if not l2.strip():
                                # empty line ends after one non-empty -> skip whole block
                                continue
                            # l2 is non-empty; check third
                            try:
                                l3 = next(fin)
                            except StopIteration:
                                # no trailing empty -> not our pattern; emit in order
                                emit_if_passes(line)
                                emit_if_passes(l1)
                                emit_if_passes(l2)
                                continue
                            if not l3.strip():
                                # two non-empty lines followed by empty -> skip whole block
                                continue
                            # third is non-empty -> not our pattern; emit in order
                            emit_if_passes(line)
                            emit_if_passes(l1)
                            emit_if_passes(l2)
                            emit_if_passes(l3)
                            continue
                        else:
                            # consecutive empty lines; not our pattern, emit both in order
                            emit_if_passes(line)
                            emit_if_passes(l1)
                            continue
                
                fout.write(line)

def main() -> None:
    p = argparse.ArgumentParser(description="Keep only lines with length <= max_len.")
    p.add_argument("input", help="Input .txt file (UTF-8).")
    p.add_argument("-o", "--output", help="Output file. Default: <input>.short.txt")
    p.add_argument("-n", "--max-len", type=int, default=50, help="Maximum allowed line length (default: 50).")
    p.add_argument("--count-newline", action="store_true",
                   help="Count the trailing newline as part of the length (usually you don't want this).")
    p.add_argument("--remove-empty", action="store_true",
                   help="Remove empty lines.")
    p.add_argument("--remove-numbered", action="store_true",
                   help="Remove lines matching pattern: whitespace '(' number ')' whitespace.")
    args = p.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output) if args.output else in_path.with_suffix(".short.txt")

    keep_short_lines(in_path, out_path, args.max_len, strip_newline=not args.count_newline)
    print(f"Wrote: {out_path}")

if __name__ == "__main__":
    main()
