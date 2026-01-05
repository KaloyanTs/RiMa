import argparse
from pathlib import Path
import re

#!/usr/bin/env python3

def remove_digit_only_lines(input_path: Path, output_path: Path) -> None:
    with input_path.open('r', encoding='utf-8', newline='') as fin, \
         output_path.open('w', encoding='utf-8', newline='') as fout:
        for line in fin:
            s = line.strip()
            if s.isdigit() or s.lower().endswith("глава") or (s.endswith(".") and s[:-1].isdigit()):
                continue
            if s.startswith("[*"):
                continue
            if s.startswith(". ."):
                continue
            if re.search(r"\d+\.", s):
                continue
            if re.search(r"[A-Za-z]", s):
                continue
            if "*" in line and not s.startswith("[*"):
                fout.write(line.replace("*", ""))
                continue
            fout.write(line)

def default_output_path(input_path: Path) -> Path:
    suffix = "_no_digits"
    if input_path.suffix:
        return input_path.with_name(f"{input_path.stem}{suffix}{input_path.suffix}")
    return input_path.with_name(f"{input_path.name}{suffix}")

def main():
    parser = argparse.ArgumentParser(
        description="Erase lines containing only digits from a text file and save with name ending in '_no_digits'."
    )
    parser.add_argument("input_file", help="Path to the input text file")
    parser.add_argument("-o", "--output", help="Optional output file path")
    args = parser.parse_args()

    in_path = Path(args.input_file)
    out_path = Path(args.output) if args.output else default_output_path(in_path)

    remove_digit_only_lines(in_path, out_path)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()