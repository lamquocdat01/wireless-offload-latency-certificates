"""Compare two numbers.json files key by key (artifact check C7). Exit 0 iff every key and value matches.

Usage: python code/compare_numbers.py results/numbers.json path/to/reference_numbers.json
"""
from __future__ import annotations

import json
import sys


def main() -> None:
    a = json.load(open(sys.argv[1], encoding="utf-8"))
    b = json.load(open(sys.argv[2], encoding="utf-8"))
    only_a, only_b = sorted(set(a) - set(b)), sorted(set(b) - set(a))
    diff = sorted(k for k in set(a) & set(b) if a[k] != b[k])
    print(f"keys: {len(a)} vs {len(b)}; only in first: {len(only_a)}; only in second: {len(only_b)}; "
          f"different values: {len(diff)}")
    for k in (only_a + only_b + diff)[:20]:
        print("  ", k, a.get(k, {}).get("tex"), "|", b.get(k, {}).get("tex"))
    sys.exit(0 if not (only_a or only_b or diff) else 1)


if __name__ == "__main__":
    main()
