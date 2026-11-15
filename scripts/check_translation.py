"""Check a translated report section against its English original, mechanically.

A translation can be fluent and still be wrong in the two ways that matter for this document:
a number can go missing or change, and a table can lose a row. Neither is visible on a read.
This checks five invariants per file and prints a per-file verdict:

  numbers     every numeric token in the English file appears the same number of times in the
              translation, and no new ones appear
  code spans  every backticked span survives byte-identically (file paths, flags, column names)
  tables      same count of table rows, and the same column count row by row
  headings    same number of headings at each level, in the same order of levels
  furniture   same number of table captions, figures and display equations, with the same
              equation numbers

Numbers are compared as multisets, not positionally, so reordering a sentence is fine and
dropping a value is not. Usage: python scripts/check_translation.py [--lang tr]
"""
import argparse
import collections
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "report"

# This report is full of U+2212 and Turkish letters, and the default Windows console codec is
# cp1254, which cannot encode either. Without this the checker crashes while printing the very
# failures it exists to report.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# The ASCII hyphen is a word joiner in this document ("fold-0-based", "1-of-3"), never a minus:
# negatives are written with U+2212. Excluding it stops a hyphenated compound from tokenising
# differently either side of a translation, which is a false alarm, not a lost measurement.
NUM = re.compile(r"[+−]?\d[\d.,·−]*")
CODE = re.compile(r"`[^`\n]+`")
HEAD = re.compile(r"^(#{1,6}) ", re.M)


def numbers(text):
    # Strip code spans first: a path like folds_v1.parquet is not a measurement. Then drop the
    # percent sign, because Turkish writes it before the number (+%23) where English writes it
    # after (+23 %); that reordering is deliberate and must not read as a changed value.
    return collections.Counter(n.strip(".,") for n in
                               NUM.findall(CODE.sub(" ", text).replace("%", "")))


def table_rows(text):
    return [ln.count("|") for ln in text.splitlines() if ln.strip().startswith("|")]


def headings(text):
    return [len(m) for m in HEAD.findall(text)]


def furniture(text):
    """Counts of the typeset structure a translation must carry over: table captions, figures,
    display equations and their numbers."""
    return {"captions": len(re.findall(r"^: ", text, re.M)),
            "figures": len(re.findall(r"!\[", text)),
            "equations": len(re.findall(r"^::: equation", text, re.M)),
            "equation numbers": re.findall(r"\[\((\d+)\)\]\{\.eqno\}", text)}


def check(name, lang):
    en = SRC / name
    tr = SRC / lang / name
    if not tr.exists():
        return None
    a, b = en.read_text(encoding="utf-8"), tr.read_text(encoding="utf-8")
    na, nb = numbers(a), numbers(b)
    lost, extra = na - nb, nb - na
    ca, cb = collections.Counter(CODE.findall(a)), collections.Counter(CODE.findall(b))
    lost_code = ca - cb
    ta, tb = table_rows(a), table_rows(b)
    ha, hb = headings(a), headings(b)

    problems = []
    if lost:
        problems.append(f"numbers missing: {sorted(lost.items())[:12]}")
    if extra:
        problems.append(f"numbers invented: {sorted(extra.items())[:12]}")
    if lost_code:
        problems.append(f"code spans lost: {sorted(lost_code)[:8]}")
    if ta != tb:
        problems.append(f"tables: {len(ta)} rows -> {len(tb)}, shapes "
                        f"{'match' if ta == tb else 'DIFFER'}")
    if ha != hb:
        problems.append(f"headings: {len(ha)} -> {len(hb)}, levels "
                        f"{'match' if ha == hb else 'DIFFER'}")
    fa, fb = furniture(a), furniture(b)
    for key in fa:
        if fa[key] != fb[key]:
            problems.append(f"{key}: {fa[key]} -> {fb[key]}")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="tr")
    a = ap.parse_args()
    names = sorted(p.name for p in SRC.glob("0*.md"))
    bad = 0
    for n in names:
        r = check(n, a.lang)
        if r is None:
            print(f"  --   {n}  (not translated, falls back to English)")
            continue
        if r:
            bad += 1
            print(f"  FAIL {n}")
            for line in r:
                print(f"         {line}")
        else:
            print(f"  ok   {n}")
    print(f"\n{bad} file(s) with problems")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
