"""Put percentages in Turkish order in the translated report sections.

English writes the sign after the number (74%, 27 %, −25 %); Turkish writes it before
(%74, %27, −%25), and `docs/15_ABLATION_LEDGER.md` -- which has been kept in Turkish since the
project started -- already does. The translators were told to preserve numbers character for
character, which they did, so the reordering is done here once, mechanically, over every file
at once rather than asked of seven separate agents.

Only the order changes. The digits, the sign and any decimal point are untouched, so the
numeric check in `scripts/check_translation.py` still passes: it reads numbers without their
percent sign.

Code spans are protected, because a percentage inside backticks is part of a path, a flag or a
literal and must stay exactly as the code has it.

Usage: python scripts/tr_percent_style.py [--lang tr] [--dry-run]
"""
import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CODE = re.compile(r"`[^`\n]+`")
# A sign (plus, or U+2212 minus -- this document never uses the ASCII hyphen as a minus),
# then digits with at most one DOT decimal part, at most one space, then the percent sign.
#
# Two deliberate restrictions, each from a bug this script actually caused:
#   no comma  -- a digit class containing ',' let the number run past a comma and swallow
#                the next one, turning '+0.0000, 95 % CI' into '+%0.0000,95 CI'. No
#                percentage in this report contains a comma, so excluding it removes the
#                entire class of failure rather than the one instance.
#   one space -- an unbounded space class would reach across a line break and reorder a
#                number against a percent sign belonging to the following line.
PCT = re.compile(r"([+−]?)(\d+(?:\.\d+)?)[  ]?%")


def convert(text):
    """Reorder outside code spans only; returns (new_text, count)."""
    parts, last, n = [], 0, 0
    for m in CODE.finditer(text):
        chunk, k = PCT.subn(lambda x: f"{x.group(1)}%{x.group(2)}", text[last:m.start()])
        parts.append(chunk)
        parts.append(m.group(0))          # the code span, verbatim
        last, n = m.end(), n + k
    chunk, k = PCT.subn(lambda x: f"{x.group(1)}%{x.group(2)}", text[last:])
    parts.append(chunk)
    return "".join(parts), n + k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="tr")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    d = ROOT / "docs" / "report" / a.lang
    if not d.is_dir():
        sys.exit(f"no such language directory: {d}")
    total = 0
    for p in sorted(d.glob("0*.md")):
        s = p.read_text(encoding="utf-8")
        new, n = convert(s)
        total += n
        if n and not a.dry_run:
            p.write_text(new, encoding="utf-8")
        print(f"  {p.name}: {n} percentage(s){' (dry run)' if a.dry_run else ''}")
    print(f"{total} reordered")


if __name__ == "__main__":
    main()
