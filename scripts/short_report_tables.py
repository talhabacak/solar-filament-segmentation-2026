"""Mark the numeric tables of the short report so the stylesheet can right-align their figures.

pandoc writes every table the same way. A table whose cells after the first column are mostly
numbers reads better right-aligned and unwrapped; a table of settings must wrap. The stylesheet
cannot tell them apart, so this adds class="numeric" to the former.

Usage: python scripts/short_report_tables.py report/short/build/report_en.html
"""
import re
import sys
from pathlib import Path

NUMBER = re.compile(r"^[\s*]*[+\-−%]*[\d.,\s]+%?[\s*]*$")   # "%69.9" is the Turkish order


def is_numeric(table_html):
    """True when at least two thirds of the non-empty cells after the first column are numbers."""
    cells = []
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", table_html, flags=re.S):
        tds = re.findall(r"<td[^>]*>(.*?)</td>", row, flags=re.S)
        cells += [re.sub(r"<[^>]+>", "", c).strip() for c in tds[1:]]
    cells = [c for c in cells if c]
    return bool(cells) and sum(bool(NUMBER.match(c)) for c in cells) >= 2 * len(cells) / 3


def main():
    path = Path(sys.argv[1])
    html = path.read_text(encoding="utf-8")
    marked = 0

    def mark(m):
        nonlocal marked
        if is_numeric(m.group(0)):
            marked += 1
            return m.group(0).replace("<table", '<table class="numeric"', 1)
        if "CI [" in m.group(0):      # the table of measured effects: wide first column
            return m.group(0).replace("<table", '<table class="effects"', 1)
        return m.group(0)

    html = re.sub(r"<table.*?</table>", mark, html, flags=re.S)
    path.write_text(html, encoding="utf-8", newline="\n")
    print(f"{path.name}: {marked} numeric table(s) marked")


if __name__ == "__main__":
    main()
