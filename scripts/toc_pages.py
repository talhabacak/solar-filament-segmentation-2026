"""Put page numbers into the report's table of contents, by reading them back from a printed PDF.

Chromium cannot resolve `target-counter()`, so a printed table of contents has entries but no
page numbers. This closes the gap in two passes. `prepare` gives every contents entry a dotted
leader and a placeholder page field of fixed width, so the first print already has the final
layout. `fill` reads that print page by page, finds each heading at or after the page of the one
before it (headings appear in document order, and searching forward is what stops a heading's
own entry on the contents page, or a mention of it in the prose, from being taken for it), and
writes the page numbers into the HTML for the second print. An entry whose heading cannot be
found is left blank and reported; it is never guessed.

Usage: python scripts/toc_pages.py prepare report.html
       python scripts/toc_pages.py fill report.html report.pdf
"""
import html
import re
import sys
import unicodedata

from pypdf import PdfReader

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

NAV = re.compile(r'(<nav id="TOC".*?</nav>)', re.S)
ENTRY = re.compile(r'<a href="#([^"]+)" id="toc-[^"]*">(.*?)</a>', re.S)
PLACEHOLDER = '<span class="toc-page">000</span>'


def norm(s):
    s = unicodedata.normalize("NFKC", html.unescape(re.sub(r"<[^>]+>", "", s)))
    s = s.replace("­", "")
    # whitespace is dropped entirely, not collapsed: the PDF text layer splits italic words into
    # letters ("n o t") and joins words with tabs, so only the letter sequence is comparable
    return re.sub(r"\s+", "", s).casefold()


def prepare(path):
    doc = open(path, encoding="utf-8").read()
    m = NAV.search(doc)
    assert m, "no pandoc table of contents in the HTML; was --toc passed?"

    def entry(e):
        if 'class="toc-text"' in e.group(0):
            return e.group(0)
        return (e.group(0)[:e.group(0).index(">") + 1]
                + f'<span class="toc-text">{e.group(2)}</span>'
                + '<span class="toc-leader"></span>' + PLACEHOLDER + "</a>")

    nav = ENTRY.sub(entry, m.group(1))
    open(path, "w", encoding="utf-8").write(doc[:m.start()] + nav + doc[m.end():])


def fill(path, pdf):
    doc = open(path, encoding="utf-8").read()
    m = NAV.search(doc)
    assert m, "no table of contents in the HTML"
    pages = []
    for p in PdfReader(pdf).pages:
        lines = [norm(ln) for ln in (p.extract_text() or "").splitlines()]
        pages.append((lines, "".join(lines)))

    entries = ENTRY.findall(m.group(1))
    # the contents pages themselves: everything up to the last page that lists the final entry
    last_title = norm(re.search(r'class="toc-text">(.*?)</span>', entries[-1][1], re.S).group(1))
    start = 0
    for i, (lines, flat) in enumerate(pages[:8]):
        if last_title in flat:
            start = i + 1
    found, cursor, missing = {}, start, []
    for anchor, inner in entries:
        title = norm(re.search(r'class="toc-text">(.*?)</span>', inner, re.S).group(1))
        hit = None
        for i in range(cursor, len(pages)):
            lines, flat = pages[i]
            # a heading is a line of its own unless it wraps; accept a whole-line match first,
            # then a line that starts the heading, then the flattened page
            if (title in lines or any(ln and title.startswith(ln) and len(ln) > 12 for ln in lines)
                    or title in flat):
                hit = i
                break
        if hit is None:
            missing.append(title)
            continue
        found[anchor] = hit + 1
        cursor = hit

    def entry(e):
        a = e.group(1)
        num = str(found.get(a, ""))
        return e.group(0).replace(PLACEHOLDER, f'<span class="toc-page">{num}</span>')

    nav = ENTRY.sub(entry, m.group(1))
    open(path, "w", encoding="utf-8").write(doc[:m.start()] + nav + doc[m.end():])
    print(f"contents: {len(found)}/{len(entries)} entries paged"
          + (f"; not found: {missing}" if missing else ""))
    return 0 if not missing else 1


if __name__ == "__main__":
    if sys.argv[1] == "prepare":
        prepare(sys.argv[2])
    else:
        sys.exit(fill(sys.argv[2], sys.argv[3]))
