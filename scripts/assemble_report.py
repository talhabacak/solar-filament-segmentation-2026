"""Assemble the report sections into one judge-facing Markdown document, in one language.

The sections are drafted as separate files so they can be edited independently. Each carries a
drafting title ("Final Report - Draft Sections 1-2") and its own heading depth, which is fine
in isolation and wrong once they are concatenated: the body sections would sit at H2 while the
appendices sat at H1. This script normalises the document to one shape - H2 parts, H3
subsections - puts the title, author, date and abstract into a YAML block so pandoc typesets
them as a title page, marks the parts that open on a new page, and copies the figures next to
the output so image links resolve from the build directory. The table of contents is pandoc's,
built from the same headings, so there is exactly one.

Translations live in `docs/report/<lang>/` with the same filenames. A language directory need
not be complete: any file it does not carry falls back to the English one, so a half-finished
translation still builds. A file may instead supply `<name>.head.md`, whose contents replace
everything in the English file above its first H2 - that is how Appendix D gets a translated
heading while its table stays in English, since those rows are verbatim copies of the code's
own docstrings and translating them would break the only thing the index is for.

Usage: python scripts/assemble_report.py [outdir] [--lang tr]
"""
import argparse
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "report"

# reading order, not drafting order.
#   wrapper: a part heading the file lacks because its own H1 was a drafting title
#   demote:  levels to push every heading down, so parts land on H2 and subsections on H3
#   drop_h1: the file opens with a drafting title ("Final Report - Section 3: The Pipeline")
#            that names the file rather than its content. Declared per file rather than matched
#            against the title text, because a translated file carries a TRANSLATED drafting
#            title and no pattern written in English will ever catch it -- which is exactly how
#            two of them survived into the Turkish build as stray top-level parts.
#   breaks:  which of the file's parts open on a new page: "all", "first", or None. Short parts
#            (5-8) run on after one another; a page per half-page part reads as padding.
SECTIONS = [
    ("00_summary.md", "summary", 1, True, "all"),        # its H1 is the old title, dropped
    ("01_metric_and_data.md", None, 0, True, "first"),   # already ## 1. / ## 2. with ### below
    ("02_pipeline.md", "pipeline", 1, True, "all"),
    ("03_characterisation.md", "characterisation", 1, True, "all"),
    ("04_morphology_and_practice.md", None, 0, True, "first"),  # ## 5.-8. with ### below
    ("08_references.md", None, 1, False, "all"),          # # References -> ##
    ("05_appendix_a_ablations.md", None, 1, False, "all"),  # # Appendix A -> ##, a real heading
    ("07_appendix_bc.md", None, 1, False, "all"),        # # Appendix B, # Appendix C -> ##
    ("06_appendix_d_scripts.md", None, 1, False, "all"),
]
ABSTRACT = "00_abstract.md"

STRINGS = {
    "en": {
        "title": "Solar Filament Segmentation Challenge 2026: Method and Measured Findings",
        "subtitle": "Technical report",
        "author": "Talha Bacak (Kaggle team TalhaBacak)",
        "date": "September 2026; corrected 15 November 2026",
        "contents": "Contents",
        "abstract": "Abstract",
        "summary": "Overview",
        "pipeline": "3. Pipeline",
        "characterisation": "4. Empirical characterisation of the problem",
    },
    "tr": {
        "title": "Solar Filament Segmentation Challenge 2026: Yöntem ve Ölçülmüş Bulgular",
        "subtitle": "Teknik rapor. Raporun aslı İngilizcedir; bu, birebir çevirisidir ve her"
                    " sayı İngilizce sürümle aynıdır.",
        "author": "Talha Bacak (Kaggle takımı TalhaBacak)",
        "date": "Eylül 2026; 15 Kasım 2026'da düzeltildi",
        "contents": "İçindekiler",
        "abstract": "Özet",
        "summary": "Genel bakış",
        "pipeline": "3. Boru hattı",
        "characterisation": "4. Problemin ampirik betimlemesi",
    },
}

HEADING = re.compile(r"^(#{1,6}) +(.*?)\s*#*\s*$")


def source_text(name, lang):
    """The translated file if there is one, else English, else English with a translated head."""
    tr = SRC / lang / name
    if lang != "en" and tr.exists():
        return tr.read_text(encoding="utf-8")
    head = SRC / lang / (name.replace(".md", ".head.md"))
    body = (SRC / name).read_text(encoding="utf-8")
    if lang != "en" and head.exists():
        # keep the English body from its first H2 onward, replace everything above it
        m = re.search(r"^## ", body, flags=re.M)
        assert m, f"{name}: no H2 to cut at, cannot apply a head override"
        return head.read_text(encoding="utf-8").rstrip("\n") + "\n\n" + body[m.start():]
    return body


def clean(name, wrapper, demote, drop_h1, breaks, lang):
    lines = source_text(name, lang).splitlines()
    out, fenced, dropped, parts = [], False, False, 0
    if wrapper:
        out += [f"## {wrapper} {{.newpage}}" if breaks else f"## {wrapper}", ""]
        parts = 1
    for ln in lines:
        if ln.lstrip().startswith("```"):
            fenced = not fenced
        if not fenced:
            m = HEADING.match(ln)
            if m:
                if drop_h1 and not dropped and len(m.group(1)) == 1:
                    dropped = True
                    continue
                level = len(m.group(1)) + demote
                ln = "#" * level + " " + m.group(2)
                if level == 2:
                    parts += 1
                    if breaks == "all" or (breaks == "first" and parts == 1):
                        ln += " {.newpage}"
            elif re.fullmatch(r"-{3,}\s*", ln):
                # a horizontal rule is a drafting separator; parts are separated by headings
                # and page breaks in the typeset document
                continue
        out.append(ln)
    text = "\n".join(out)
    return re.sub(r"\n{3,}", "\n\n", text).strip("\n")


def abstract_yaml(lang):
    body = source_text(ABSTRACT, lang)
    body = "\n".join(ln for ln in body.splitlines() if not ln.startswith("# ")).strip("\n")
    return "abstract: |\n" + "\n".join(("  " + ln) if ln else "" for ln in body.splitlines())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir", nargs="?", default=str(ROOT / "outputs" / "report"))
    ap.add_argument("--lang", default="en", choices=sorted(STRINGS))
    a = ap.parse_args()
    outdir, lang = Path(a.outdir), a.lang
    S = STRINGS[lang]
    outdir.mkdir(parents=True, exist_ok=True)

    bodies = [clean(n, S[w] if w else None, d, h1, b, lang) for n, w, d, h1, b in SECTIONS]
    joined = "\n\n".join(bodies)

    q = json.dumps  # a JSON string is a valid YAML double-quoted scalar, colons and dashes included
    head = ["---",
            f"title: {q(S['title'], ensure_ascii=False)}",
            f"subtitle: {q(S['subtitle'], ensure_ascii=False)}",
            f"author: {q(S['author'], ensure_ascii=False)}",
            f"date: {q(S['date'], ensure_ascii=False)}",
            f"lang: {lang}",
            f"toc-title: {q(S['contents'], ensure_ascii=False)}",
            f"abstract-title: {q(S['abstract'], ensure_ascii=False)}",
            abstract_yaml(lang),
            "---", ""]
    md = "\n".join(head) + "\n" + joined + "\n"
    name = "report.md" if lang == "en" else f"report_{lang}.md"
    (outdir / name).write_text(md, encoding="utf-8")

    figs = SRC / "figures"
    if figs.is_dir():
        (outdir / "figures").mkdir(exist_ok=True)
        for f in figs.glob("*.png"):
            shutil.copy2(f, outdir / "figures" / f.name)

    links = re.findall(r"!\[(?:[^\[\]]|\[[^\]]*\])*\]\(([^)]+)\)", md)
    missing = [m for m in links if not (outdir / m).exists()]
    assert not missing, f"figure links do not resolve from the build directory: {missing}"
    heads = [HEADING.match(ln) for ln in joined.splitlines()]
    parts = sum(1 for m in heads if m and len(m.group(1)) == 2)
    subs = sum(1 for m in heads if m and len(m.group(1)) == 3)
    translated = (sum(1 for n, *_ in SECTIONS if (SRC / lang / n).exists())
                  if lang != "en" else len(SECTIONS))
    print(f"[{lang}] {translated}/{len(SECTIONS)} sources in this language -> {parts} parts, "
          f"{subs} subsections, {len(links)} figures, {len(md.split())} words "
          f"-> {outdir / name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
