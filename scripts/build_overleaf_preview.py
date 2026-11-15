"""Check the Overleaf sources of the technical report and print a preview PDF without LaTeX.

The report is written in the organisers' Overleaf template (report/overleaf/main.tex, ACM
`acmart`, sigconf). Overleaf is where it is compiled; this script exists because the machine the
report was written on has no LaTeX engine, and two things still had to be known before uploading:

  1. that the source is well formed: balanced braces and environments, every \\cite key present
     in main.bib, every \\ref matched by a \\label, every figure file on disk, no stray special
     characters;
  2. roughly how long it is. pandoc converts the LaTeX to HTML, report/overleaf/preview.css sets
     the page to the sigconf geometry (US Letter, two 240 pt columns, 9 pt type on 10 pt leading,
     Times in place of Linux Libertine, which is slightly wider and therefore errs on the long
     side), a headless Chromium browser prints it, and the page on which the Acknowledgment
     starts is reported. The template allows four pages before the Acknowledgment.

The preview is an estimate of the layout, not the layout: floats stay where they are written
instead of moving to the top of a column, and citations are printed author-year. The PDF that
counts is the one Overleaf compiles.

It also writes report/Overleaf_Project.zip (main.tex, preamble.tex, main.bib, figures/) for
Overleaf's "Upload Project". The Turkish review copy main_tr.tex is previewed but not packed.

Usage: python scripts/build_overleaf_preview.py [main.tex main_tr.tex ...]
"""
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "report" / "overleaf"
BUILD = SRC / "build"
BROWSERS = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            "google-chrome", "chromium", "chromium-browser"]


def strip_comments(tex):
    """Drop LaTeX comments (an unescaped % to the end of the line)."""
    return re.sub(r"(?<!\\)%.*", "", tex)


def lint(path):
    """Return a list of problems found in one .tex file; empty when it looks well formed."""
    tex = strip_comments(path.read_text(encoding="utf-8"))
    problems = []
    depth = 0
    for i, ch in enumerate(tex):
        if ch in "{}" and (i == 0 or tex[i - 1] != "\\"):
            depth += 1 if ch == "{" else -1
            if depth < 0:
                problems.append(f"closing brace without an opening one near: {tex[max(0, i - 40):i + 1]!r}")
                depth = 0
    if depth:
        problems.append(f"{depth} unclosed brace(s)")
    stack = []
    for m in re.finditer(r"\\(begin|end)\{([^}]+)\}", tex):
        if m.group(1) == "begin":
            stack.append(m.group(2))
        elif not stack or stack.pop() != m.group(2):
            problems.append(f"\\end{{{m.group(2)}}} does not close the open environment")
    problems += [f"environment never closed: {e}" for e in stack]
    if tex.count("$") % 2:
        problems.append("odd number of $ signs")

    bib = (SRC / "main.bib").read_text(encoding="utf-8")
    keys = set(re.findall(r"@\w+\{\s*([^,\s]+)\s*,", bib))
    cited = {k.strip() for m in re.finditer(r"\\cite\{([^}]+)\}", tex) for k in m.group(1).split(",")}
    problems += [f"cited but not in main.bib: {k}" for k in sorted(cited - keys)]
    labels = set(re.findall(r"\\label\{([^}]+)\}", tex))
    refs = set(re.findall(r"\\ref\{([^}]+)\}", tex))
    problems += [f"\\ref without \\label: {r}" for r in sorted(refs - labels)]
    for f in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex):
        if not (SRC / f).exists():
            problems.append(f"figure file missing: {f}")

    # special characters that must be escaped in running text
    body = re.sub(r"\\(?:label|ref|cite|includegraphics|url|href|input|bibliography|bibliographystyle|documentclass)"
                  r"(?:\[[^\]]*\])?\{[^}]*\}(?:\{[^}]*\})?", "", tex)
    body = re.sub(r"\$[^$]*\$", "", body)
    body = re.sub(r"\\begin\{equation\}.*?\\end\{equation\}", "", body, flags=re.S)
    for ch in "_#^":
        for m in re.finditer(r"(?<!\\)" + re.escape(ch), body):
            problems.append(f"unescaped {ch!r} near: {body[max(0, m.start() - 30):m.start() + 15]!r}")
    in_table = False
    for line in body.splitlines():
        if "\\begin{tabular}" in line:
            in_table = True
        if in_table and "\\\\" in line:
            cells = len(re.findall(r"(?<!\\)&", line)) + 1
            if "ncols" not in locals() or ncols is None:
                ncols = cells
            elif cells != ncols:
                problems.append(f"table row with {cells} cells in a {ncols}-column table: {line.strip()[:60]!r}")
        elif not in_table and re.search(r"(?<!\\)&", line):
            problems.append(f"unescaped '&' outside a table: {line.strip()[:60]!r}")
        if "\\end{tabular}" in line:
            in_table, ncols = False, None
    return problems, len(cited), len(labels)


def for_pandoc(tex):
    """The parts of the acmart source that pandoc cannot read are removed or simplified."""
    tex = re.sub(r"\\input\{preamble\}|\\guideStyle(?:true|false)|\\maketitle", "", tex)
    tex = re.sub(r"\\Description\{[^}]*\}", "", tex)
    tex = re.sub(r"\\affiliation\{%?(.*?)\n\}", "", tex, flags=re.S)
    tex = re.sub(r"\\email\{[^}]*\}", "", tex)
    tex = tex.replace("\\ref{eq:pq}", "1")          # pandoc does not number equations
    tex = re.sub(r"\\bibliographystyle\{[^}]*\}", "", tex)
    tex = tex.replace("\\begin{figure*}", "\\begin{figure}").replace("\\end{figure*}", "\\end{figure}")
    return tex.replace("\\documentclass[sigconf]{acmart}", "\\documentclass{article}")


def browser():
    for b in BROWSERS:
        if Path(b).exists():
            return b
        if shutil.which(b):
            return shutil.which(b)
    return None


def preview(path):
    """main.tex -> build/<stem>.html -> report/<stem>_preview.pdf; returns the PDF path or None."""
    BUILD.mkdir(parents=True, exist_ok=True)
    tex = path.read_text(encoding="utf-8")
    meta = {k: re.search(r"\\" + k + r"\{(.*?)\}\s*(?:%.*)?\n", tex).group(1) for k in ("title", "subtitle", "author")}
    lang = "tr" if path.stem.endswith("_tr") else "en"
    tmp = BUILD / f"{path.stem}.pandoc.tex"
    tmp.write_text(for_pandoc(tex), encoding="utf-8")
    html = BUILD / f"{path.stem}.html"
    subprocess.run(["pandoc", str(tmp), "-f", "latex", "-t", "html", "-o", str(html), "--standalone", "--mathml",
                    "--number-sections", "--citeproc", "--bibliography", str(SRC / "main.bib"),
                    "--template", str(SRC / "preview_template.html"), "--css", "../preview.css",
                    "--resource-path", str(SRC), "-V", f"lang={lang}", "-V", "abstract-title=" + ("Özet" if lang == "tr" else "Abstract"), "-V", f"subtitle={meta['subtitle']}",
                    "-M", f"title={meta['title']}", "-M", f"author={meta['author']}"], check=True, cwd=SRC)
    s = html.read_text(encoding="utf-8").replace('src="figures/', 'src="../figures/')
    html.write_text(s, encoding="utf-8")
    exe = browser()
    if not exe:
        print("no Chromium-based browser found; open", html, "and print it")
        return None
    pdf = ROOT / "report" / ("Technical_Report_TR_preview.pdf" if lang == "tr" else "Technical_Report_EN_preview.pdf")
    pdf.unlink(missing_ok=True)
    subprocess.run([exe, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf}", html.as_uri()], check=False, capture_output=True)
    return pdf if pdf.exists() else None


def report_length(pdf):
    """Total pages, and the page and height at which the Acknowledgment starts."""
    from pypdf import PdfReader
    pages = PdfReader(str(pdf)).pages
    where = None
    for n, p in enumerate(pages, start=1):
        text = p.extract_text() or ""
        if re.search(r"Acknowledg|ACKNOWLEDG|Teşekkür|TEŞEKKÜR", text):
            where = n
            break
    return len(pages), where


def main():
    files = [SRC / a for a in sys.argv[1:]] or [p for p in (SRC / "main.tex", SRC / "main_tr.tex") if p.exists()]
    ok = True
    for path in files:
        problems, n_cite, n_label = lint(path)
        print(f"{path.name}: {n_cite} citation keys, {n_label} labels, {len(problems)} problem(s)")
        for p in problems:
            print("   -", p)
        ok &= not problems
        pdf = preview(path)
        if pdf:
            total, ack = report_length(pdf)
            print(f"   preview {pdf.name}: {total} pages; the Acknowledgment starts on page {ack}")
    with zipfile.ZipFile(ROOT / "report" / "Overleaf_Project.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in [SRC / "main.tex", SRC / "preamble.tex", SRC / "main.bib"]:   # main_tr.tex is a review copy
            if p.exists():
                z.write(p, p.name)
        for p in sorted((SRC / "figures").glob("*.png")):
            z.write(p, f"figures/{p.name}")
    print("wrote report/Overleaf_Project.zip")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
