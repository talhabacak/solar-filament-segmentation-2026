"""Turn pandoc's LaTeX body of the short report into something a two-column template accepts.

pandoc writes every table as a `longtable`, which LaTeX refuses in two-column mode, and wraps
figures in its own `\\pandocbounded` macro. This rewrites both: tables become `table` + `tabular`
with booktabs rules, the figure becomes a two-column `figure*`, and the abstract paragraph is put
inside an `abstract` environment. The result is a body to paste into the organisers' Overleaf
template after its title block; it needs the packages booktabs, array, calc, graphicx and amsmath.

The output has NOT been compiled on the machine that wrote it (no LaTeX engine is installed
there), so check the first Overleaf compile for errors.

Usage: python scripts/short_report_tex.py report/short/overleaf/report_en.tex
"""
import re
import sys
from pathlib import Path

HEADER = """% Body of the technical report, generated from report/short/report_{lang}.md.
% Paste after the title block of the organisers' template. Required packages:
%   \\usepackage{{booktabs,array,calc,graphicx,amsmath}}
% and, because pandoc writes compact lists with it:
%   \\providecommand{{\\tightlist}}{{\\setlength{{\\itemsep}}{{0pt}}\\setlength{{\\parskip}}{{0pt}}}}
% The figure file is docs/report/figures/fig4_fine_structure.png; upload it next to main.tex.

"""


SYMBOLS = {"−": "$-$", "±": "$\\pm$", "≥": "$\\geq$", "≤": "$\\leq$", "·": "$\\cdot$", "°": "$^\\circ$",
           "→": "$\\rightarrow$", "×": "$\\times$", "½": "$\\tfrac12$", "₁": "$_1$", "₂": "$_2$"}


def table(m):
    """One pandoc longtable -> a floating table with a tabular inside."""
    src = m.group(0)
    spec = re.search(r"\\begin\{longtable\}\[\]\{(.*?)\}\s*\n(?=\\caption|\\toprule)", src, flags=re.S).group(1)
    cap = re.search(r"\\caption\{(.*?)\}\\tabularnewline", src, flags=re.S)
    head = re.search(r"\\toprule\\noalign\{\}\n(.*?)\\midrule\\noalign\{\}", src, flags=re.S).group(1)
    body = re.search(r"\\endlastfoot\n(.*?)\\end\{longtable\}", src, flags=re.S).group(1)
    body = body.replace("\\bottomrule\\noalign{}\n", "")
    out = ["\\begin{table}[t]", "\\centering\\footnotesize"]
    if cap:
        out.append("\\caption{" + cap.group(1) + "}")
    out += ["\\begin{tabular}{" + spec + "}", "\\toprule", head.rstrip(), "\\midrule", body.rstrip(),
            "\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(out)


def settings_widths(tex):
    """The settings table has a short first column; pandoc splits the width evenly."""
    for label in ("Component", "Bileşen"):
        i = tex.find("\\begin{minipage}[b]{\\linewidth}\\raggedright\n" + label)
        if i < 0:
            continue
        j = tex.rfind("\\begin{tabular}", 0, i)
        spec = tex[j:i].replace("\\real{0.5000}", "\\real{0.2700}", 1).replace("\\real{0.5000}", "\\real{0.7300}", 1)
        return tex[:j] + spec + tex[i:]
    return tex


def main():
    path = Path(sys.argv[1])
    lang = path.stem.split("_")[-1]
    tex = path.read_text(encoding="utf-8")
    tex, n_tab = re.subn(r"\\begin\{longtable\}.*?\\end\{longtable\}", table, tex, flags=re.S)
    tex = settings_widths(tex)
    tex = re.sub(r"\\pandocbounded\{\\includegraphics\[keepaspectratio(?:,alt=\{.*?\})?\]\{.*?([^/{}]+\.png)\}\}",
                 r"\\includegraphics[width=0.8\\textwidth]{\1}", tex, flags=re.S)
    tex = tex.replace("\\begin{figure}", "\\begin{figure*}[t]").replace("\\end{figure}", "\\end{figure*}")
    for ch, latex in SYMBOLS.items():      # pdfLaTeX has no glyph for these without extra packages
        tex = tex.replace(ch, latex)
    tex, n_abs = re.subn(r"\\textbf\{(Abstract|Özet)\.\}\s*(.*?)\n\n", r"\\begin{abstract}\n\2\n\\end{abstract}\n\n",
                         tex, count=1, flags=re.S)
    path.write_text(HEADER.format(lang=lang) + tex, encoding="utf-8", newline="\n")
    print(f"{path.name}: {n_tab} table(s) rewritten, abstract wrapped: {bool(n_abs)}")


if __name__ == "__main__":
    main()
