#!/usr/bin/env bash
# Assemble the report into Markdown, HTML and PDF, in every language it has sources for.
#
# Appendix D is regenerated first so a built artifact can never carry a stale script index.
# The assembler normalises heading depth, writes the title block and abstract, copies the
# figures next to the output and asserts every image link resolves; a language directory that
# is incomplete falls back to English file by file, so a partial translation still builds.
#
# Markdown -> HTML -> PDF via pandoc and Edge's headless printer: no LaTeX engine is installed
# on this machine, and Edge exists on any Windows box, so the build needs no toolchain the
# next person has to install first. Typography lives in docs/report/report.css; equations are
# rendered as MathML, which Chromium typesets natively. The PDF is printed twice: the first
# print is read back to put page numbers into the table of contents, the second carries them.
#
# Usage: bash scripts/build_report.sh [lang ...]      (default: en tr)
set -u
cd "$(dirname "$0")/.."
OUT=outputs/report
mkdir -p "$OUT"

python src/eda/gen_script_index.py

LANGS=("$@")
[ ${#LANGS[@]} -eq 0 ] && LANGS=(en tr)

EDGE="/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"
[ -x "$EDGE" ] || EDGE="/c/Program Files/Microsoft/Edge/Application/msedge.exe"

print_pdf() {  # $1 = stem; prints $OUT/$1.html to $OUT/$1.pdf and waits for the file
  local ABS
  ABS=$(cd "$OUT" && pwd -W 2>/dev/null || (cd "$OUT" && pwd))
  rm -f "$OUT/$1.pdf"
  "$EDGE" --headless --disable-gpu --no-pdf-header-footer \
      --print-to-pdf="${ABS}/$1.pdf" "file:///${ABS}/$1.html" 2>/dev/null
  for _ in $(seq 1 30); do [ -s "$OUT/$1.pdf" ] && break; sleep 1; done
  sleep 2
  [ -s "$OUT/$1.pdf" ]
}

for LANG_ in "${LANGS[@]}"; do
  if [ "$LANG_" = "en" ]; then STEM=report; else STEM="report_${LANG_}"; fi
  python scripts/assemble_report.py "$OUT" --lang "$LANG_" || continue
  MD="$OUT/$STEM.md"

  if ! command -v pandoc >/dev/null 2>&1; then
    echo "pandoc not installed; $MD is ready (figures in $OUT/figures/)"
    continue
  fi

  # --resource-path lets pandoc find figures/ next to the assembled Markdown; the title, author,
  # abstract and language come from the YAML block the assembler writes.
  pandoc "$MD" -o "$OUT/$STEM.html" --standalone --embed-resources \
      --toc --toc-depth=3 --mathml --css docs/report/report.css \
      --resource-path="$OUT"
  python scripts/toc_pages.py prepare "$OUT/$STEM.html"
  echo "wrote $OUT/$STEM.html"

  if [ ! -x "$EDGE" ]; then
    echo "no Edge found; print $STEM.html from any browser"
    continue
  fi
  if ! print_pdf "$STEM"; then
    echo "Edge produced no PDF for $LANG_; print $STEM.html from a browser"
    continue
  fi
  # an entry whose heading is not found stays blank and is reported; the second print still runs
  python scripts/toc_pages.py fill "$OUT/$STEM.html" "$OUT/$STEM.pdf" \
    || echo "contents page numbers incomplete for $LANG_ (see above)"
  print_pdf "$STEM" && echo "wrote $OUT/$STEM.pdf"
done
