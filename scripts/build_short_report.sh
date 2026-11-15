#!/usr/bin/env bash
# Build the four-page technical report in every language it has a source for.
#
# Source: report/short/report_<lang>.md (pandoc Markdown). Outputs, in report/:
#   Technical_Report_<LANG>.pdf   two-column US Letter, printed by a headless Chromium browser
#   short/build/report_<lang>.html
#   short/overleaf/report_<lang>.tex   the same text as a LaTeX body, for the organisers' template
#
# No LaTeX engine is needed: pandoc writes HTML, report/short/short.css sets the page, and Edge or
# Chrome prints it. The page count is printed at the end; the report must be four pages.
#
# Usage: bash scripts/build_short_report.sh [lang ...]      (default: en tr)
set -u
cd "$(dirname "$0")/.."
SRC=report/short
BUILD=$SRC/build
mkdir -p "$BUILD" "$SRC/overleaf"

LANGS=("$@")
[ ${#LANGS[@]} -eq 0 ] && LANGS=(en tr)

BROWSER=""
for b in "/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe" \
         "/c/Program Files/Microsoft/Edge/Application/msedge.exe" \
         "/c/Program Files/Google/Chrome/Application/chrome.exe" \
         "$(command -v google-chrome 2>/dev/null)" "$(command -v chromium 2>/dev/null)"; do
  [ -n "$b" ] && [ -x "$b" ] && { BROWSER="$b"; break; }
done

for L in "${LANGS[@]}"; do
  MD="$SRC/report_$L.md"
  [ -f "$MD" ] || { echo "no source for $L"; continue; }
  UP=$(echo "$L" | tr '[:lower:]' '[:upper:]')
  HTML="$BUILD/report_$L.html"
  PDF="report/Technical_Report_$UP.pdf"

  pandoc "$MD" -o "$HTML" --standalone --embed-resources --mathml \
      --template "$SRC/template.html" --css "$SRC/short.css" --resource-path="$SRC"
  python scripts/short_report_tables.py "$HTML"

  pandoc "$MD" -o "$SRC/overleaf/report_$L.tex" --to latex --wrap=preserve
  python scripts/short_report_tex.py "$SRC/overleaf/report_$L.tex"

  if [ -z "$BROWSER" ]; then
    echo "no Chromium-based browser found; print $HTML to PDF by hand"
    continue
  fi
  ABS=$(pwd -W 2>/dev/null || pwd)
  rm -f "$PDF"
  "$BROWSER" --headless --disable-gpu --no-pdf-header-footer \
      --print-to-pdf="$ABS/$PDF" "file:///$ABS/$HTML" 2>/dev/null
  for _ in $(seq 1 30); do [ -s "$PDF" ] && break; sleep 1; done
  sleep 1
  python -c "import sys; from pypdf import PdfReader; print(sys.argv[1], len(PdfReader(sys.argv[1]).pages), 'pages')" "$PDF"
done
