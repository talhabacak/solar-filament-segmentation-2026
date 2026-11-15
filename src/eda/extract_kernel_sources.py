"""Extract markdown+code cells from pulled Kaggle kernels (.ipynb) into plain .txt files.

Re-runnable: paths are relative to the repo root, derived from this file's location.
Usage: python src/eda/extract_kernel_sources.py
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
KERNELS = REPO / "outputs" / "research" / "kernels"


def main():
    for nb_path in sorted(KERNELS.glob("*/*.ipynb")):
        out_path = nb_path.with_suffix(".extracted.txt")
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        lines = []
        for i, cell in enumerate(nb.get("cells", [])):
            ctype = cell.get("cell_type", "?")
            src = cell.get("source", [])
            if isinstance(src, list):
                src = "".join(src)
            lines.append(f"\n===== CELL {i} [{ctype}] =====\n")
            lines.append(src)
        out_path.write_text("".join(lines), encoding="utf-8")
        print(f"{nb_path.parent.name}: {len(nb.get('cells', []))} cells -> {out_path.name}")


if __name__ == "__main__":
    main()
