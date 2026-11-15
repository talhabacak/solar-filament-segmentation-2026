"""Re-runnable scout of Kaggle community knowledge for filament-segmentation-2026.

Downloads (read-only, authenticated kaggle CLI required):
  - full public leaderboard CSV -> outputs/eda/public_leaderboard_snapshot_<date>.csv
  - key public kernels (incl. the organizer's official self-evaluation notebook)
    -> outputs/research/kernels/<n>/

NOTE: kernels are pulled into short-numbered subdirectories because the combination
of long repo paths + long kernel slugs exceeds Windows MAX_PATH and makes
`kaggle kernels pull` fail with FileNotFoundError.

Usage:  python src/eda/fetch_kaggle_community.py
"""
from __future__ import annotations

import datetime as _dt
import subprocess
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_EDA = REPO_ROOT / "outputs" / "eda"
OUT_KERNELS = REPO_ROOT / "outputs" / "research" / "kernels"

COMPETITION = "filament-segmentation-2026"

KERNELS = [
    "azimahmadzadeh/self-evaluation-notebook",          # OFFICIAL scorer (organizer)
    "artkomissar/please-fix-the-leaderboard",            # old-metric exploit disclosure
    "ektarr/new-metric-solar-filament",                  # documents Aug-7 Dice->PQ change
    "xiaoleilian/hardcore-eda-filament-segmentation-2026",  # leak count, rescore date, IAA
    "phuongncn/lb-0-70-topology-safe-solar-filaments-handoff",
    "timgluhkikh/best-score-magfilo-yolo11s-detect-crop-u-net",
    "muhammadshoaibaltaf/magfilo-v9-evaluation-aligned-solar-filament-s",
    "dattadhebe/solar-filament-grouped-cv-and-strategy-writeup",
    "vmerckle/simple-dark-filament-segmentation-0-46",
    "hdjojo/solar-filament-seg-inference",               # public LB leader (0.53) inference
]


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def fetch_leaderboard() -> None:
    OUT_EDA.mkdir(parents=True, exist_ok=True)
    run(["kaggle", "competitions", "leaderboard", "-c", COMPETITION, "-d", "-p", str(OUT_EDA)])
    zip_path = OUT_EDA / f"{COMPETITION}.zip"
    stamp = _dt.date.today().isoformat()
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            target = OUT_EDA / f"public_leaderboard_snapshot_{stamp}.csv"
            target.write_bytes(zf.read(name))
            print("wrote", target)
    zip_path.unlink(missing_ok=True)


def fetch_kernels() -> None:
    for index, ref in enumerate(KERNELS, start=1):
        dest = OUT_KERNELS / str(index)  # short path: avoids Windows MAX_PATH failures
        dest.mkdir(parents=True, exist_ok=True)
        try:
            run(["kaggle", "kernels", "pull", ref, "-p", str(dest)])
        except subprocess.CalledProcessError as error:
            print(f"WARN: failed to pull {ref}: {error}", file=sys.stderr)
        (dest / "SOURCE.txt").write_text(ref + "\n", encoding="utf-8")


if __name__ == "__main__":
    fetch_leaderboard()
    fetch_kernels()
