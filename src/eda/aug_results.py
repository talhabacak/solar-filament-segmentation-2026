"""Read the augmentation study's trajectories, not just its final numbers.

The headline is that more augmentation lost: light 0.6995, heavy 0.6975, medium
0.6960. But three numbers cannot say WHY, and the why decides what to do next.
Two hypotheses make opposite predictions about the curves:

  H1  the augmentation attacks the wrong axis. Then it raises the training loss
      without moving validation at all -- the model was never overfitting along
      the photometric/geometric axes, so making them harder just wastes capacity.
      Prediction: train loss up, val flat or slightly down, and the two arms
      converge at the same epoch as light.

  H2  the 30-epoch budget is too short. Heavy augmentation slows convergence
      because each epoch shows a harder problem, so the arm is still climbing
      when the schedule ends. Prediction: heavy's val curve is still rising at
      epoch 29, and its best epoch is later than light's.

H1 says stop; the axis is dead. H2 says rerun long, and it would also settle the
ledger's "100 epochs degrades" negative, which was recorded under the LIGHT
recipe and therefore cannot distinguish a too-long schedule from a
too-thin-augmentation one.

Usage: python src/eda/aug_results.py
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "eda"
ARMS = ("light", "medium", "heavy")
REF = 0.6981          # sem_b3_f0_768_cldice, the recorded reference


def log(*a):
    print(*a, flush=True)


def main():
    H = {}
    for a in ARMS:
        p = ROOT / "outputs" / "runs_aug" / f"aug_{a}_f0_768" / "history.json"
        H[a] = json.load(open(p))
    n = min(len(h) for h in H.values())
    log(f"{n} epochs per arm\n")

    log("final state and where the best epoch fell:")
    log(f"{'arm':8s} {'best':>7s} {'at ep':>6s} {'last val':>9s} "
        f"{'last loss':>10s} {'d vs light':>11s}")
    best = {}
    for a in ARMS:
        v = [x["val_dice"] for x in H[a]]
        l = [x["train_loss"] for x in H[a]]
        b = int(np.argmax(v))
        best[a] = max(v)
        log(f"{a:8s} {max(v):7.4f} {b:6d} {v[-1]:9.4f} {l[-1]:10.4f} "
            f"{max(v)-best['light']:+11.4f}")
    log(f"\nreference sem_b3_f0_768_cldice = {REF:.4f}; light reproduces it at "
        f"{best['light']:.4f} ({best['light']-REF:+.4f}) -> harness is sound")

    # H2's test: is the heavy arm still climbing at the end?
    log("\nH2 -- is the 30-epoch budget the binding constraint?")
    for a in ARMS:
        v = np.array([x["val_dice"] for x in H[a]])
        tail = v[-6:]
        slope = np.polyfit(np.arange(len(tail)), tail, 1)[0]
        log(f"  {a:8s} best at epoch {int(np.argmax(v)):2d}/{len(v)-1}, "
            f"last-6-epoch slope {slope:+.5f}/epoch, "
            f"final-minus-best {v[-1]-v.max():+.4f}")
    log("  a still-climbing arm shows a positive tail slope AND its best in the "
        "last few epochs")

    # H1's test: did augmentation raise the training loss without buying val?
    log("\nH1 -- did the extra difficulty buy any generalisation?")
    log(f"  {'arm':8s} {'train loss':>11s} {'val':>8s} "
        f"{'loss cost':>10s} {'val gain':>9s} {'exchange':>10s}")
    l0 = H["light"][-1]["train_loss"]
    v0 = best["light"]
    for a in ARMS:
        l = H[a][-1]["train_loss"]
        dv, dl = best[a] - v0, l - l0
        ex = (dv / dl) if abs(dl) > 1e-6 else float("nan")
        log(f"  {a:8s} {l:11.4f} {best[a]:8.4f} {dl:+10.4f} {dv:+9.4f} "
            f"{ex:+10.3f}")
    log("  'exchange' is val gained per unit of training loss paid. Negative "
        "means the difficulty bought nothing and cost accuracy.")

    # Convergence shape: when did each arm reach 99% of its own best?
    log("\nconvergence speed (epoch at which each arm first hit 99% of its best):")
    for a in ARMS:
        v = np.array([x["val_dice"] for x in H[a]])
        hit = int(np.argmax(v >= 0.99 * v.max()))
        log(f"  {a:8s} epoch {hit:2d}   (best {v.max():.4f} at {int(np.argmax(v)):2d})")

    # Per-epoch cost of the augmentation
    log("\nwall clock per epoch (augmentation is CPU-side, 3 workers):")
    for a in ARMS:
        s = np.median([x["secs"] for x in H[a]])
        log(f"  {a:8s} {s:5.1f} s/epoch")

    verdict = {}
    for a in ARMS:
        v = np.array([x["val_dice"] for x in H[a]])
        tail = np.polyfit(np.arange(6), v[-6:], 1)[0]
        verdict[a] = dict(best=float(v.max()), best_ep=int(np.argmax(v)),
                          tail_slope=float(tail),
                          last_loss=float(H[a][-1]["train_loss"]))
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "aug_results.json", "w") as f:
        json.dump(verdict, f, indent=1)
    log(f"\nsaved -> {OUT/'aug_results.json'}")


if __name__ == "__main__":
    main()
