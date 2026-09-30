"""One-off: move closed-loop outputs that were written to the wrong folder (--apply to actually move)."""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import REPO_ROOT, STUDY_ROOT

WRONG_ROOT = REPO_ROOT / "results" / "studies" / "gru_architecture_history_sensitivity" / "results"
CORRECT_ROOT = STUDY_ROOT / "results"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--apply", action="store_true",
                   help="Actually move files. Without this, only prints what would move.")
    args = p.parse_args()

    if not WRONG_ROOT.exists():
        print(f"{WRONG_ROOT} does not exist -- nothing to migrate.")
        return

    n_moved = 0
    n_conflict = 0
    for wrong_dir in sorted(WRONG_ROOT.glob("*/stage*/*/closed_loop_eval")):
        rel = wrong_dir.relative_to(WRONG_ROOT)
        correct_dir = CORRECT_ROOT / rel
        correct_dir.mkdir(parents=True, exist_ok=True)
        for f in sorted(wrong_dir.iterdir()):
            if not f.is_file():
                continue
            dest = correct_dir / f.name
            if dest.exists():
                print(f"[CONFLICT] {dest} already exists -- leaving {f} in place, not overwriting.")
                n_conflict += 1
                continue
            print(f"{'[MOVE]' if args.apply else '[DRY RUN would move]'} {f} -> {dest}")
            if args.apply:
                f.rename(dest)
            n_moved += 1

    print(f"\n{n_moved} file(s) {'moved' if args.apply else 'would be moved'}, {n_conflict} conflict(s).")
    if args.apply and n_conflict == 0:
        import shutil
        shutil.rmtree(REPO_ROOT / "results" / "studies")
        print(f"Removed the now-empty {REPO_ROOT / 'results' / 'studies'}")
    elif args.apply:
        print("Conflicts found -- leaving results/studies/ in place for manual review.")


if __name__ == "__main__":
    main()
