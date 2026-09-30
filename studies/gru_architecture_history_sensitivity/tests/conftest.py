import sys
from pathlib import Path

STUDY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STUDY_ROOT.parents[1]
for p in (STUDY_ROOT / "scripts", REPO_ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
