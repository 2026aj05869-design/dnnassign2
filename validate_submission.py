"""Final structural validator; it cannot judge scientific correctness."""
from pathlib import Path
import csv,json,sys
from config import *
from src.common import RESULT_COLUMNS
ROOT=Path(__file__).resolve().parent
required=["README.md","requirements.txt","report.pdf","results.csv","problem1/problem1.ipynb","problem2/problem2.ipynb","problem2/generated_samples.txt","problem2/THREE_TOKEN_ATTENTION.md"]
errors=[f"Missing {x}" for x in required if not (ROOT/x).exists()]
for nb in [ROOT/"problem1/problem1.ipynb",ROOT/"problem2/problem2.ipynb"]:
    if nb.exists(): json.load(nb.open(encoding="utf-8"))
if RESULTS_CSV.exists():
    with RESULTS_CSV.open(encoding="utf-8") as f:
        if next(csv.reader(f),[])!=RESULT_COLUMNS: errors.append("Incorrect results.csv schema")
for p in ROOT.rglob("*.py"):
    if ".venv" not in p.parts and "__pycache__" not in p.parts:
        text=p.read_text(encoding="utf-8")
        if p.name=="student_models.py" and "raise NotImplementedError" in text:
            errors.append("Assessed STUDENT TODO implementations are incomplete")
if errors:
    print("NOT READY"); print("\n".join("- "+e for e in errors)); sys.exit(1)
print("Structural validation passed. Also confirm Restart and Run All, final results/figures/samples, report <=6 pages and ZIP <=25 MB.")
