"""Run before creating the final ZIP. Checks structure, schema and notebook JSON."""
from pathlib import Path
import csv, json, sys
from src.common import RESULT_COLUMNS
ROOT = Path(__file__).resolve().parent
REQUIRED = [
    "README.md", "requirements.txt", "report.pdf", "results.csv",
    "problem1/problem1.ipynb", "problem2/problem2.ipynb",
    "problem2/generated_samples.txt", "problem1/figures", "problem2/figures"
]
errors = [f"Missing: {name}" for name in REQUIRED if not (ROOT / name).exists()]
with (ROOT / "results.csv").open(encoding="utf-8") as handle:
    header = next(csv.reader(handle), [])
if header != RESULT_COLUMNS:
    errors.append("results.csv does not have the exact required columns/order.")
for notebook in [ROOT / "problem1/problem1.ipynb", ROOT / "problem2/problem2.ipynb"]:
    with notebook.open(encoding="utf-8") as handle:
        json.load(handle)
if errors:
    print("VALIDATION FAILED")
    print("\n".join(f"- {error}" for error in errors))
    sys.exit(1)
print("Basic package validation passed.")
print("Manual checks: Restart and Run All, saved outputs, final evidence, report <=6 pages, ZIP <=25 MB.")
