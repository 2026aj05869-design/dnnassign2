"""Checks imports, configuration, schema and non-assessed data helpers."""
import csv,sys
from config import *
from src.common import RESULT_COLUMNS
from src.data import assert_adding_data_is_valid
validate_choices(); assert_adding_data_is_valid()
with RESULTS_CSV.open(encoding="utf-8") as f: header=next(csv.reader(f))
assert header==RESULT_COLUMNS
print("Preflight passed")
print({"student_id":STUDENT_ID,"h":HIDDEN_SIZE,"d":D_MODEL,"heads":N_HEADS,"seeds":SEEDS,"ablation":assigned_ablation()})
