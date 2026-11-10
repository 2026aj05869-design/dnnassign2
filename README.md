# 2026AJ05869 DNN Assignment

**Student:** Rajesh Garg  
**Student ID:** 2026AJ05869

## Chosen values under the clarification
The variants table and related FAQ are intentionally ignored.

- Common recurrent hidden size: **h = 64** for RNN, LSTM and GRU
- Transformer width: **d = 128**
- Default attention heads: **4**, so `128 % 4 == 0`
- Three seeds: **7, 17, 27**

These values must also appear on the report title page. Parameter counts quoted elsewhere for other widths must not be copied; use the counts printed by this package for the chosen configuration.

## Student-specific experiment
Student ID 2026AJ05869 ends in 9. The package therefore keeps the previously assigned Problem 1 individual experiment as **LSTM forget-gate bias 0 versus 1 at T=200**. This is separate from the now-ignored variants table.

## Academic-integrity boundary
The assessed from-scratch recurrent cells, scaled attention, multi-head attention, positional encoding and encoder block are marked `STUDENT TODO`. Complete those implementations yourself and retain the verification tests. The package provides data pipelines, framework comparison models, experiment utilities, checks, comments, report structure and the clarified 3-token worked example.

## Quick run in Google Colab
1. Upload and extract `2026AJ05869_DNN_Assignment.zip`.
2. Select **Runtime > Change runtime type > T4 GPU**.
3. Change directory into the extracted folder.
4. Run `!python -m pip install -r requirements.txt`.
5. Run `!python preflight.py`.
6. Open `problem1/problem1.ipynb`, complete the assessed TODOs, then **Restart session and run all**.
7. Open `problem2/problem2.ipynb`, complete the assessed TODOs, then **Restart session and run all**.
8. Set `SMOKE_MODE=False` in `config.py` for final experiments.
9. Replace the report template with your final report and run `!python validate_submission.py`.

## Local run
```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python preflight.py
jupyter notebook
```

## Clarified 3-token example
See `problem2/THREE_TOKEN_ATTENTION.md`. It contains small Q, K and V matrices, the hand calculation, expected scores, attention weights and output. Test the student attention implementation against those expected values.

## Final packaging
Do not include `data/`, checkpoints, virtual environments, caches or downloaded archives. Keep notebook outputs, `results.csv`, figures, samples and final `report.pdf`.
