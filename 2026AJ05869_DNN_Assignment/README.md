# DNN Assignment: RNN, LSTM, GRU and Transformer

## 1. Before running
1. Rename the top folder and final ZIP to `<StudentID>_DNN_Assignment`.
2. Open `config.py` and replace `REPLACE_WITH_STUDENT_ID`.
3. Install the packages in `requirements.txt`.
4. Complete the clearly marked `STUDENT TODO` sections in `src/student_models.py` using your own implementation.

The assessed from-scratch components are intentionally provided as specifications and commented TODOs, not completed solutions. This supports the requirement that the recurrent cells, attention, multi-head attention and encoder block are the student's own code.

## 2. Recommended Google Colab steps
1. Upload and extract this package to the Colab session.
2. Select **Runtime > Change runtime type > T4 GPU**.
3. Run `!pip install -r requirements.txt` from the assignment folder.
4. Run `problem1/problem1.ipynb` using **Runtime > Restart session and run all**.
5. Run `problem2/problem2.ipynb` the same way.
6. Keep notebook outputs saved.
7. Run `!python validate_submission.py`.

## 3. Local execution
```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
jupyter notebook
```
Open each notebook and use **Restart Kernel and Run All Cells**.

## 4. Full experiments
`config.py` starts with `SMOKE_MODE=True` so the notebooks can validate imports and data plumbing quickly. Before collecting final evidence:
- set `SMOKE_MODE=False`;
- use all three configured seeds;
- run HAR for RNN, LSTM and GRU with the same training budget;
- run the adding task for T = 50, 100, 200 and 400;
- perform all required sweeps and ablations;
- save every run immediately to `results.csv`.

## 5. Outputs
- `results.csv`: exact required column order.
- `problem1/figures/`: class balance, confusion matrix, adding curves, gradient norms, hidden-size sweep and assigned ablation.
- `problem2/figures/`: attention heatmaps, learning curves, head ablation, no-mask result and context time/memory.
- `problem2/generated_samples.txt`: three labelled 500-character samples.
- `report.pdf`: replace the template with your final report, maximum 6 content pages.

## 6. Known input gap
The brief refers to a hand-checkable instructor-supplied three-token attention example, but no numeric example appears in the provided document. Obtain that example from the instructor and enter it in Problem 2.
