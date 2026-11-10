# Completion checklist

## Problem 1
- [ ] HAR four-subject validation split for each seed
- [ ] Training-only channel mean/std
- [ ] Class-balance plot
- [ ] Adding generator assert tests
- [ ] MyRNNCell/MyLSTMCell/MyGRUCell tests within 1e-5
- [ ] Hand and PyTorch parameter counts reconciled
- [ ] HAR three models x three seeds with equal budget
- [ ] Accuracy, macro-F1, parameters, seconds/epoch and best confusion matrix
- [ ] Adding T=50/100/200/400 x three models x three seeds
- [ ] Baseline 0.167, solved threshold 0.05, T=200 curves
- [ ] Gradient norms before/after at T=100
- [ ] LSTM/GRU hidden sizes 16/32/64/128
- [ ] Student-ID ablation and wearable recommendation

## Problem 2
- [ ] Attention rows sum to 1
- [ ] Output matches PyTorch SDPA within 1e-5
- [ ] Causal future weights are zero
- [ ] Instructor three-token hand example added
- [ ] MHA, positional encoding and pre-LN block from scratch
- [ ] Parameter count reconciliation
- [ ] HAR CLS classifier and positional ablation
- [ ] Shakespeare 90/10 chronological split, context 128, d=128, 4 heads, 4 layers
- [ ] Validation nats and bits/character
- [ ] Three 500-character top-k samples
- [ ] Two head heatmaps, heads 1/4/8, no-mask study
- [ ] Context 64/128/256/512 time/memory study

## Submission
- [ ] Every table/figure numbered, captioned and cited in report text
- [ ] Report content <=6 pages, A4 11pt, margins >=2 cm
- [ ] Notebook outputs saved and Restart/Run All succeeds
- [ ] No data, checkpoints or virtual environment in ZIP
- [ ] Correct StudentID ZIP/folder name and size <=25 MB
