# Checkpoint 4

## Goal

Our main goal was to understand why did merged dataset produce such awesome results compared to single datasets

## Solution

We did adverserial validation and also additional analysis to determine whether we had data leakage or not. Results were the following:

```
===============
ADVERSARIAL VALIDATION — VERDICT
======================================================================
Adversarial train-vs-test AUC : 0.505  (~0.5 ⇒ no real held-out gap)
Source-identification AUC     : 1.000
Test-window subject overlap   : 1.000
Adjacent near-duplicate frac  : 0.976
bal-acc random / grouped      : 0.955 / 0.791   gap = +0.164
----------------------------------------------------------------------
  [X] test-window subject overlap > 0.5
  [X] adjacent near-duplicate fraction > 0.05
  [X] random - grouped bal-acc gap > 0.10
----------------------------------------------------------------------
LEAKAGE CONFIRMED: True
Dominant mechanism(s):
  - overlapping near-duplicate windows across the split (97.6% of test)
  - subject leakage (100.0% of test windows share a train subject)
  - trivially separable sources (OVR-AUC 1.00)

Recommended fix:
  Replace the random StratifiedShuffleSplit in
  src/merged_dataset.py::load_and_prepare_data with a SUBJECT-GROUPED,
  class-stratified split (StratifiedGroupKFold / GroupShuffleSplit on the
  subject array, which the loader must read from the npz). This keeps every
  subject — and therefore all its overlapping windows — entirely within one split,
  eliminating both the near-duplicate and subject-identity leakage.
======================================================================
```

After that, we fixed the leakage: removed overlapped timeseries and also replaced our split with "group" split by subject (which is basically user_id)

```
Epoch 21: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 698/698 [00:41<00:00, 16.97it/s, loss=0.0345, acc=0.9714]
✓ Epoch 21 | Loss: 0.0607 | Acc: 0.9763

✓ Val Balanced Accuracy: 0.8911 | Val F1 Macro: 0.8812
   downstairs    1138 / 8669 samples   F1: 0.7543
   sit           1287 / 8669 samples   F1: 0.9339
   stand         1463 / 8669 samples   F1: 0.9320
   upstairs      1444 / 8669 samples   F1: 0.8539
   walk          3337 / 8669 samples   F1: 0.9320
```

So, the results are looking way more realistic and we can proceed with optimizing this algorithm with supervised / self-supervised approaches
