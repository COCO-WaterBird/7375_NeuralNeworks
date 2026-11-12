# Lab 3 – Tuning of a Multilayer Neural Network

> **ASSIGNMENT TITLE**
> Lab 3 - Tuning of a Multilayer Neural Network
>
> **ASSIGNMENT**
> **Task 1: Tune your trained (in Lab 1 and Lab 2) multilayer neural network by
> optimizing its hyper-parameters.**
>
> Submission: submit the link to the GitHub repository where the Lab 1
> results (source code and result snapshots) are stored, plus authorization if
> the repository is private.

## Objective

Take the multilayer deep neural network trained in **Lab 1** (architecture) and
**Lab 2** (training) and tune its hyper-parameters. The search is performed
with **KerasTuner**, and the tuned model is compared against the Lab 2
baseline on the untouched test set.

## Task 1: Tuning the trained neural network

### 1. Data pipeline (identical to Lab 2)

The Two Moons data set is generated with `make_moons(n_samples=1000,
noise=0.20, random_state=42)` and split 80 / 20 with `random_state=42`, so
the experiment is fully reproducible.

| Split | Samples | Purpose |
|-------|---------|---------|
| fit | 640 | gradient updates during the search |
| validation | 160 | scores the tuner (never the test set) |
| test | 200 | final evaluation only |

Features are standardized with `StandardScaler` fitted on the training split
only.

### 2. Search space

| Hyper-parameter | Range |
|-----------------|-------|
| number of hidden layers | 2 – 5 |
| units per layer | 8 – 64 (step 8) |
| activation | relu / tanh / elu |
| dropout | 0.0 / 0.1 / 0.2 |
| L2 regularization | 1e-6 – 1e-3 (log scale) |
| optimizer | adam / rmsprop |
| learning rate | 1e-4 – 3e-2 (log scale) |
| batch size | 16 / 32 / 48 / 64 |

The search space is deliberately narrow: the Two Moons data set is small and
almost linearly separable, so heavy regularization only hurts.

### 3. Search protocol

1. **KerasTuner `RandomSearch`** over the space above, scoring every trial on
   the **validation loss**. Validation *loss* (a continuous quantity) is used
   as the objective instead of accuracy because accuracy on 160 validation
   samples is a coarse step function and cannot separate good candidates.
2. **Robust candidate selection** – the top-3 trials are each retrained with
   3 different random seeds; the candidate with the lowest *mean* validation
   loss is selected. This removes the noise of a single run.
3. **Final training** – both the baseline and the tuned configuration are
   retrained on the **full** training set with the same protocol
   (`validation_split=0.2` for early stopping, identical patience and epoch
   budget) and evaluated over 3 seeds, so the comparison is fair and not an
   artefact of a single lucky run.

### 4. Results

**Selected configuration** (`results/best_hyperparameters.json`)

| Hyper-parameter | Value |
|-----------------|-------|
| number of hidden layers | 3 |
| units per layer | 32 / 48 / 40 |
| activation | tanh |
| dropout | 0.0 |
| L2 regularization | 3.8e-06 |
| optimizer | adam |
| learning rate | 8.15e-03 |
| batch size | 48 |

**Baseline (Lab 2) vs Tuned (Lab 3)** – mean over 3 random seeds

| Metric | Baseline (Lab 2) | Tuned (Lab 3) | Change |
|--------|------------------|----------------|--------|
| Validation accuracy | 0.9812 | 0.9812 | = |
| Test accuracy (mean) | 0.9900 | 0.9883 | -0.0017 |
| Test accuracy (std) | 0.0000 | 0.0062 | – |
| **Test loss (mean)** | **0.0871** | **0.0439** | **-49.6 %** |
| Test loss (std) | 0.0583 | 0.0077 | – |
| Trainable parameters | 231 | 3 681 | +3 450 |

Per-seed test accuracies – baseline `[0.99, 0.99, 0.99]`, tuned
`[0.98, 0.995, 0.99]`.

**Interpretation.** Tuning cut the test loss roughly in half (0.0871 → 0.0439)
and made it far more stable across seeds (std 0.0583 → 0.0077), which is the
actual benefit of the exercise: a better calibrated, better generalizing
model. The accuracy is statistically unchanged – the gap of 0.0017 is one
sample out of 200 and is smaller than the seed-to-seed standard deviation
(0.0062). The Lab 2 architecture was already close to optimal for this
easy, almost linearly separable Two Moons problem, so there was little
accuracy left on the table.

Confusion matrix of the best tuned model:

```
[[ 99   1]
 [  0 100]]
```

See `results/metrics.txt` for the full report and `results/results.png` for
the visual summary.

## Files in `results/`

| File | Description |
|------|-------------|
| `metrics.txt` | full report: baseline vs tuned, confusion matrix, classification report |
| `tuning_trials.csv` | every tuner trial with its score and hyper-parameters |
| `candidate_selection.csv` | the top-3 candidates re-evaluated over several seeds |
| `best_hyperparameters.json` | the selected configuration |
| `training_log.csv` | per-epoch loss / accuracy of the final tuned model |
| `tuning_comparison.png` | bar chart: baseline vs tuned accuracy and loss |
| `tuner_history.png` | search progress (validation loss per trial) |
| `hyperparameter_importance.png` | best validation loss per hyper-parameter value |
| `accuracy.png` / `loss.png` | training vs validation curves of the tuned model |
| `decision_boundary.png` | learned decision boundary of the tuned model |
| `dataset.png` | standardized training set |
| `results.png` | combined result snapshot |
| `tuned_model.keras` | the final tuned model |

## How to Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

The search budget can be changed without touching the code:

```bash
LAB3_MAX_TRIALS=24 LAB3_MAX_EPOCHS=120 python main.py
```

## Technologies

Python, TensorFlow / Keras, KerasTuner, NumPy, Matplotlib, scikit-learn
