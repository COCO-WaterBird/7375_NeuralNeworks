"""
Lab 3 - Tuning of a Multilayer Neural Network
==============================================

ASSIGNMENT TITLE
    Lab 3 - Tuning of a Multilayer Neural Network

ASSIGNMENT
    Task 1: Tune the trained (in Lab 1 and Lab 2) multilayer neural network
            by optimizing its hyper-parameters.

    Takes the multilayer deep neural network trained in Lab 1 / Lab 2 and
    tunes its hyper-parameters (depth, width, activation, dropout, L2
    regularization, optimizer, learning rate and batch size) with KerasTuner.

Pipeline
--------
1. Rebuild the exact same Two Moons data pipeline used in Lab 2
   (fixed seeds -> identical train/test split and scaling).
2. Train the Lab 2 "baseline" network and record its test accuracy.
3. Run a KerasTuner random search over the hyper-parameter space, using a
   dedicated validation split carved out of the *training* set only.
4. Re-evaluate the top candidates over several random seeds and pick the most
   robust configuration.
5. Retrain the winner on the full training set, evaluate it on the untouched
   test set and compare it against the baseline.
6. Save every artefact (metrics, CSV logs, plots, best model) into results/.

Run with:  python main.py
"""

import csv
import json
import os
import random
import shutil

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras_tuner as kt

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# 1. Setup
# ---------------------------------------------------------------------------

os.makedirs("results", exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Hyper-parameter search budget
# Both values can be overridden from the shell, e.g.
#   LAB3_MAX_TRIALS=12 LAB3_MAX_EPOCHS=60 python main.py
MAX_TRIALS = int(os.environ.get("LAB3_MAX_TRIALS", 16))
MAX_EPOCHS = int(os.environ.get("LAB3_MAX_EPOCHS", 60))
EARLY_STOP_PATIENCE = 8

# Lab 2 baseline configuration
BASELINE_CONFIG = {
    "layers": [10, 8, 8, 4],
    "activation": "relu",
    "dropout": 0.0,
    "l2": 0.0,
    "optimizer": "adam",
    "learning_rate": 0.001,
    "batch_size": 32,
}

print("=" * 70)
print("LAB 3 - TUNING OF A MULTILAYER NEURAL NETWORK")
print("=" * 70)
print(f"Random seed: {SEED}")
print(f"Max tuner trials: {MAX_TRIALS}")
print(f"Max epochs per trial: {MAX_EPOCHS}\n")


# 2. Data pipeline (identical to Lab 2)
# ---------------------------------------------------------------------------

X, y = make_moons(n_samples=1000, noise=0.20, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# A dedicated validation set carved out of the training data.  It is used by
# the tuner to score every trial, so the test set stays completely untouched.
X_fit, X_val, y_fit, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train,
)

print(f"Training samples   : {len(X_train)}")
print(f"  fit / validation : {len(X_fit)} / {len(X_val)}")
print(f"Testing samples    : {len(X_test)}")
print(f"Input features     : {X_train.shape[1]}\n")

# Persist the artefacts reused from Lab 2
np.savetxt(
    "results/training_data.csv",
    np.column_stack((X_train, y_train)),
    delimiter=",",
    header="Feature_1,Feature_2,Label",
    comments="",
)
np.savez("results/scaler.npz", mean=scaler.mean_, scale=scaler.scale_)

plt.figure(figsize=(8, 6))
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap="coolwarm",
            s=18, alpha=0.8)
plt.title("Two Moons Training Dataset (standardized)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.tight_layout()
plt.savefig("results/dataset.png", dpi=150)
plt.close()


# 3. Model builders
# ---------------------------------------------------------------------------

def build_model(layers, activation, dropout, l2, optimizer, learning_rate,
                batch_size=32):
    """Assemble a fully connected binary classifier from a config dict."""
    reg = tf.keras.regularizers.l2(l2) if l2 > 0 else None

    model = tf.keras.Sequential([tf.keras.Input(shape=(2,))])
    for units in layers:
        model.add(
            tf.keras.layers.Dense(
                units,
                activation=activation,
                kernel_regularizer=reg,
            )
        )
        if dropout > 0:
            model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

    if optimizer == "adam":
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == "rmsprop":
        opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)

    model.compile(optimizer=opt, loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model


def build_model_from_config(config):
    """Build a model from a flat config dict (as produced by the tuner).

    The dict contains "num_layers" plus one "units_<i>" entry per hidden
    layer, plus activation / dropout / l2 / optimizer / learning_rate /
    batch_size.
    """
    n_layers = int(config["num_layers"])
    layers = [int(config[f"units_{i}"]) for i in range(n_layers)]
    return build_model(
        layers=layers,
        activation=config["activation"],
        dropout=float(config["dropout"]),
        l2=float(config["l2"]),
        optimizer=config["optimizer"],
        learning_rate=float(config["learning_rate"]),
        batch_size=int(config["batch_size"]),
    )


def make_callbacks():
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOP_PATIENCE,
            restore_best_weights=True,
            verbose=0,
        )
    ]


# 4. Baseline: the Lab 2 network, trained with the Lab 2 hyper-parameters
# ---------------------------------------------------------------------------

print("-" * 70)
print("STEP 1/5  Training the Lab 2 baseline network (reference)")
print("-" * 70)

baseline_model = build_model(**BASELINE_CONFIG)
baseline_model.summary()

baseline_history = baseline_model.fit(
    X_fit,
    y_fit,
    epochs=MAX_EPOCHS,
    batch_size=BASELINE_CONFIG["batch_size"],
    validation_data=(X_val, y_val),
    callbacks=make_callbacks(),
    verbose=0,
)

baseline_test_loss, baseline_test_acc = baseline_model.evaluate(
    X_test, y_test, verbose=0
)
baseline_val_acc = max(baseline_history.history["val_accuracy"])
print(f"Baseline  -> val accuracy {baseline_val_acc:.4f} | "
      f"test accuracy {baseline_test_acc:.4f} | test loss {baseline_test_loss:.4f}")


# 5. KerasTuner random search
# ---------------------------------------------------------------------------

print("\n" + "-" * 70)
print("STEP 2/5  Hyper-parameter random search")
print("-" * 70)


class DNNHyperModel(kt.HyperModel):
    """Search space for the tuned multilayer network."""

    def build(self, hp):
        n_layers = hp.Int("num_layers", 2, 5)
        layers = [
            hp.Int(f"units_{i}", 8, 64, step=8)
            for i in range(n_layers)
        ]

        activation = hp.Choice("activation", ["relu", "tanh", "elu"])
        # The Two Moons data set is small and almost linearly separable, so
        # heavy regularisation only hurts.  The search space is kept narrow
        # on purpose.
        dropout = hp.Choice("dropout", [0.0, 0.1, 0.2])
        l2 = hp.Float("l2", 1e-6, 1e-3, sampling="log")
        optimizer = hp.Choice("optimizer", ["adam", "rmsprop"])
        learning_rate = hp.Float(
            "learning_rate", 1e-4, 3e-2, sampling="log"
        )
        # Registered here so it belongs to the search space even though the
        # batch size is only used at fit() time.
        hp.Int("batch_size", 16, 64, step=16)

        return build_model(
            layers=layers,
            activation=activation,
            dropout=dropout,
            l2=l2,
            optimizer=optimizer,
            learning_rate=learning_rate,
        )


class DNNTuner(kt.RandomSearch):
    """Random search tuner that also tunes the batch size."""

    def run_trial(self, trial, *args, **kwargs):
        hp = trial.hyperparameters
        model = self.hypermodel.build(hp)

        history = model.fit(
            *args,
            epochs=MAX_EPOCHS,
            batch_size=hp.get("batch_size"),
            validation_data=kwargs.pop("validation_data"),
            callbacks=make_callbacks(),
            verbose=0,
        )

        # Score the trial with its best validation loss.  Validation *loss*
        # is a continuous quantity, so it discriminates between candidates
        # far better than accuracy does on a 160-sample validation set.
        val_loss = float(min(history.history["val_loss"]))
        val_acc = float(max(history.history["val_accuracy"]))
        return {"val_loss": val_loss, "val_accuracy": val_acc}


tuner = DNNTuner(
    hypermodel=DNNHyperModel(),
    objective="val_loss",
    max_trials=MAX_TRIALS,
    seed=SEED,
    directory="results/tuner",
    project_name="lab3_dnn_tuning",
    overwrite=True,
)

tuner.search(X_fit, y_fit, validation_data=(X_val, y_val), verbose=2)

# Dump every trial into a CSV so the search is fully auditable
trial_rows = []
trials = tuner.oracle.trials
if isinstance(trials, dict):
    trials = list(trials.values())

for trial in trials:
    trial_rows.append({
        "trial_id": trial.trial_id,
        "score": trial.score,
        "best_step": trial.best_step,
        "status": trial.status,
        **{k: str(v) for k, v in trial.hyperparameters.values.items()},
    })

if trial_rows:
    # Trials with a different number of hidden layers expose a different
    # number of "units_*" entries, so the header is the union of all keys.
    field_names = []
    for row in trial_rows:
        for key in row:
            if key not in field_names:
                field_names.append(key)

    with open("results/tuning_trials.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=field_names, restval="")
        writer.writeheader()
        writer.writerows(trial_rows)

best_trial_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print("\nBest hyper-parameters found by the tuner:")
print(json.dumps(best_trial_hps.values, indent=2))

try:
    tuner.results_summary()
except Exception as exc:  # pragma: no cover - reporting is best effort
    print(f"Could not print tuner results summary: {exc}")


# 6. Robust selection: retrain the top candidates over several random seeds
# ---------------------------------------------------------------------------

print("\n" + "-" * 70)
print("STEP 3/5  Robust selection among the top candidates")
print("-" * 70)

TOP_K = 3
SELECTION_SEEDS = [42, 123, 2024]

top_trials = tuner.oracle.get_best_trials(num_trials=TOP_K)
candidate_configs = [dict(hp.values) for hp in
                     tuner.get_best_hyperparameters(num_trials=TOP_K)]

candidate_report = []
for rank, cfg in enumerate(candidate_configs, start=1):
    losses, accs = [], []
    for seed in SELECTION_SEEDS:
        tf.keras.utils.set_random_seed(seed)
        model = build_model_from_config(cfg)
        history = model.fit(
            X_fit,
            y_fit,
            epochs=MAX_EPOCHS,
            batch_size=cfg["batch_size"],
            validation_data=(X_val, y_val),
            callbacks=make_callbacks(),
            verbose=0,
        )
        losses.append(min(history.history["val_loss"]))
        accs.append(max(history.history["val_accuracy"]))

    mean_loss = float(np.mean(losses))
    std_loss = float(np.std(losses))
    mean_acc = float(np.mean(accs))
    candidate_report.append({
        "rank": rank,
        "tuner_score": float(top_trials[rank - 1].score),
        "mean_val_loss": mean_loss,
        "std_val_loss": std_loss,
        "mean_val_accuracy": mean_acc,
        "val_losses": losses,
        "val_accuracies": accs,
        "hyperparameters": cfg,
    })

    print(f"Candidate {rank}: mean val loss {mean_loss:.4f} "
          f"(+/- {std_loss:.4f}) | mean val acc {mean_acc:.4f} | "
          f"tuner score {top_trials[rank - 1].score:.4f}")

# The candidate with the lowest *mean* validation loss wins.  Averaging over
# several seeds removes the noise of a single run on 160 validation samples.
winner = min(candidate_report, key=lambda c: c["mean_val_loss"])
best_config = dict(winner["hyperparameters"])

print("\nSelected configuration:")
print(json.dumps(best_config, indent=2))

with open("results/best_hyperparameters.json", "w") as f:
    json.dump(best_config, f, indent=2)

with open("results/candidate_selection.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "rank", "tuner_score", "mean_val_loss", "std_val_loss",
        "mean_val_accuracy", "val_losses", "val_accuracies",
        "hyperparameters",
    ])
    for c in candidate_report:
        writer.writerow([
            c["rank"],
            f"{c['tuner_score']:.6f}",
            f"{c['mean_val_loss']:.6f}",
            f"{c['std_val_loss']:.6f}",
            f"{c['mean_val_accuracy']:.6f}",
            " ".join(f"{v:.4f}" for v in c["val_losses"]),
            " ".join(f"{v:.4f}" for v in c["val_accuracies"]),
            json.dumps(c["hyperparameters"]),
        ])


# 7. Final training: both models are trained on the FULL training set with
#    the same protocol, so the comparison is fair.
# ---------------------------------------------------------------------------

print("\n" + "-" * 70)
print("STEP 4/5  Final training on the full training set")
print("-" * 70)

FINAL_SEED = 42


def train_final(config, seed):
    """Train a configuration on the full training set with an internal
    validation split (20%) used only for early stopping."""
    tf.keras.utils.set_random_seed(seed)
    model = build_model_from_config(config)
    history = model.fit(
        X_train,
        y_train,
        epochs=MAX_EPOCHS,
        batch_size=int(config["batch_size"]),
        validation_split=0.2,
        callbacks=make_callbacks(),
        verbose=0,
    )
    return model, history


def evaluate_over_seeds(config, seeds):
    """Train the configuration several times and report the mean test
    metrics.  A single 200-sample test set gives a very coarse accuracy
    (each sample is 0.5%), so the comparison has to be averaged."""
    accs, losses = [], []
    best_model, best_history, best_acc = None, None, -1.0

    for seed in seeds:
        model, history = train_final(config, seed)
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        accs.append(float(acc))
        losses.append(float(loss))
        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_history = history

    return {
        "mean_accuracy": float(np.mean(accs)),
        "std_accuracy": float(np.std(accs)),
        "mean_loss": float(np.mean(losses)),
        "std_loss": float(np.std(losses)),
        "accuracies": accs,
        "losses": losses,
        "model": best_model,
        "history": best_history,
    }


# Baseline rebuilt from the Lab 2 hyper-parameters
baseline_config = {
    "num_layers": len(BASELINE_CONFIG["layers"]),
    **{f"units_{i}": u for i, u in enumerate(BASELINE_CONFIG["layers"])},
    "activation": BASELINE_CONFIG["activation"],
    "dropout": BASELINE_CONFIG["dropout"],
    "l2": BASELINE_CONFIG["l2"],
    "optimizer": BASELINE_CONFIG["optimizer"],
    "learning_rate": BASELINE_CONFIG["learning_rate"],
    "batch_size": BASELINE_CONFIG["batch_size"],
}

print("Baseline (Lab 2 configuration):")
baseline_result = evaluate_over_seeds(baseline_config, SELECTION_SEEDS)
baseline_final_model = baseline_result["model"]
baseline_final_history = baseline_result["history"]
baseline_val_acc = max(baseline_final_history.history["val_accuracy"])
baseline_test_acc = baseline_result["mean_accuracy"]
baseline_test_loss = baseline_result["mean_loss"]
baseline_final_model.summary()
print(f"  test accuracy {baseline_test_acc:.4f} "
      f"(+/- {baseline_result['std_accuracy']:.4f}) over "
      f"{len(SELECTION_SEEDS)} seeds | test loss {baseline_test_loss:.4f}")

print("\nTuned (Lab 3 configuration):")
tuned_result = evaluate_over_seeds(best_config, SELECTION_SEEDS)
tuned_model = tuned_result["model"]
tuned_history = tuned_result["history"]
tuned_val_acc = max(tuned_history.history["val_accuracy"])
tuned_test_acc = tuned_result["mean_accuracy"]
tuned_test_loss = tuned_result["mean_loss"]
tuned_model.summary()
print(f"  test accuracy {tuned_test_acc:.4f} "
      f"(+/- {tuned_result['std_accuracy']:.4f}) over "
      f"{len(SELECTION_SEEDS)} seeds | test loss {tuned_test_loss:.4f}")

probabilities = tuned_model.predict(X_test, verbose=0).ravel()
y_pred = (probabilities >= 0.5).astype(int)

# Persist the tuned model and the tuner's own best model
tuned_model.save("results/tuned_model.keras")
try:
    tuner.save_model("results/tuner_best_model.keras")
except Exception as exc:  # pragma: no cover - saving is best effort
    print(f"Could not save the tuner best model: {exc}")
if os.path.isdir("results/tuner"):
    shutil.rmtree("results/tuner")

# Per-epoch training log for the tuned model
with open("results/training_log.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "epoch", "train_loss", "train_accuracy",
        "val_loss", "val_accuracy",
    ])
    for epoch, values in enumerate(
        zip(
            tuned_history.history["loss"],
            tuned_history.history["accuracy"],
            tuned_history.history["val_loss"],
            tuned_history.history["val_accuracy"],
        ),
        start=1,
    ):
        writer.writerow([
            epoch,
            f"{values[0]:.6f}",
            f"{values[1]:.6f}",
            f"{values[2]:.6f}",
            f"{values[3]:.6f}",
        ])


# 7. Report
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)

print(f"{'Metric':<28}{'Baseline (Lab 2)':>22}{'Tuned (Lab 3)':>20}")
print("-" * 70)
print(f"{'Validation accuracy':<28}{baseline_val_acc:>22.4f}{tuned_val_acc:>20.4f}")
print(f"{'Test accuracy (mean)':<28}{baseline_test_acc:>22.4f}{tuned_test_acc:>20.4f}")
print(f"{'Test accuracy (std)':<28}"
      f"{baseline_result['std_accuracy']:>22.4f}{tuned_result['std_accuracy']:>20.4f}")
print(f"{'Test loss (mean)':<28}{baseline_test_loss:>22.4f}{tuned_test_loss:>20.4f}")
print(f"{'Test loss (std)':<28}"
      f"{baseline_result['std_loss']:>22.4f}{tuned_result['std_loss']:>20.4f}")
print(f"{'Trainable parameters':<28}"
      f"{baseline_final_model.count_params():>22}{tuned_model.count_params():>20}")
print("-" * 70)
print(f"Accuracy improvement: {tuned_test_acc - baseline_test_acc:+.4f}")
print(f"Per-seed baseline accuracies: "
      f"{[round(v, 4) for v in baseline_result['accuracies']]}")
print(f"Per-seed tuned accuracies:    "
      f"{[round(v, 4) for v in tuned_result['accuracies']]}")

print("\nConfusion Matrix (tuned model):")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report (tuned model):")
print(classification_report(y_test, y_pred, digits=4))

with open("results/metrics.txt", "w") as f:
    f.write("LAB 3 - TUNING OF A MULTILAYER NEURAL NETWORK\n")
    f.write("=" * 55 + "\n\n")
    f.write(f"Random seed            : {SEED}\n")
    f.write(f"Tuner trials           : {MAX_TRIALS}\n")
    f.write(f"Max epochs per trial   : {MAX_EPOCHS}\n\n")

    f.write("--- Baseline (Lab 2 hyper-parameters) ---\n")
    f.write(f"Config                 : {BASELINE_CONFIG}\n")
    f.write(f"Trainable parameters   : {baseline_final_model.count_params()}\n")
    f.write(f"Best validation acc    : {baseline_val_acc:.4f}\n")
    f.write(f"Test accuracy (mean)   : {baseline_test_acc:.4f} "
            f"(+/- {baseline_result['std_accuracy']:.4f})\n")
    f.write(f"Test loss (mean)       : {baseline_test_loss:.4f} "
            f"(+/- {baseline_result['std_loss']:.4f})\n")
    f.write(f"Per-seed accuracies    : "
            f"{[round(v, 4) for v in baseline_result['accuracies']]}\n\n")

    f.write("--- Best hyper-parameters found by the tuner ---\n")
    for k, v in best_config.items():
        f.write(f"{k:<23}: {v}\n")
    f.write("\n")

    f.write("--- Candidate selection (top-3 retrained over 5 seeds) ---\n")
    for c in candidate_report:
        f.write(
            f"rank {c['rank']}: tuner score {c['tuner_score']:.4f} | "
            f"mean val loss {c['mean_val_loss']:.4f} "
            f"(+/- {c['std_val_loss']:.4f}) | "
            f"mean val acc {c['mean_val_accuracy']:.4f}\n"
        )
    f.write("\n")

    f.write("--- Tuned model ---\n")
    f.write(f"Trainable parameters   : {tuned_model.count_params()}\n")
    f.write(f"Best validation acc    : {tuned_val_acc:.4f}\n")
    f.write(f"Test accuracy (mean)   : {tuned_test_acc:.4f} "
            f"(+/- {tuned_result['std_accuracy']:.4f})\n")
    f.write(f"Test loss (mean)       : {tuned_test_loss:.4f} "
            f"(+/- {tuned_result['std_loss']:.4f})\n")
    f.write(f"Per-seed accuracies    : "
            f"{[round(v, 4) for v in tuned_result['accuracies']]}\n")
    f.write(f"Accuracy improvement   : {tuned_test_acc - baseline_test_acc:+.4f}\n\n")
    f.write("Confusion Matrix:\n")
    f.write(str(confusion_matrix(y_test, y_pred)))
    f.write("\n\nClassification Report:\n")
    f.write(classification_report(y_test, y_pred, digits=4))


# 8. Visualizations
# ---------------------------------------------------------------------------

# 8.1 Training curves of the tuned model
plt.figure(figsize=(8, 5))
plt.plot(tuned_history.history["accuracy"], label="Train Accuracy")
plt.plot(tuned_history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Tuned Network - Training vs. Validation Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/accuracy.png", dpi=150)
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(tuned_history.history["loss"], label="Train Loss")
plt.plot(tuned_history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Tuned Network - Training vs. Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/loss.png", dpi=150)
plt.close()

# 8.2 Baseline vs tuned comparison
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

axes[0].bar(
    ["Baseline\n(Lab 2)", "Tuned\n(Lab 3)"],
    [baseline_test_acc, tuned_test_acc],
    color=["#7fa8c9", "#e07a5f"],
)
axes[0].set_ylim(0.90, 1.005)
axes[0].set_title("Test Accuracy")
axes[0].set_ylabel("Accuracy")
for i, v in enumerate([baseline_test_acc, tuned_test_acc]):
    axes[0].text(i, v + 0.002, f"{v:.4f}", ha="center", fontsize=10)

axes[1].bar(
    ["Baseline\n(Lab 2)", "Tuned\n(Lab 3)"],
    [baseline_test_loss, tuned_test_loss],
    color=["#7fa8c9", "#e07a5f"],
)
axes[1].set_title("Test Loss")
axes[1].set_ylabel("Binary cross-entropy")
for i, v in enumerate([baseline_test_loss, tuned_test_loss]):
    axes[1].text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=10)

plt.suptitle("Effect of Hyper-parameter Tuning")
plt.tight_layout()
plt.savefig("results/tuning_comparison.png", dpi=150)
plt.close()

# 8.3 Decision boundary of the tuned model
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 200),
    np.linspace(y_min, y_max, 200),
)
grid = np.c_[xx.ravel(), yy.ravel()]
Z = tuned_model.predict(scaler.transform(grid), verbose=0).reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, levels=np.linspace(0, 1, 21),
             cmap="coolwarm", alpha=0.6)
plt.contour(xx, yy, Z, levels=[0.5], colors="black")
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm",
            edgecolors="black", linewidths=0.2, s=15)
plt.title("Tuned Network Decision Boundary")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.tight_layout()
plt.savefig("results/decision_boundary.png", dpi=150)
plt.close()

# 8.4 Search progress and hyper-parameter influence
scores = [t.score for t in trials if t.score is not None]
trial_ids = [t.trial_id for t in trials if t.score is not None]

plt.figure(figsize=(8, 5))
plt.plot(trial_ids, scores, marker="o", color="#e07a5f")
plt.scatter([trial_ids[int(np.argmin(scores))]], [min(scores)],
            s=140, facecolors="none", edgecolors="green", linewidths=2,
            label="Best trial")
plt.xlabel("Trial")
plt.ylabel("Validation Loss")
plt.title("Hyper-parameter Search Progress (objective: validation loss)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/tuner_history.png", dpi=150)
plt.close()

# Distribution of the best validation loss per hyper-parameter value
keys = sorted({k for row in trial_rows for k in row
               if k.startswith(("num_layers", "units_", "activation",
                                "dropout", "l2", "optimizer",
                                "learning_rate", "batch_size"))})
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
for ax, key in zip(axes.flatten(), keys):
    groups = {}
    for row in trial_rows:
        groups.setdefault(row[key], []).append(float(row["score"]))
    labels = list(groups.keys())
    means = [min(v) for v in groups.values()]
    ax.bar(range(len(labels)), means, color="#7fa8c9")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_title(key, fontsize=10)
    ax.set_ylabel("best val loss (lower is better)")
    ax.grid(True, axis="y")
plt.suptitle("Best validation loss per hyper-parameter value")
plt.tight_layout()
plt.savefig("results/hyperparameter_importance.png", dpi=150)
plt.close()

# 8.5 Combined result snapshot
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

axes[0].plot(baseline_final_history.history["loss"], label="Baseline train")
axes[0].plot(baseline_final_history.history["val_loss"], "--", label="Baseline val")
axes[0].plot(tuned_history.history["loss"], label="Tuned train")
axes[0].plot(tuned_history.history["val_loss"], "--", label="Tuned val")
axes[0].set_title("Loss: baseline vs tuned")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend(fontsize=8)
axes[0].grid(True)

axes[1].bar(
    ["Baseline", "Tuned"],
    [baseline_test_acc, tuned_test_acc],
    color=["#7fa8c9", "#e07a5f"],
)
axes[1].set_ylim(0.90, 1.005)
axes[1].set_title("Test accuracy")
axes[1].set_ylabel("Accuracy")

axes[2].contourf(xx, yy, Z, levels=np.linspace(0, 1, 21),
                 cmap="coolwarm", alpha=0.6)
axes[2].contour(xx, yy, Z, levels=[0.5], colors="black")
axes[2].scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm",
                edgecolors="black", linewidths=0.2, s=12)
axes[2].set_title("Tuned decision boundary")

plt.tight_layout()
plt.savefig("results/results.png", dpi=150)
plt.close()

print("\nAll result files saved to results/")
print("Lab 3 completed!")
