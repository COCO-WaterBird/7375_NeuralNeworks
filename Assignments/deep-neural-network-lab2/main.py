import csv
import os
import random

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# 1. Setup


os.makedirs("results", exist_ok=True)

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)


# 2. Task 2: Build the binary classification dataset (Experiment 1)
#
# The dataset is generated once with a fixed seed, so the split is always
# identical. The training split is exported to results/training_data.csv and
# the standardization statistics to results/scaler.npz, so the training stage
# below can reuse exactly the training set produced in Experiment 1.

DATA_CSV = "results/training_data.csv"
SCALER_NPZ = "results/scaler.npz"

X, y = make_moons(
    n_samples=1000,
    noise=0.20,
    random_state=42
)

# Split into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Standardize input features using training data only
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Input features:", X_train.shape[1])

# Save dataset preview
np.savetxt(
    DATA_CSV,
    np.column_stack((X_train, y_train)),
    delimiter=",",
    header="Feature_1,Feature_2,Label",
    comments=""
)

# Save the standardization statistics so the training stage can reload the
# training set without regenerating it.
np.savez(
    SCALER_NPZ,
    mean=scaler.mean_,
    scale=scaler.scale_,
)

# Plot training data
plt.figure(figsize=(8, 6))
plt.scatter(
    X_train[:, 0],
    X_train[:, 1],
    c=y_train,
    cmap="coolwarm",
    s=18,
    alpha=0.8
)
plt.title("Two Moons Training Dataset")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.tight_layout()
plt.savefig("results/dataset.png", dpi=150)
plt.close()


# 3. Task 1: Build the required deep neural network
# Architecture:
# (10/ReLU) - (8/ReLU)^2 - (4/ReLU) - (1/Sigmoid)


model = tf.keras.Sequential([
    tf.keras.Input(shape=(2,)),

    tf.keras.layers.Dense(10, activation="relu"),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("\nNeural Network Architecture:")
model.summary()


# 4. Train the neural network
#
# The training set built in Experiment 1 is reloaded from
# results/training_data.csv (the standardized features plus labels) and fed
# to the network. 20% of it is held out for validation, the remaining 80%
# drives the gradient updates. The 200 test samples are never seen by fit().

if os.path.exists(DATA_CSV):
    training_set = np.loadtxt(DATA_CSV, delimiter=",", skiprows=1)
    X_train = training_set[:, :-1]
    y_train = training_set[:, -1]
    print(f"\nLoaded training set from {DATA_CSV}: {X_train.shape}")
else:
    print(f"\n{DATA_CSV} not found, using the in-memory training set.")

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=20,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath="results/best_model.keras",
        monitor="val_loss",
        save_best_only=True,
        verbose=0
    )
]

history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

# Per-epoch training log
with open("results/training_log.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "epoch",
        "train_loss",
        "train_accuracy",
        "val_loss",
        "val_accuracy",
    ])
    for epoch, values in enumerate(
        zip(
            history.history["loss"],
            history.history["accuracy"],
            history.history["val_loss"],
            history.history["val_accuracy"],
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


# 5. Evaluate the trained model


test_loss, test_accuracy = model.evaluate(
    X_test, y_test, verbose=0
)

probabilities = model.predict(X_test, verbose=0).ravel()
y_pred = (probabilities >= 0.5).astype(int)

print("\n========== RESULTS ==========")
print(f"Epochs completed: {len(history.history['loss'])}")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

# Save metrics
with open("results/metrics.txt", "w") as f:
    f.write(f"Epochs completed: {len(history.history['loss'])}\n")
    f.write(f"Test Loss: {test_loss:.4f}\n")
    f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
    f.write(f"Accuracy Score: {accuracy_score(y_test, y_pred):.4f}\n")
    f.write("Confusion Matrix:\n")
    f.write(str(confusion_matrix(y_test, y_pred)))
    f.write("\n\nClassification Report:\n")
    f.write(classification_report(y_test, y_pred, digits=4))


# 6. Visualize training history


plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/accuracy.png", dpi=150)
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/loss.png", dpi=150)
plt.close()


# 7. Visualize the decision boundary


x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 200),
    np.linspace(y_min, y_max, 200)
)

grid = np.c_[xx.ravel(), yy.ravel()]
grid_scaled = scaler.transform(grid)

Z = model.predict(grid_scaled, verbose=0)
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(
    xx, yy, Z,
    levels=np.linspace(0, 1, 21),
    cmap="coolwarm",
    alpha=0.6
)
plt.contour(
    xx, yy, Z,
    levels=[0.5],
    colors="black"
)
plt.scatter(
    X[:, 0], X[:, 1],
    c=y,
    cmap="coolwarm",
    edgecolors="black",
    linewidths=0.2,
    s=15
)
plt.title("Deep Neural Network Decision Boundary")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.tight_layout()
plt.savefig("results/decision_boundary.png", dpi=150)
plt.close()

print("\nAll result files saved to results/")
print("Lab 2 completed!")
