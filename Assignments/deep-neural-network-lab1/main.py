
import os
import random

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix


# 1. Setup


os.makedirs("results", exist_ok=True)

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)


# 2. Task 2: Create a binary classification dataset


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
    "results/training_data.csv",
    np.column_stack((X_train, y_train)),
    delimiter=",",
    header="Feature_1,Feature_2,Label",
    comments=""
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
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("\nNeural Network Architecture:")
model.summary()


# 4. Train the neural network


history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)


# 5. Evaluate the trained model


test_loss, test_accuracy = model.evaluate(
    X_test, y_test, verbose=0
)

probabilities = model.predict(X_test, verbose=0).ravel()
y_pred = (probabilities >= 0.5).astype(int)

print("\n========== RESULTS ==========")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save metrics
with open("results/metrics.txt", "w") as f:
    f.write(f"Test Loss: {test_loss:.4f}\n")
    f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
    f.write("Confusion Matrix:\n")
    f.write(str(confusion_matrix(y_test, y_pred)))

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
print("Lab 1 completed!")
