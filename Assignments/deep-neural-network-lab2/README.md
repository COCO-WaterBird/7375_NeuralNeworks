# Lab 2 – Training the Multilayer Neural Network

## Objective

Train the multilayer deep neural network built in Lab 1 using the training
set produced by the Experiment 1 data pipeline, then evaluate the trained
model on the held-out test set.

## Task 1: Neural Network Architecture

The model contains five fully connected Dense layers:

| Layer | Neurons | Activation |
|-------|---------|------------|
| 1 | 10 | ReLU |
| 2 | 8 | ReLU |
| 3 | 8 | ReLU |
| 4 | 4 | ReLU |
| 5 | 1 | Sigmoid |

Architecture:
(10/ReLU) - (8/ReLU)^2 - (4/ReLU) - (1/Sigmoid)

Total trainable parameters: **231**

## Task 2: Training Dataset (Experiment 1)

The Two Moons dataset was generated using scikit-learn with a fixed seed, so
the split is reproducible.

- Total samples: 1,000
- Input features: 2
- Classes: 0 and 1
- Training set: 800 samples
- Test set: 200 samples

Input features were standardized using `StandardScaler` fitted on the training
split only; the test split is transformed with the same statistics.

The training split is exported to `results/training_data.csv` (standardized
features + label) together with the standardization statistics in
`results/scaler.npz`.

## Task 3: Training the Multilayer Neural Network

The training stage reloads the Experiment 1 training set from
`results/training_data.csv` and feeds it to the network.

- Epochs: 100 (max)
- Batch size: 32
- Optimizer: Adam, learning rate 0.001
- Loss: Binary Cross-Entropy
- Evaluation metric: Accuracy
- Validation split: 0.2 of the training set (640 train / 160 validation)
- Early stopping on `val_loss`, patience 20, best weights restored
- Best checkpoint saved to `results/best_model.keras`

The 200 test samples are never seen during `fit()`, so the reported test
metrics are unbiased.

## Results

The results folder contains:

- dataset.png – standardized training set scatter plot
- accuracy.png – training vs. validation accuracy curves
- loss.png – training vs. validation loss curves
- decision_boundary.png – learned nonlinear decision boundary
- training_data.csv – the Experiment 1 training set
- training_log.csv – per-epoch loss / accuracy log
- metrics.txt – test metrics and confusion matrix
- best_model.keras – best checkpoint (weights restored for evaluation)
- scaler.npz – standardization statistics

## How to Run

1. Create a Python virtual environment.
2. Install dependencies:

   pip install -r requirements.txt

3. Run:

   python main.py

## Technologies

Python, TensorFlow, Keras, NumPy, Matplotlib, scikit-learn

## Model Evaluation Results

- Total Parameters: 231
- Epochs completed: 96 (early stopping triggered, best epoch 76)
- Test Accuracy: **99.00%**
- Test Loss: 0.0400

### Confusion Matrix

```
[[ 98   2]
 [  0 100]]
```

The model correctly classified 198 out of 200 test samples (both errors are
false negatives on class 0). Per-class precision/recall are 1.00/0.98 for
class 0 and 0.98/1.00 for class 1.

The results demonstrate that the five-layer deep neural network successfully
learned the nonlinear decision boundary of the Two Moons dataset.
