
# Lab 1 – Multilayer Deep Neural Network

## Objective
Develop a deep neural network for binary classification
using the architecture specified in Lab 1.

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

The model uses the Adam optimizer and binary
cross-entropy loss function.

## Task 2: Training Dataset

The Two Moons dataset was generated using scikit-learn.

- Total samples: 1,000
- Input features: 2
- Classes: 0 and 1
- Training set: 800 samples
- Test set: 200 samples

Input features were standardized using StandardScaler.

## Model Training

- Epochs: 100
- Batch size: 32
- Optimizer: Adam
- Loss: Binary Cross-Entropy
- Evaluation metric: Accuracy

## Results

The results folder contains:

- dataset.png
- accuracy.png
- loss.png
- decision_boundary.png
- training_data.csv
- metrics.txt

## How to Run

1. Create a Python virtual environment.
2. Install dependencies:

   pip install -r requirements.txt

3. Run:

   python main.py

## Technologies

Python, TensorFlow, Keras, NumPy,
Matplotlib, scikit-learn


## Model Evaluation Results

- Total Parameters: 231
- Training Epochs: 100
- Test Accuracy: 99.50%
- Test Loss: 0.0354

### Confusion Matrix

[[99, 1],
 [0, 100]]

The model correctly classified 199 out of 200 test samples.

The results demonstrate that the five-layer deep neural
network successfully learned the nonlinear decision
boundary of the Two Moons dataset.
