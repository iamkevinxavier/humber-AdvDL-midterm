# Advanced Deep Learning (AIGC 5500) - Midterm Project: Deep Learning Optimizers

## Team Members

| Student Name | Student ID | Worked on |
|---|---|---|
| Vishal | n01676081 | Adam optimizer & NN model |
| Kevin Xavier Antony Arul Xavier | n01584909 | Data Preprocessing & AdamW optimizer |
| Komal | n01715621 | RMSprop optimizer |
| Riya Choudhary Kolli | n01485460 | AdamW optimizer |
| Paramjeet | n01692988 | RMSprop optimizer |

## Introduction

**Objective:** This research investigates and compares the performance of three popular optimization algorithms: Adam, RMSprop, and AdamW, applied to a feedforward fully connected neural network trained on the KMNIST dataset.

**Importance:**  Choosing the right optimizer significantly impacts the convergence speed, performance, and training efficiency of a neural network. Comparing these algorithms on the KMNIST dataset provides valuable insights into their suitability for various deep learning tasks.

## Description

**1. Dataset**

The KMNIST dataset is a benchmark dataset for handwritten character recognition, similar to MNIST but with Japanese characters. It contains 70,000 grayscale images (28x28 pixels) categorized into 10 classes. The dataset is split into 60,000 training images and 10,000 test images.

**2. Deep Learning Model**

A fully connected feedforward neural network is used with the following architecture:

| Layers | Architecture | Activation Function |
|---|---|---|
| Input Layer | 784 neurons (28x28 pixels) | ReLU |
| Hidden Layer [1] | 128 neurons | ReLU |
| Hidden Layer [2] | 64 neurons | ReLU |
| Output Layer | 10 neurons | Softmax |

**3. Framework**

PyTorch is used for implementation due to its flexibility in optimizer tuning and integration with sklearn for hyperparameter search.

**4. Optimizers**

* **RMSprop (Root Mean Square Prop):**  An adaptive learning rate algorithm that dynamically adjusts the learning rate for each parameter by keeping a moving average of squared gradients.
* **Adam (Adaptive Momentum Estimation):** Combines the benefits of SGD with momentum and RMSprop.
* **AdamW (Adam with Weight Decay):** A variant of Adam that incorporates weight decay as a regularization technique to prevent overfitting.

**Comparison of Parameters:**

| Parameter | RMSprop | Adam | AdamW |
|---|---|---|---|
| Learning Rate | α | α | α |
| Decay Rate | ρ | - | - |
| Momentum | β1 | β1 | β1 |
| Second Moment | - | β2 | β2 |
| Epsilon | ε | ε | ε |
| Weight Decay | - | - | λ |

**Key Differences:**

* RMSprop lacks a momentum term.
* Adam combines momentum and adaptive learning rates.
* AdamW includes weight decay.

## Methodology

**1. Training Test Split**

The KMNIST dataset is pre-split into training and test sets (9:1 ratio) with balanced classes.

**2. Standardization**

Sklearn's StandardScaler is used to standardize the training set (mean: 0, std: 1) and transform the test set accordingly.

**3. Hyperparameter Tuning**

RandomSearch with Skorch library is used to tune optimizer parameters.

**RandomSearch Grid:**

| Params | RMSProp | Adam | AdamW |
|---|---|---|---|
| Learning Rate | [1e-2, 1e-3, 1e-4] | [1e-2, 1e-3, 1e-4] | [1e-2, 1e-3, 1e-4] |
| Decay Rate | [0.9, 0.99] | - | - |
| Momentum | [0, 0.9] | (0.9, 0.8) | (0.9, 0.8) |
| Second Moment | - | (0.999, 0.999) | (0.999, 0.999) |
| Epsilon | [1e-8, 1e-7] | [1e-8, 1e-7] | [1e-8, 1e-7] |
| Weight Decay | - | - | [0, 1e-4, 1e-5] |

**Best Parameters:**

| Params | RMSProp | Adam | AdamW |
|---|---|---|---|
| Learning Rate | 1e-3 | 1e-3 | 1e-3 |
| Decay Rate | 0.99 | - | - |
| Momentum | 0 | 0.9 | 0.9 |
| Second Moment | - | 0.999 | 0.999 |
| Epsilon | 1e-8 | 1e-7 | 1e-8 |
| Weight Decay | - | - | 1e-5 |
| Best Acc. | 0.939 | 0.935 | 0.932 |

**4. K-Fold**

K-Fold cross-validation (k=5) is used for robust training.

## Results

**1. Training and Validation Loss**

(Include plots or tables of training and validation loss for each optimizer)

**2. Final Test Accuracy**

| RMSProp | Adam | AdamW |
|---|---|---|
| 88.05% | 87.45% | 89.25% |

## Interpretation

RMSprop effectively managed the learning rate, leading to good performance. Adam, with its adaptive learning rate and momentum, also showed competitive results. AdamW's inclusion of weight decay helped prevent overfitting and improved accuracy in this case.

## Conclusion

The study highlights the importance of optimizer selection. AdamW achieved the highest accuracy on the KMNIST dataset, but the best choice may vary depending on the task. Future work could explore more optimizers and hyperparameter tuning techniques.

## References

1. Brownlee, J. (2021). Gentle introduction to the adam optimization algorithm for deep learning. MachineLearningMastery.com. 
2. Brownlee, J. (2022). How to grid search hyperparameters for deep learning models in python with keras. MachineLearningMastery.com. 
3. Bonnet, A. (2024). Fine-tuning models: hyperparameter optimization. Encord. 
4. In what order should we tune hyperparameters in Neural Networks? Stack Overflow. 
5. Skorch gridsearch using sklearn wrapper How to Grid Search Hyperparameters for PyTorch Models - MachineLearningMastery.com
