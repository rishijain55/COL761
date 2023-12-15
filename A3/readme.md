# Assignment README

## Task Overview

This assignment focuses on graph property prediction using Graph Neural Networks (GNN) for two primary tasks: classification and regression. The provided dataset includes an evaluator script for assessment and an optional node and edge feature encoder. The tasks involve predicting graph properties through GNN models, with the requirement to utilize both node and edge features.

### Task 1: Classification
- **Objective:** Predict a binary label associated with graphs using GNN.
- **Model Choice:** The choice of GNN model architecture and training paradigm is up to you.
- **Evaluation Metric:** Receiver Operating Characteristic Area Under the Curve (ROC-AUC).
- **Additional Requirements:** Utilize both node and edge features, plot training and validation learning curves, visualize graphs using NetworkX, and investigate misclassifications or poor performance.

### Task 2: Regression
- **Objective:** Predict a continuous numerical value associated with the graphs using GNN.
- **Model Choice:** You have flexibility in choosing the GNN model architecture and training approach.
- **Evaluation Metric:** Root Mean Squared Error (RMSE).
- **Additional Requirements:** Utilize both node and edge features, plot learning curves, and visualize graphs to identify areas where the model exhibits significant errors in prediction.

## Implementation Details

Both tasks have been implemented using Graph Attention Networks (GAT), a type of GNN. For more detailed information, refer to the accompanying `report.pdf`. To execute the tasks, use the provided `interface2.sh` script.

### Usage:
```bash
sh interface2.sh <C/R> <train/eval> </path/to/model> </path/to/dataset> [/path/to/val_dataset]
```
**C/R:**
- `C`: Classification
- `R`: Regression

**/path/to/model:**
- A Linux-style path to the model file.
- When `<train>` is specified, the model is saved at this path.
- When `<eval>` is specified, the model is loaded from this path.

**/path/to/dataset:**
- Path to the dataset.

**/path/to/val_dataset (Optional):**
- Path to the validation dataset (provided only when `<train>` option is provided).

## Report:
Refer to report.pdf for a detailed analysis and insights.



