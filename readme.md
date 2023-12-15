# COL 761 - Data Mining Assignments

## Overview
This repository contains the solutions for three assignments completed as part of the course COL 761 - Data Mining, under the guidance of Prof. Sayan Ranu.




### Assignment 1: FP-Tree Based Data Compression
- Developed a data compression algorithm using FP-tree.
- Structured approach involving transaction sorting, block division, FP-growth mining, and pattern replacement.
- Compressed dataset with human-readable output and mapping.

### Assignment 2: Subgraph Mining and Clustering
#### Task 1: Subgraph Mining
- Used three subgraph mining algorithms: FSG, gSpan, and Gaston.
- Executed on a yeast dataset (167.txt_graph) using HPC (High-Performance Computing).
- Plotted runtime vs support curves for comparative analysis.

#### Task 2: Clustering
- Conducted k-means clustering on datasets using Python.
- Generated an elbow plot for optimal cluster determination.
- Used a dataset generator (`generateDataset_d_dim_hpc_compiled`) for dataset creation on HPC.

### Assignment 3: Graph Property Prediction
#### Task 1: Classification
- Implemented a Graph Neural Network (GNN) for binary graph classification.
- Utilized PyTorch Geometric and GAT (Graph Attention Network) for model development.
- Evaluated using ROC-AUC as the performance metric.

#### Task 2: Regression
- Developed a GNN for predicting continuous numerical values associated with graphs.
- Leveraged PyTorch Geometric and GAT for model architecture.
- Evaluated using Root Mean Squared Error (RMSE) as the performance metric.

## Running the Assignments
- Detailed instructions for each assignment are provided in their respective directories.
- Ensure necessary dependencies are installed before running the scripts.
- For HPC execution, follow specific instructions provided in each assignment.

## Course Information
- **Course Name:** COL 761 - Data Mining
- **Instructor:** Prof. Sayan Ranu
