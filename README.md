# Brain Network Analysis with GNNs and Baseline Models

This project explores how functional brain connectivity networks can be used to classify demographic and cognitive traits, and compares the performance of **classical machine learning models** with **Graph Neural Networks (GNNs)** under different dataset scales.

Brains are modeled as graphs where nodes represent regions of interest (ROIs) and edges represent functional connectivity derived from fMRI data.

---

## Project Motivation

Classical machine learning methods often treat brain connectivity matrices as high-dimensional vectors, ignoring underlying graph structure. Graph Neural Networks, on the other hand, explicitly model connectivity topology.

This project investigates:

- When do GNNs outperform classical ML?
- How does dataset size affect model choice?
- What trade-offs exist between simplicity, interpretability, and performance?

---

## Datasets

### 1. CMU Cognitive Creativity Dataset
- ~114 subjects
- 70 × 70 functional connectivity matrices
- Labels:
  - Gender
  - Subject type (Normal / High-math / Creative)
- Small, high-dimensional, class-imbalanced dataset
<img width="1130" height="378" alt="image" src="https://github.com/user-attachments/assets/119b702f-7fbf-4200-8e60-9fd375f186a4" />
<img width="1218" height="274" alt="image" src="https://github.com/user-attachments/assets/07bbb77f-5b70-44cf-b399-552e15e3f49d" />


### 2. NeuroGraph HCP Gender Dataset
- ~1200 subjects
- Graphs constructed from fMRI using NeuroGraph
- Gender classification task
- Large-scale graph learning setting

---

## Tasks

- Gender classification
- Subject type classification
  - Binary and multi-class variants
- Cross-dataset comparison of modeling strategies

---

## Methodology

### Baseline Models (Classical ML)

Connectivity matrices are vectorized by extracting upper-triangular values and passed through standard preprocessing.

**Models:**
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM, RBF kernel)
- Random Forest
- Multi-Layer Perceptron (MLP)
- XGBoost

**Preprocessing:**
- Feature standardization
- PCA for dimensionality reduction
- SMOTE for class imbalance
- 5-fold cross-validation

---

### Graph Neural Networks

Instead of flattening features, GNNs operate directly on graph structure.

**Approaches:**
- Graph Convolutional Network (GCN) for CMU dataset
- NeuroGraph framework for HCP dataset

**Details:**
- Nodes represent ROIs
- Edge weights represent functional connectivity
- Global pooling for graph-level prediction
- Cross-entropy loss with class weighting

---

## Key Results
<img width="600" height="400" alt="image" src="https://github.com/user-attachments/assets/e5c51ecc-a709-4e43-ba50-57084aafceaa" />
<img width="600" height="400" alt="image" src="https://github.com/user-attachments/assets/5275bbf3-40d6-48b7-850f-cba4571ca11c" />



- Classical ML models (especially SVMs) outperform GNNs on small datasets
- GNNs underperform in low-sample, high-dimensional regimes due to overfitting
- On large datasets, GNNs significantly outperform classical baselines
- Dataset scale is the dominant factor in determining model effectiveness

## Authors

This project was completed as part of a **team of four**.

- **Adwaita Basu Bal**
- **Ishani Ashok Kumar**
- **Luka Micevic**
- **Baaz Jhaj**
