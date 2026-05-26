# Philippine Medicinal Plant Leaf Classification: Classical ML vs. CNN Benchmark

> **Course:** AI201 — Introduction to Artificial Intelligence / Machine Learning
> **Team:** [Your Name] · [Partner's Name]

---

## Overview

This project benchmarks **classical machine learning pipelines against a CNN baseline** for the task of classifying 40 species of Philippine medicinal plant leaves from image data.

The dataset contains **4,922 leaf images across 40 classes**. We compare hand-crafted feature engineering approaches (AutoML, Autoencoder + SVM, LDA + XGBoost, etc.) against a deep learning CNN reference (VGG19) to answer the question: *how far can classical ML go before CNNs pull ahead?*

---

## Results Summary

| Approach | Model | Accuracy |
|---|---|---|
| Baseline (CNN Literature) | VGG19 (De Luna et al.) | 92.67% |
| Baseline (AutoML) | AutoSklearn | 98.00% |
| Baseline (Feature Extractor) | VGG19 Features + SVM | 98.00% |
| Optimized | Autoencoder + SVM (Optuna) | 97.00% |
| Optimized | LDA + XGBoost | 96.00% |
| Optimized | Autoencoder + MLP | 95.80% |
| Optimized | Autoencoder + kNN | 95.50% |

**Key finding:** Classical ML with engineered features is highly competitive with CNN baselines on this dataset, matching or exceeding the literature CNN benchmark across all optimized configurations.

---

## Repository Structure

```
├── notebooks/
│   ├── abunda_automl.ipynb              # AutoSklearn baseline
│   ├── abunda_eda.ipynb                 # Exploratory data analysis
│   ├── abunda_etc.ipynb                 # Miscellaneous experiments
│   ├── abunda_training.ipynb            # Classical ML training runs
│   ├── bg_removal.ipynb                 # Background removal preprocessing
│   ├── cantor_official_classification.ipynb  # Full classification reports
│   ├── cantor_test_1.ipynb              # Scratch / experimental
│   ├── cantor_training.ipynb            # SVM pipeline training
│   └── cantor_training_autoencoder.ipynb     # Autoencoder training + UMAP/t-SNE
│
├── src/
│   ├── utils_autoencoder.py    # Autoencoder model definition and training helpers
│   ├── utils_classical.py      # Classical ML pipelines (SVM, XGBoost, kNN, MLP)
│   ├── utils_data.py           # Dataset loading and splitting
│   ├── utils_image_conversion.py  # Image format conversion utilities
│   ├── utils_nn.py             # Neural network utilities (VGG feature extraction)
│   ├── utils_plotting.py       # Visualization helpers (confusion matrix, t-SNE, UMAP)
│   └── utils_preprocessing.py  # Feature extraction and preprocessing pipelines
│
├── models/
│   ├── autoencoder_weights.pth          # Trained autoencoder weights
│   └── optimized_model_parameters.json # Best hyperparameters from Optuna
│
├── requirements.txt
└── .gitignore
```

---

## Setup

**Prerequisites:** Python 3.12, CUDA-capable GPU recommended (tested on GTX 1650, CUDA 11.8)

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/Philippine-Medical-Plant-Leaf-Classification.git
cd Philippine-Medical-Plant-Leaf-Classification

# 2. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Register the Jupyter kernel (for VS Code)
python -m ipykernel install --user --name=ai201_venv --display-name "Python (AI201)"
```

Then open any notebook in VS Code and select the **Python (AI201)** kernel.

---

## Dataset

The dataset is **not included** in this repository due to size. It consists of 4,922 `.webp` leaf images across 40 Philippine medicinal plant species, sourced from:

> De Luna, R. G., et al. *"Classifying Philippine Medicinal Plants Based on Their Leaves Using Deep Learning."* IEEE, 2023.

Place the dataset in a `data/` folder at the project root before running notebooks.

---

## Dependencies

Key packages (see `requirements.txt` for full list):

- `torch==2.5.1+cu118` · `torchvision`
- `scikit-learn==1.8.0`
- `xgboost==3.2.0`
- `optuna`
- `umap-learn`
- `opencv-python`
- `albumentations`
