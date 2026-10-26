# Philippine Medicinal Plant Leaf Classification
### Classical Machine Learning vs. CNN Benchmark

AI201 Mini-Project · Master of Engineering in Artificial Intelligence, University of the Philippines Diliman

**Authors:** Adriane Jone A. Abunda, Joshua Cantor

---

## Overview

This project classifies leaf images of 40 Philippine medicinal plant species using **traditional image processing, hand-crafted feature extraction, and classical machine learning**, then benchmarks the result against deep learning approaches on the same dataset.

The pipeline extracts a 371-dimensional feature vector per image (color, texture, shape, and vein features), selects the Top-N features with ANOVA F-scores, and trains an RBF-kernel SVM tuned with Grid Search.

**Key result:** the optimized SVM reaches **98.06% test accuracy**, surpassing the published VGG19 + CNN benchmark (92.67%) and falling within 0.33 percentage points of a VGG19 feature extractor + SVM baseline (98.39%), while being far more lightweight and interpretable.

## Objectives

1. Extract leaf features using traditional image processing methods and use dimensionality reduction to identify the important features.
2. Train an SVM and select the Top-N features to address the curse of dimensionality.
3. Optimize the best classical model with Grid Search and compare it to previous deep learning models.
4. Show the viability of manual feature extraction and classical methods in the era of deep learning.

## Dataset

**PhilMedic: Philippine Medicinal Plant Leaf Dataset** (openly available on Mendeley Data)

| Property | Value |
|---|---|
| Images | 4,922 |
| Species (classes) | 40 |
| Images per class | 100 to 140 (approximately uniform) |
| Capture device | 48MP Android camera, white background |
| Original resolution | 3000 × 4000 px (PNG, ~3.02 MB per image) |
| Original total size | 7.76 GB |

The dataset was introduced and benchmarked by Custodio (2023) using VGG19, which achieved 92.67% accuracy.

## Methodology

### 1. Data Preparation
- Raw images were compressed to **WebP** and resized to **500 × 500 px**, reducing the dataset from **7.76 GB to 36.6 MB** without losing feature-relevant quality.
- **Background removal** was applied to every image so that all extracted features reflect leaf-intrinsic properties rather than imaging artifacts.

### 2. Feature Extraction (371 dimensions)

| Group | Descriptors |
|---|---|
| **Color** (RGB and LAB) | Green channel ratio vs. red/blue, per-channel moments (mean, std, skewness), color histograms, average saturation, average brightness |
| **Texture** | GLCM statistics (contrast, dissimilarity, homogeneity, correlation, energy), LBP histograms, edge density, roughness |
| **Shape** | Area, perimeter, circularity (4πA/P²), eccentricity, solidity |
| **Vein** | Frangi vesselness filter-derived vein density, mean vein orientation, orientation std, branching density |

The resulting N × 371 feature matrix is exported as a CSV and used as input to the dimensionality reduction and classification stages.

### 3. Dimensionality Reduction
- t-SNE and UMAP of the raw 371-D space showed heavily overlapping classes.
- **Top-N feature selection with ANOVA F-scores** was applied. After selection, both t-SNE and UMAP showed markedly more compact and distinguishable clusters.

### 4. Model Training
- **Classifier:** SVM with RBF kernel
- **Hyperparameter search:** Grid Search jointly over the number of selected features `k`, regularization `C`, and kernel bandwidth `γ`, using 5-fold cross-validation accuracy as the criterion
- **Best configuration:** `k = 200`, `C = 100`, `γ = 0.001`, RBF kernel
- **Evaluation split:** held-out test set of 20% of the data

### 5. Evaluation
Accuracy, weighted precision / recall / F1-score, confusion matrix, and ANOVA F-score feature importance analysis.

## Results

### Effect of Feature Selection (5-fold CV accuracy)

| Top-N features (k) | CV Accuracy (%) |
|---|---|
| 10 | 65.19 |
| 25 | 92.96 |
| 50 | 95.83 |
| 75 | 97.10 |
| 100 | 96.95 |
| 150 | 97.13 |
| **200** | **97.38** |
| 250 | 97.30 |
| 300 | 96.78 |
| 350 | 96.49 |
| 371 (all) | 96.49 |

Accuracy rises sharply up to about k = 50 to 75 and shows diminishing returns beyond k = 200.

### Comparison with Deep Learning Baselines (held-out test set)

| Kind | Model | Accuracy (%) |
|---|---|---|
| Baseline | VGG19 + SVM | 98.39 |
| **Classical (this work)** | **SVM + Top-N (k = 200)** | **98.06** |
| Baseline | VGG19 + CNN (Custodio, 2023) | 92.67 |

### Classification Report (best model, 40-class test set)

| | Precision | Recall | F1-Score |
|---|---|---|---|
| Accuracy | | | 0.98 |
| Macro avg | 0.98 | 0.98 | 0.98 |
| Weighted avg | 0.98 | 0.98 | 0.98 |

### Key Findings
- **Color is the dominant discriminative modality.** The top ANOVA F-scores belong to histogram-based color descriptors from the RGB, HSV, and LAB color spaces.
- **Eccentricity** is the top-ranked shape feature. LBP texture features appear across the top 20 but at lower individual scores, while GLCM and vein features contribute least individually.
- **Most misclassifications** occur between morphologically similar species, often within the same botanical family. Species with atypical morphology (compound leaves, highly irregular margins) show lower recall.
- The narrow gap to VGG19 + SVM suggests the bottleneck is **feature representation quality**, not classifier choice.

### Figures

<!-- Update these paths to match your repo -->
| Figure | Preview |
|---|---|
| Preprocessing (raw vs. background-removed WebP) | `assets/preprocessing.png` |
| CV accuracy vs. number of features | `assets/accuracy_vs_k.png` |
| Confusion matrix | `assets/confusion_matrix.png` |
| Top 20 features by ANOVA F-score | `assets/top20_features.png` |
| t-SNE / UMAP before and after Top-N selection | `assets/embeddings.png` |

## Why Classical ML?

- **Lightweight:** SVM inference on a 200-dimensional vector needs negligible computation compared to a forward pass through a 19-layer CNN, making it suitable for mobile, embedded, and offline use.
- **Interpretable:** hand-crafted features and ANOVA F-score rankings give a transparent view of which leaf properties drive predictions.
- **Compact:** the full dataset shrinks from 7.76 GB to 36.6 MB after preprocessing.

## Tech Stack

- Python 3.12
- scikit-learn
- OpenCV
- UMAP-learn
- NumPy, Pandas, Matplotlib

## Getting Started

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the PhilMedic dataset from Mendeley Data and place it in the data/ folder

# 4. Run the notebooks / scripts in order
#    (1) preprocessing -> (2) feature extraction -> (3) dimensionality reduction -> (4) training & evaluation
```

## Project Structure

```
<your-repo>/
├── data/                 # dataset (raw and preprocessed WebP images)
├── notebooks/            # preprocessing, feature extraction, training, evaluation
├── features/             # exported 371-feature CSV
├── assets/               # figures used in this README
├── requirements.txt
└── README.md
```

## Future Work

1. Expand the dataset with more species, varied lighting and backgrounds, multiple leaf surfaces, and developmental stages to test generalization beyond controlled conditions.
2. Enrich the feature set with additional color space decompositions and finer-grained color histograms to reduce confusion among morphologically similar species. Add vein and margin descriptors and targeted augmentation for atypical classes.
3. Validate the pipeline on a mobile deployment platform for rural health practitioners and field botanists.

## References

1. E. F. Custodio, "Classifying Philippine medicinal plants based on their leaves using deep learning," *2023 IEEE World AI IoT Congress (AIIoT)*, 2023, pp. 0029-0035.
2. S. P. Mohanty, D. P. Hughes, and M. Salathé, "Using deep learning for image-based plant disease detection," *Frontiers in Plant Science*, vol. 7, p. 1419, 2016.
3. G. Saleem et al., "Automated analysis of visual leaf shape features for plant classification," *Computers and Electronics in Agriculture*, vol. 157, pp. 270-280, 2019.
4. A. Kadir et al., "Leaf classification using shape, color, and texture features," *Int. J. Computer Trends and Technology*, vol. 1, no. 3, 2013.
5. L. van der Maaten and G. Hinton, "Visualizing data using t-SNE," *JMLR*, vol. 9, 2008.
6. L. McInnes, J. Healy, and J. Melville, "UMAP: Uniform manifold approximation and projection for dimension reduction," arXiv:1802.03426, 2018.

See the full written report for the complete reference list.

## Acknowledgments

Developed as the AI201 Mini-Project at the University of the Philippines Diliman.