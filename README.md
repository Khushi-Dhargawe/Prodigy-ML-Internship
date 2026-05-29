# 🤖 Prodigy InfoTech — Machine Learning Internship

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange?logo=scikit-learn)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![Internship](https://img.shields.io/badge/Prodigy%20InfoTech-ML%20Internship-blueviolet)

> **Organisation:** Prodigy InfoTech — Machine Learning Internship  
> **Skills:** Python · Scikit-learn · TensorFlow · OpenCV · K-Means · SVM · Linear Regression · CNN · EDA

---

## 📌 Internship Overview

This repository contains all 5 machine learning tasks completed during the **Prodigy InfoTech ML Internship**. Each task applies a different algorithm and covers the full pipeline — data loading, preprocessing, model training, evaluation, and visualisation.

The tasks collectively demonstrate proficiency across **supervised learning, unsupervised learning, computer vision, and exploratory data analysis** — spanning regression, clustering, SVM classification, deep learning, and large-scale EDA.

---

## 🗂️ Repository Structure & How Tasks Connect

```
📁 Prodigy-ML-Internship/
│
├── 📁 Task-01_House_Price_Prediction/
│   ├── PRODIGY_ML_Task01.py          ← Linear Regression model
│   ├── house_data.csv                ← KC Housing dataset
│   ├── fig1_correlation_heatmap.png  ← Feature correlations
│   ├── fig2_price_distribution.png   ← Price EDA
│   ├── fig3_actual_vs_predicted.png  ← Model performance
│   └── fig4_feature_importance.png   ← Coefficients
│
├── 📁 Task-02_Customer_Segmentation/
│   ├── PRODIGY_ML_Task02.py          ← K-Means Clustering
│   ├── Mall_Customers.csv            ← Mall customer dataset
│   ├── fig1_feature_distributions.png
│   ├── fig2_eda_gender.png
│   ├── fig3_elbow_silhouette.png     ← Optimal K selection
│   ├── fig4_customer_clusters.png    ← Cluster scatter plot
│   └── fig5_cluster_profiles.png    ← Business segment profiles
│
├── 📁 Task-03_Cats_vs_Dogs_SVM/
│   ├── PRODIGY_ML_Task03.py          ← SVM classifier
│   ├── fig1_confusion_matrix.png
│   ├── fig2_roc_curve.png
│   ├── fig3_pca_variance.png
│   └── fig4_sample_predictions.png
│
├── 📁 Task-04_Hand_Gesture_Recognition/
│   ├── PRODIGY_ML_Task04.py          ← CNN model (TensorFlow)
│   ├── fig1_sample_images.png
│   ├── fig2_training_curves.png
│   └── fig3_confusion_matrix.png
│
├── 📁 Task-05_US_Accident_EDA/
│   ├── PRODIGY_ML_Task05.py          ← EDA & hotspot analysis
│   ├── fig1_missing_data.png
│   ├── fig2_temporal_patterns.png    ← Hour/day/month/year
│   ├── fig3_severity_analysis.png
│   ├── fig4_top_states.png
│   ├── fig5_weather_visibility.png
│   ├── fig6_hotspot_map.png          ← Geographic scatter map
│   └── fig7_road_conditions.png
│
├── README.md                         ← This file
└── requirements.txt                  ← All dependencies
```

### 🔗 How All 5 Tasks Connect

```
[Task 01]              [Task 02]              [Task 03]
Linear Regression  →   K-Means Clustering →   SVM Classification
Supervised/Tabular     Unsupervised/Tabular    Supervised/Images
       ↓                      ↓                      ↓
  Regression             Clustering             Computer Vision
       │                      │                      │
       └──────────────────────┴──────────────────────┘
                              ↓
                    [Task 04]          [Task 05]
                    CNN Deep Learning   Large-Scale EDA
                    TF/Keras + GPU      Pandas + Seaborn
```

All tasks share the same Python ML ecosystem and demonstrate breadth across the core ML paradigms expected in a BA/DA/ML role.

---

## 📋 Task Summary

| # | Task | Algorithm | Dataset | Key Result |
|---|---|---|---|---|
| 01 | House Price Prediction | Linear Regression | KC Housing (21K rows) | R² = 0.60+, RMSE ~$200K |
| 02 | Customer Segmentation | K-Means (k=5) | Mall Customers (200 rows) | Silhouette Score ~0.44 |
| 03 | Cats vs Dogs Classifier | SVM + PCA | Kaggle Dogs vs Cats | Accuracy ~70%+ (64×64 input) |
| 04 | Hand Gesture Recognition | CNN (3 conv blocks) | LeapGestRecog (Kaggle) | Val Accuracy ~95%+ |
| 05 | US Accident EDA | EDA + Visualisation | US Accidents 2016–2020 (3M rows) | 7 insights identified |

---

## 🔬 Task Details

### Task 01 — House Price Prediction (Linear Regression)
**Brief:** Implement a linear regression model to predict house prices based on square footage, bedrooms, and bathrooms.

**Pipeline:**
- Loaded KC House Sales dataset (21,613 records, 21 features)
- Feature selection: sqft_living, bedrooms, bathrooms, sqft_lot, floors, sqft_above
- Standardised features, 80/20 train-test split
- Evaluated with R², RMSE, MAE, and 5-fold cross-validation
- Residual plot confirms homoscedasticity at lower price ranges

**Key finding:** `sqft_living` is the strongest predictor — a 1 std increase predicts ~$130K higher price.

---

### Task 02 — Customer Segmentation (K-Means)
**Brief:** Create a K-means clustering algorithm to group retail customers by purchase history.

**Pipeline:**
- Loaded Mall Customers dataset (200 customers, 5 features)
- Standardised Age, Annual Income, Spending Score
- Determined optimal k=5 using Elbow Method + Silhouette Score
- Profiled 5 segments with business interpretation

**Cluster Profiles:**
| Cluster | Income | Spending | Segment |
|---|---|---|---|
| 0 | Low | Low | Budget Shoppers |
| 1 | High | High | Premium Targets |
| 2 | Medium | Medium | Standard Customers |
| 3 | Low | High | Impulsive Spenders |
| 4 | High | Low | Careful Savers |

---

### Task 03 — Cats vs Dogs Image Classification (SVM)
**Brief:** Implement SVM to classify cat and dog images from the Kaggle dataset.

**Pipeline:**
- Images resized to 64×64 and flattened (12,288 features)
- StandardScaler + PCA (150 components, ~85% variance retained)
- SVM with RBF kernel (C=10, gamma='scale')
- Evaluated with accuracy, classification report, ROC-AUC, confusion matrix

**Note:** Dataset requires Kaggle download. The script detects whether the data folder exists — if not, it runs a synthetic demo automatically so the code executes end-to-end.

---

### Task 04 — Hand Gesture Recognition (CNN)
**Brief:** Develop a CNN model to identify and classify hand gestures from the LeapGestRecog dataset.

**Architecture:**
```
Input (64×64×3)
→ Conv2D(32) + BN + MaxPool + Dropout(0.25)
→ Conv2D(64) + BN + MaxPool + Dropout(0.25)
→ Conv2D(128) + BN + MaxPool + Dropout(0.4)
→ Flatten → Dense(256) + BN + Dropout(0.5)
→ Softmax Output
```
**Optimiser:** Adam (lr=0.001) | **Callbacks:** EarlyStopping, ReduceLROnPlateau  
**Run in Google Colab** (GPU recommended — ~20 min on T4)

---

### Task 05 — US Traffic Accident EDA
**Brief:** Analyse US traffic accident data to identify patterns related to road conditions, weather, and time. Visualise accident hotspots and contributing factors.

**Dataset:** 2.9M accidents across the US (2016–2020)

**Key Findings:**
1. Peak accident hours are 7–9 AM and 4–6 PM (rush hours)
2. Friday has the most accidents; Sunday the fewest
3. California accounts for the highest proportion of accidents
4. Severity 2 represents ~80% of all recorded incidents
5. 71% of accidents occur during daylight hours
6. Fair weather paradoxically sees the most accidents (by volume)
7. Low visibility (< 2 miles) is present in ~8% of accidents

---

## 🚀 How to Run

### Tasks 01, 02, 05 — Local or Colab
```bash
# Install dependencies
pip install -r requirements.txt

# Navigate to task folder
cd Task-01_House_Price_Prediction

# Run script
python PRODIGY_ML_Task01.py
```

### Task 03 — SVM Cats vs Dogs
```bash
# Download dataset from kaggle.com/c/dogs-vs-cats/data
# Place in: Task-03_Cats_vs_Dogs_SVM/train/cats/ and .../dogs/
python PRODIGY_ML_Task03.py
```

### Task 04 — CNN Hand Gesture (Google Colab)
```
1. Open PRODIGY_ML_Task04.py in Google Colab
2. Uncomment the Drive mount + Kaggle download cells
3. Runtime → Change Runtime Type → GPU
4. Run All
```

---

## 📦 Requirements

See `requirements.txt` for full list. Core dependencies:

```
pandas · numpy · scikit-learn · matplotlib · seaborn
tensorflow · opencv-python · Pillow
```

---

## 📁 Related Projects

| # | Project | Skills |
|---|---|---|
| 8 | [Signature Detection & Verification](../Signature-Detection-Verification) | PyTorch · CNN · Cosine Similarity |
| 9 | [Healthify — Multi-Disease Prediction](../Healthify-Multi-Disease-Prediction) | Streamlit · ResNet50 · PyTorch |
| **10** | **Prodigy ML Internship ← You are here** | **Python · SVM · CNN · K-Means · Linear Regression** |
| 4 | [Customer Churn Prediction](../Customer-Churn-Prediction) | XGBoost · SHAP · LIME |
| 5 | [Global Supply Chain Optimisation](../Global-Supply-Chain-Optimisation) | MILP · Monte Carlo · NetworkX |

---

## 👩‍💻 Author

**Khushi Dhargawe**  
MSc Business Analytics — University College Cork (UCC)  
BE Artificial Intelligence & Machine Learning (Hons. Cybersecurity) — Mumbai University

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://linkedin.com/in/khushi-dhargawe)
[![GitHub](https://img.shields.io/badge/GitHub-Portfolio-black?logo=github)](https://github.com/Khushi-Dhargawe)

---

## 📜 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
