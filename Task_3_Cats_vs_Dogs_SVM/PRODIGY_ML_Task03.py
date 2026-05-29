# ============================================================
# PRODIGY INFOTECH — MACHINE LEARNING INTERNSHIP
# Task 03: Cats vs Dogs Image Classification using SVM
# Dataset: https://www.kaggle.com/c/dogs-vs-cats/data
# Tools: Python, Scikit-learn, OpenCV, Matplotlib
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
import zipfile
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, roc_curve, auc)
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("TASK 03 — CATS vs DOGS CLASSIFICATION (SVM)")
print("=" * 60)

# 1. Dataset Setup 
# Expected folder structure after downloading from Kaggle:
# train/
#   cats/cat.0.jpg ... cat.999.jpg
#   dogs/dog.0.jpg ... dog.999.jpg
# If running in Colab: upload the zip, unzip, and update paths below

IMAGE_SIZE = (64, 64)
MAX_PER_CLASS = 500  # Use 500 per class for speed — scale up for production

def load_images_from_folder(folder, label, max_images=MAX_PER_CLASS):
    images, labels = [], []
    files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg','.jpeg','.png'))][:max_images]
    for fname in files:
        path = os.path.join(folder, fname)
        img = cv2.imread(path)
        if img is not None:
            img = cv2.resize(img, IMAGE_SIZE)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
            labels.append(label)
    return images, labels

# Detect dataset location 
possible_train_dirs = ['train', 'data/train', '/content/train', 'dataset/train']
train_dir = None
for d in possible_train_dirs:
    if os.path.exists(d):
        train_dir = d
        break

if train_dir is None:
    print("\n[INFO] Dataset not found locally — generating synthetic demo data.")
    print("To use real data: download from kaggle.com/c/dogs-vs-cats/data")
    print("Place in: train/cats/ and train/dogs/\n")

    # Synthetic demo — random images with realistic pixel patterns
    np.random.seed(42)
    n_samples = 200
    X_cats = np.random.randn(n_samples, IMAGE_SIZE[0] * IMAGE_SIZE[1] * 3) * 40 + 100
    X_dogs = np.random.randn(n_samples, IMAGE_SIZE[0] * IMAGE_SIZE[1] * 3) * 50 + 120
    X = np.vstack([X_cats, X_dogs]).astype(np.uint8)
    y = np.array([0]*n_samples + [1]*n_samples)
    print(f"Synthetic demo dataset: {X.shape[0]} samples, {X.shape[1]} features")
else:
    cats_dir = os.path.join(train_dir, 'cats')
    dogs_dir = os.path.join(train_dir, 'dogs')
    cat_imgs, cat_labels = load_images_from_folder(cats_dir, 0)
    dog_imgs, dog_labels = load_images_from_folder(dogs_dir, 1)
    all_images = np.array(cat_imgs + dog_imgs)
    y = np.array(cat_labels + dog_labels)
    X = all_images.reshape(len(all_images), -1)
    print(f"Loaded {len(cat_imgs)} cats + {len(dog_imgs)} dogs")

# 2. Train/Test Split 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# 3. Preprocessing — Scale + PCA 
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

pca = PCA(n_components=150, random_state=42)
X_train_pca = pca.fit_transform(X_train_sc)
X_test_pca  = pca.transform(X_test_sc)
print(f"PCA: {X_train_sc.shape[1]} -> {X_train_pca.shape[1]} components")
print(f"Variance explained: {pca.explained_variance_ratio_.sum():.2%}")

# 4. Train SVM 
print("\nTraining SVM (RBF kernel)...")
svm = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
svm.fit(X_train_pca, y_train)

# 5. Evaluation 
y_pred = svm.predict(X_test_pca)
y_proba = svm.predict_proba(X_test_pca)[:, 1]

acc = accuracy_score(y_test, y_pred)
print("\n" + "=" * 60)
print("MODEL EVALUATION RESULTS")
print("=" * 60)
print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
print(f"\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=['Cat', 'Dog']))

# 6. Confusion Matrix 
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Cat', 'Dog'], yticklabels=['Cat', 'Dog'],
            linewidths=0.5, cbar_kws={'shrink': 0.8})
plt.title(f'Confusion Matrix — SVM\nAccuracy: {acc:.4f}', fontweight='bold')
plt.ylabel('Actual', fontweight='bold')
plt.xlabel('Predicted', fontweight='bold')
plt.tight_layout()
plt.savefig('fig1_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig1_confusion_matrix.png")

# 7. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='steelblue', lw=2, label=f'SVM ROC (AUC = {roc_auc:.4f})')
plt.plot([0,1],[0,1], color='gray', linestyle='--', lw=1.5, label='Random Classifier')
plt.fill_between(fpr, tpr, alpha=0.1, color='steelblue')
plt.xlabel('False Positive Rate', fontweight='bold')
plt.ylabel('True Positive Rate', fontweight='bold')
plt.title('ROC Curve — SVM Cat vs Dog Classifier', fontweight='bold')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('fig2_roc_curve.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig2_roc_curve.png")

# 8. PCA Variance Plot 
plt.figure(figsize=(9, 5))
cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
plt.plot(range(1, len(cumvar)+1), cumvar, color='seagreen', lw=2)
plt.axhline(y=90, color='red', linestyle='--', lw=1.5, label='90% variance')
plt.xlabel('Number of PCA Components', fontweight='bold')
plt.ylabel('Cumulative Variance Explained (%)', fontweight='bold')
plt.title('PCA — Cumulative Variance Explained', fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('fig3_pca_variance.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig3_pca_variance.png")

# 9. Sample Predictions 
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
class_names = ['Cat', 'Dog']
for i, ax in enumerate(axes.flatten()):
    idx = np.random.randint(0, len(X_test))
    img = X_test[idx].reshape(IMAGE_SIZE[0], IMAGE_SIZE[1], 3).clip(0, 255).astype(np.uint8)
    actual = class_names[y_test[idx]]
    predicted = class_names[y_pred[idx]]
    color = 'green' if actual == predicted else 'red'
    ax.imshow(img)
    ax.set_title(f'A: {actual}\nP: {predicted}', color=color, fontsize=9, fontweight='bold')
    ax.axis('off')

plt.suptitle('Sample Predictions (Green=Correct, Red=Wrong)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('fig4_sample_predictions.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig4_sample_predictions.png")

print(f"\n[TASK 03 COMPLETE] Accuracy: {acc*100:.2f}% | AUC: {roc_auc:.4f}")
