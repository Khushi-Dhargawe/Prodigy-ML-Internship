# ============================================================
# PRODIGY INFOTECH — MACHINE LEARNING INTERNSHIP
# Task 02: Customer Segmentation using K-Means Clustering
# Dataset: https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python
# Tools: Python, Scikit-learn, Pandas, Matplotlib, Seaborn
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# 1. Load Dataset 
df = pd.read_csv('Mall_Customers.csv')
print("=" * 60)
print("TASK 02 — CUSTOMER SEGMENTATION (K-MEANS CLUSTERING)")
print("=" * 60)
print(f"\nDataset shape: {df.shape}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nMissing values: {df.isnull().sum().sum()}")
print(f"\nStatistical summary:\n{df.describe()}")

# 2. EDA Visualisations 
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, col, color in zip(axes, ['Age', 'Annual Income (k$)', 'Spending Score (1-100)'],
                           ['steelblue', 'seagreen', 'coral']):
    ax.hist(df[col], bins=20, color=color, edgecolor='white', alpha=0.85)
    ax.set_title(f'{col} Distribution', fontweight='bold')
    ax.set_xlabel(col)
    ax.set_ylabel('Frequency')
    ax.axvline(df[col].mean(), color='black', linestyle='--', lw=1.5, label=f'Mean: {df[col].mean():.1f}')
    ax.legend(fontsize=8)

plt.suptitle('Feature Distributions — Mall Customers', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('fig1_feature_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: fig1_feature_distributions.png")

# Gender breakdown
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
gender_counts = df['Genre'].value_counts() if 'Genre' in df.columns else df['Gender'].value_counts()
gender_col = 'Genre' if 'Genre' in df.columns else 'Gender'
axes[0].pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%',
            colors=['steelblue', 'coral'], startangle=90)
axes[0].set_title('Gender Distribution', fontweight='bold')

axes[1].scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'],
                c=df[gender_col].map({gender_counts.index[0]: 'steelblue',
                                      gender_counts.index[1]: 'coral'}),
                alpha=0.7, s=60, edgecolors='white')
axes[1].set_xlabel('Annual Income (k$)', fontweight='bold')
axes[1].set_ylabel('Spending Score (1-100)', fontweight='bold')
axes[1].set_title('Income vs Spending by Gender', fontweight='bold')
axes[1].grid(alpha=0.3)

plt.suptitle('Exploratory Data Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('fig2_eda_gender.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig2_eda_gender.png")

# 3. Feature Scaling 
features = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
scaler = StandardScaler()
scaled = scaler.fit_transform(features)

# 4. Elbow + Silhouette 
wcss, sil_scores = [], []
k_range = range(2, 11)
for k in k_range:
    km = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=42)
    km.fit(scaled)
    wcss.append(km.inertia_)
    sil_scores.append(silhouette_score(scaled, km.labels_))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(k_range, wcss, marker='o', color='steelblue', lw=2, markersize=8)
axes[0].axvline(x=5, color='red', linestyle='--', lw=1.5, label='Optimal k=5')
axes[0].set_title('Elbow Method — Optimal K', fontweight='bold')
axes[0].set_xlabel('Number of Clusters')
axes[0].set_ylabel('WCSS (Inertia)')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(k_range, sil_scores, marker='s', color='seagreen', lw=2, markersize=8)
axes[1].axvline(x=5, color='red', linestyle='--', lw=1.5, label='Optimal k=5')
axes[1].set_title('Silhouette Score vs K', fontweight='bold')
axes[1].set_xlabel('Number of Clusters')
axes[1].set_ylabel('Silhouette Score')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.suptitle('Optimal Cluster Selection', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('fig3_elbow_silhouette.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig3_elbow_silhouette.png")

# 5. Final Clustering (k=5) 
optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', max_iter=300, n_init=10, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled)

sil = silhouette_score(scaled, df['Cluster'])
print(f"\n[ CLUSTERING RESULTS ]")
print(f"Optimal K         : {optimal_k}")
print(f"Silhouette Score  : {sil:.4f}")
print(f"Inertia (WCSS)    : {kmeans.inertia_:.2f}")

# 6. Cluster Scatter 
palette = ['#e74c3c','#3498db','#2ecc71','#f39c12','#9b59b6']
plt.figure(figsize=(10, 7))
for i in range(optimal_k):
    mask = df['Cluster'] == i
    plt.scatter(df[mask]['Annual Income (k$)'],
                df[mask]['Spending Score (1-100)'],
                c=palette[i], label=f'Cluster {i}', s=80, alpha=0.8, edgecolors='white')

centers = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(centers[:, 1], centers[:, 2], c='black', marker='X',
            s=200, zorder=5, label='Centroids')
plt.xlabel('Annual Income (k$)', fontweight='bold')
plt.ylabel('Spending Score (1-100)', fontweight='bold')
plt.title(f'Customer Segments — K-Means (k={optimal_k})\nSilhouette Score: {sil:.4f}',
          fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('fig4_customer_clusters.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig4_customer_clusters.png")

# 7. Cluster Profiles 
profile = df.groupby('Cluster')[['Age','Annual Income (k$)','Spending Score (1-100)']].mean().round(1)
profile['Count'] = df['Cluster'].value_counts().sort_index()
profile['Segment'] = ['Budget Shoppers','High Value','Standard','Young Spenders','Careful Savers'][:optimal_k]
print(f"\n[ CLUSTER PROFILES ]\n{profile.to_string()}")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
metrics = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
for ax, metric, color in zip(axes, metrics, ['steelblue','seagreen','coral']):
    bars = ax.bar(profile.index, profile[metric], color=color, edgecolor='white', alpha=0.85)
    ax.set_title(f'Avg {metric} by Cluster', fontweight='bold')
    ax.set_xlabel('Cluster')
    ax.set_ylabel(metric)
    for bar, val in zip(bars, profile[metric]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.suptitle('Cluster Profile Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('fig5_cluster_profiles.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig5_cluster_profiles.png")

print("\n[TASK 02 COMPLETE] All 5 charts saved.")
print("\nBusiness Insights:")
print("- Cluster with High Income + High Spending = Premium target for loyalty programmes")
print("- Cluster with Low Income + High Spending = Risk of debt — credit monitoring needed")
print("- Cluster with High Income + Low Spending = Untapped potential — targeted promotions")
