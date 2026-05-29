# ============================================================
# PRODIGY INFOTECH — MACHINE LEARNING INTERNSHIP
# Task 01: House Price Prediction using Linear Regression
# Dataset: https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data
# Tools: Python, Scikit-learn, Pandas, Matplotlib, Seaborn
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 1. Load Dataset 
df = pd.read_csv('house_data.csv')
print("=" * 60)
print("TASK 01 — HOUSE PRICE PREDICTION (LINEAR REGRESSION)")
print("=" * 60)
print(f"\nDataset shape: {df.shape}")
print(f"Columns: {list(df.columns[:10])}...")
print(f"\nFirst 5 rows:\n{df.head()}")

# 2. EDA 
print("\n[ DATA QUALITY CHECK ]")
print(f"Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
print(f"\nPrice stats:\n{df['price'].describe()}")

# 3. Feature Selection 
features = ['sqft_living', 'bedrooms', 'bathrooms', 'sqft_lot',
            'floors', 'sqft_above', 'sqft_basement']
# Use only columns that exist
features = [f for f in features if f in df.columns]
target = 'price'

X = df[features]
y = df[target]

print(f"\nFeatures used: {features}")
print(f"Target: {target}")

# 4. Correlation Heatmap 
plt.figure(figsize=(10, 7))
corr_data = df[features + [target]].corr()
mask = np.triu(np.ones_like(corr_data, dtype=bool))
sns.heatmap(corr_data, annot=True, fmt='.2f', cmap='coolwarm',
            mask=mask, linewidths=0.5,
            cbar_kws={'shrink': 0.8})
plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('fig1_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: fig1_correlation_heatmap.png")

# 5. Price Distribution 
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(y, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
axes[0].set_title('House Price Distribution', fontweight='bold')
axes[0].set_xlabel('Price ($)')
axes[0].set_ylabel('Frequency')

axes[1].hist(np.log1p(y), bins=50, color='seagreen', edgecolor='white', alpha=0.8)
axes[1].set_title('Log-Transformed Price Distribution', fontweight='bold')
axes[1].set_xlabel('Log(Price)')
axes[1].set_ylabel('Frequency')

plt.suptitle('Price Distribution Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('fig2_price_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig2_price_distribution.png")

# 6. Train/Test Split 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

print(f"\nTrain size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

# 7. Train Model 
model = LinearRegression()
model.fit(X_train_sc, y_train)
predictions = model.predict(X_test_sc)

# 8. Evaluation Metrics 
mse  = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(y_test, predictions)
r2   = r2_score(y_test, predictions)

cv_r2 = cross_val_score(model, scaler.fit_transform(X), y,
                        cv=5, scoring='r2').mean()

print("\n" + "=" * 60)
print("MODEL EVALUATION RESULTS")
print("=" * 60)
print(f"R-Squared (R2)          : {r2:.4f}")
print(f"Cross-Val R2 (5-fold)   : {cv_r2:.4f}")
print(f"Root Mean Squared Error : ${rmse:,.0f}")
print(f"Mean Absolute Error     : ${mae:,.0f}")
print(f"Mean Squared Error      : ${mse:,.0f}")
print("\nModel Coefficients:")
for feat, coef in zip(features, model.coef_):
    print(f"  {feat:20s}: {coef:+.2f}")
print(f"  {'Intercept':20s}: {model.intercept_:+.2f}")

# 9. Actual vs Predicted 
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

axes[0].scatter(y_test, predictions, alpha=0.4, color='steelblue', edgecolors='none', s=20)
min_val, max_val = min(y_test.min(), predictions.min()), max(y_test.max(), predictions.max())
axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
axes[0].set_xlabel('Actual Price ($)', fontweight='bold')
axes[0].set_ylabel('Predicted Price ($)', fontweight='bold')
axes[0].set_title(f'Actual vs Predicted Prices\nR² = {r2:.4f}', fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

residuals = y_test - predictions
axes[1].scatter(predictions, residuals, alpha=0.4, color='coral', edgecolors='none', s=20)
axes[1].axhline(y=0, color='black', linestyle='--', lw=2)
axes[1].set_xlabel('Predicted Price ($)', fontweight='bold')
axes[1].set_ylabel('Residuals ($)', fontweight='bold')
axes[1].set_title('Residual Plot', fontweight='bold')
axes[1].grid(alpha=0.3)

plt.suptitle('Model Performance Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('fig3_actual_vs_predicted.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig3_actual_vs_predicted.png")

# 10. Feature Importance 
coef_df = pd.DataFrame({
    'Feature': features,
    'Coefficient': model.coef_
}).sort_values('Coefficient', ascending=True)

colors = ['coral' if c < 0 else 'steelblue' for c in coef_df['Coefficient']]
plt.figure(figsize=(9, 5))
bars = plt.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors, edgecolor='white')
plt.axvline(x=0, color='black', linestyle='-', lw=0.8)
plt.title('Feature Coefficients (Standardised)', fontsize=13, fontweight='bold')
plt.xlabel('Coefficient Value')
for bar, val in zip(bars, coef_df['Coefficient']):
    plt.text(val + (0.01 * max(abs(coef_df['Coefficient']))),
             bar.get_y() + bar.get_height()/2,
             f'{val:.0f}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig('fig4_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig4_feature_importance.png")

# 11. Predict on New Data 
print("\n[ SAMPLE PREDICTION ]")
new_home = pd.DataFrame({
    'sqft_living': [2000], 'bedrooms': [3], 'bathrooms': [2],
    'sqft_lot': [5000], 'floors': [1], 'sqft_above': [2000], 'sqft_basement': [0]
})
new_home = new_home[features]
new_home_sc = scaler.transform(new_home)
pred_price = model.predict(new_home_sc)[0]
print(f"  Input:  {dict(zip(features, new_home.values[0]))}")
print(f"  Predicted Price: ${pred_price:,.0f}")

print("\n[TASK 01 COMPLETE] All 4 charts saved.")
