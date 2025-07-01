# epoxy_ml_pipeline.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import cross_val_predict, KFold
from mpl_toolkits.mplot3d import Axes3D

# === STEP 1: Load Data ===
full_df = pd.read_csv("combinations_dataset.csv")  # Make sure this file exists in your repo

# === STEP 2: Prepare Initial Training Set (Simulated) ===
np.random.seed(42)
initial_sample = full_df.sample(n=32, random_state=42).reset_index(drop=True)
initial_sample['lg|Z|0.01Hz'] = np.random.uniform(4.5, 11.0, size=32)

# Simulate top-5 candidates from first cycle
cycle1_indices = [94, 158, 223, 238, 222]
cycle1_samples = full_df.iloc[cycle1_indices].copy()
cycle1_samples['lg|Z|0.01Hz'] = np.random.uniform(9.3, 11.0, size=5)

# Combine training data
training_df = pd.concat([initial_sample, cycle1_samples], ignore_index=True)

# === STEP 3: Train Model ===
X = training_df.drop(columns=['lg|Z|0.01Hz'])
y = training_df['lg|Z|0.01Hz']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_scaled, y)

# === STEP 4: Predict on All 256 Combinations ===
X_all_scaled = scaler.transform(full_df)
full_df['Predicted log|Z|'] = rf_model.predict(X_all_scaled)

# === STEP 5: Visualizations ===

# 1. Histogram
plt.figure(figsize=(8, 5))
sns.histplot(full_df['Predicted log|Z|'], bins=15, kde=True)
plt.title("Distribution of Predicted log|Z|₀.₀₁Hz")
plt.xlabel("Predicted log|Z|")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("hist_predicted_logZ.png", dpi=300)
plt.close()

# 2. 2D Scatter Plot
plt.figure(figsize=(8, 5))
sns.scatterplot(data=full_df, x='r', y='Predicted log|Z|', hue='MgO/ZIF-8 content (wt.%)', palette='viridis')
plt.title("r vs Predicted log|Z| Colored by MgO Content")
plt.xlabel("Epoxy/Hardener ratio (r)")
plt.ylabel("Predicted log|Z|")
plt.tight_layout()
plt.savefig("scatter_r_vs_logZ.png", dpi=300)
plt.close()

# 3. Pairplot
sns.pairplot(full_df[['MWc (g·mol⁻¹)', 'r', 'UPy-D400 content (mol%)', 'MgO/ZIF-8 content (wt.%)', 'Predicted log|Z|']],
             hue='Predicted log|Z|', palette='coolwarm', corner=True)
plt.savefig("pairplot_features_vs_logZ.png", dpi=300)
plt.close()

# 4. 3D Plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
x = full_df['MWc (g·mol⁻¹)']
y = full_df['r']
z = full_df['Predicted log|Z|']
c = full_df['UPy-D400 content (mol%)']

sc = ax.scatter(x, y, z, c=c, cmap='viridis', s=60, edgecolor='k')
ax.set_xlabel("MWc (g/mol)")
ax.set_ylabel("r (Epoxy/Hardener)")
ax.set_zlabel("Predicted log|Z|₀.₀₁Hz")
ax.set_title("3D Scatter: MWc vs r vs Predicted log|Z|")

cbar = fig.colorbar(sc, pad=0.1)
cbar.set_label("UPy Content (%)")
plt.tight_layout()
plt.savefig("3dplot_MWc_r_logZ.png", dpi=300)
plt.close()

# === STEP 6: Export Top Candidates ===
top_10 = full_df.sort_values(by="Predicted log|Z|", ascending=False).head(10)
top_10.to_csv("top_10_formulations.csv", index=False)
