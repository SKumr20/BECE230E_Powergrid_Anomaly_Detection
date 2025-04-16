# BECE230E - Embedded C Programming - Winter-sem 2024-25 Project
# 22BEE1311 Satyam Kumar
# Data - Tamil Nadu Electricity Board Hourly Readings

# Import required libraries
import os
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
# Get the absolute path of the project directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Construct the correct data file path
data_path = os.path.join(BASE_DIR, "data", "csv_result-eb.csv")

# Load the dataset
df = pd.read_csv(data_path)

# Display basic info about the dataset
print(df.info())
print(df.head())

# Convert ForkVA to numeric (handling errors)
df["ForkVA"] = pd.to_numeric(df["ForkVA"], errors="coerce")

# Drop rows with missing ForkVA or ForkW values
df.dropna(subset=["ForkVA", "ForkW"], inplace=True)

# ==========================
# ANOMALY DETECTION METHODS
# ==========================

# ---------- Z-Score Method ----------
# Compute Z-scores for ForkVA and ForkW
z_scores = np.abs(stats.zscore(df[["ForkVA", "ForkW"]]))

# Mark as anomaly if Z-score > 3
df["Z_Anomaly"] = (z_scores > 3).any(axis=1)

# ---------- IQR Method ----------
# Compute Q1, Q3, and IQR
Q1 = df[["ForkVA", "ForkW"]].quantile(0.25)
Q3 = df[["ForkVA", "ForkW"]].quantile(0.75)
IQR = Q3 - Q1

# Mark anomalies if values are beyond 1.5 * IQR
df["IQR_Anomaly"] = ((df[["ForkVA", "ForkW"]] < (Q1 - 1.5 * IQR)) | 
                      (df[["ForkVA", "ForkW"]] > (Q3 + 1.5 * IQR))).any(axis=1)

# Count anomalies
print("Z-Score Anomalies:", df["Z_Anomaly"].sum())
print("IQR Anomalies:", df["IQR_Anomaly"].sum())

# ---------- Isolation Forest ----------
# Initialize Isolation Forest model
iso_forest = IsolationForest(contamination=0.01, random_state=42)

# Fit model and detect anomalies
df["IsoForest_Anomaly"] = iso_forest.fit_predict(df[["ForkVA", "ForkW"]]) == -1

# ---------- Local Outlier Factor (LOF) ----------
# Initialize Local Outlier Factor model
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.01)

# Fit model and detect anomalies
df["LOF_Anomaly"] = lof.fit_predict(df[["ForkVA", "ForkW"]]) == -1

# Count anomalies detected
print("Isolation Forest Anomalies:", df["IsoForest_Anomaly"].sum())
print("Local Outlier Factor Anomalies:", df["LOF_Anomaly"].sum())


# GRAPHS

# ---------- 1. Scatter Plot: Isolation Forest Anomalies ----------
plt.figure(figsize=(10, 6))

sns.scatterplot(x=df["ForkVA"], y=df["ForkW"], label="Normal", color="blue")
sns.scatterplot(x=df[df["IsoForest_Anomaly"]]["ForkVA"], 
                y=df[df["IsoForest_Anomaly"]]["ForkW"], 
                label="Anomalies", color="red")

plt.xlabel("VA (Apparent Power)")
plt.ylabel("kW (Real Power)")
plt.title("Fault Detection Using Isolation Forest (Scatter Plot)")
plt.legend()
plt.show(block=False)

# ---------- 2. Box Plot for Outliers ----------
plt.figure(figsize=(12, 5))

# Box plot for ForkVA
plt.subplot(1, 2, 1)
sns.boxplot(y=df["ForkVA"])
plt.title("ForkVA (Apparent Power) Distribution")

# Box plot for ForkW
plt.subplot(1, 2, 2)
sns.boxplot(y=df["ForkW"])
plt.title("ForkW (Real Power) Distribution")

plt.show()

# ---------- 3. Overlap Visualization ----------
# Classify anomalies
df["Anomaly_Label"] = "Normal"
df.loc[df["IsoForest_Anomaly"] & ~df["LOF_Anomaly"], "Anomaly_Label"] = "Isolation Forest Only"
df.loc[~df["IsoForest_Anomaly"] & df["LOF_Anomaly"], "Anomaly_Label"] = "LOF Only"
df.loc[df["IsoForest_Anomaly"] & df["LOF_Anomaly"], "Anomaly_Label"] = "Both"

# Set colors for categories
palette = {
    "Normal": "lightgrey",
    "Isolation Forest Only": "red",
    "LOF Only": "blue",
    "Both": "purple"
}

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="ForkVA", y="ForkW", hue="Anomaly_Label", palette=palette)

plt.title("Anomaly Detection Overlap: Isolation Forest vs LOF")
plt.xlabel("VA (Apparent Power)")
plt.ylabel("kW (Real Power)")
plt.legend()
plt.show()
