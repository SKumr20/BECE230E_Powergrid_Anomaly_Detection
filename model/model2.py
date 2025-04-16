import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import classification_report
import os

# Set paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(BASE_DIR, 'data', '4bus_fault_train.csv')
TEST_PATH = os.path.join(BASE_DIR, 'data', '4bus_fault_test.csv')

# Load datasets
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

# Encode fault type labels for training
label_encoder = LabelEncoder()
train_df['Fault_Type'] = label_encoder.fit_transform(train_df['Fault_Type'])

# Features and labels
X_train = train_df[['Bus', 'Voltage_Magnitude', 'Voltage_Angle', 'Current_Magnitude', 'Current_Angle']]
y_train = train_df['Fault_Type']
X_test = test_df[['Bus', 'Voltage_Magnitude', 'Voltage_Angle', 'Current_Magnitude', 'Current_Angle']]

# --- Feature Scaling ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Hyperparameter Tuning with GridSearchCV ---
# Random Forest hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Perform Grid Search
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train_scaled, y_train)

# Best model from GridSearchCV
best_rf_model = grid_search.best_estimator_

# --- Cross-validation ---
cv_scores = cross_val_score(best_rf_model, X_train_scaled, y_train, cv=5)
print(f'Average Cross-Validation Score: {cv_scores.mean():.4f}')

# --- Train the best model ---
best_rf_model.fit(X_train_scaled, y_train)

# Predict fault types on the test set
predictions = best_rf_model.predict(X_test_scaled)

# Decode predicted labels to human-readable fault names
predicted_fault_types = label_encoder.inverse_transform(predictions)

# Add predictions to the test dataframe
test_df['Predicted_Fault_Type'] = predicted_fault_types

# Save predictions to CSV
PREDICTION_OUTPUT = os.path.join(BASE_DIR, 'data', '4bus_fault_predictions.csv')
test_df.to_csv(PREDICTION_OUTPUT, index=False)

# --- Visualization: Train Distribution vs Predicted Distribution ---
plt.figure(figsize=(10, 5))

# Training fault type distribution
plt.subplot(1, 2, 1)
train_fault_counts = train_df['Fault_Type'].value_counts()
train_fault_labels = label_encoder.inverse_transform(train_fault_counts.index)
plt.bar(train_fault_labels, train_fault_counts.values, color='lightgreen')
plt.title('Train Data Fault Type Distribution')
plt.xlabel('Fault Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Predicted fault type distribution
plt.subplot(1, 2, 2)
predicted_fault_counts = test_df['Predicted_Fault_Type'].value_counts()
plt.bar(predicted_fault_counts.index, predicted_fault_counts.values, color='skyblue')
plt.title('Predicted Fault Type Distribution (Test Data)')
plt.xlabel('Fault Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# --- Classification Report ---
print("\nClassification Report on Test Data:")
print(classification_report(y_train, best_rf_model.predict(X_train_scaled), target_names=label_encoder.classes_))
