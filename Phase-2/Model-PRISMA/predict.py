import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

# File paths
train_file_path = "/home/user/Documents/FYP/SPLIT_N_P_K_LOW_MEDIUM_HIGH.xlsx"
test_file_path = "/home/user/Downloads/PRISMA_DATA.xlsx"

# Load training dataset
train_df = pd.read_excel(train_file_path, sheet_name="SPLIT_N_P_K_LOW_MEDIUM_HIGH")
test_df = pd.read_excel(test_file_path)

# Ensure column names are strings
train_df.columns = train_df.columns.astype(str)
test_df.columns = test_df.columns.astype(str)

# Correct PRISMA Wavelength Mapping (171 Bands: SWIR only)
prisma_wavelengths = np.linspace(920, 2500, 171).tolist()
test_df.columns = [str(round(w)) for w in prisma_wavelengths]  # Convert to string format

# Extract spectral bands from the training dataset
train_band_names = [col for col in train_df.columns if col.replace('.', '', 1).isdigit()]

# Find common features (bands that exist in both datasets)
common_features = list(set(train_band_names) & set(test_df.columns))
if "OC" in common_features:
    common_features.remove("OC")  # Remove target variable

if len(common_features) == 0:
    raise ValueError("No matching spectral bands found! Verify dataset formats and column naming.")

print(f"Common Features Found: {len(common_features)} bands")

# Define X and Y for training
X_train = train_df[common_features]
y_train = train_df["OC"]

# Categorize OC values based on predefined thresholds
thresholds = {
    "Low": y_train.min(),
    "Medium": y_train.mean(),
    "High": y_train.max()
}

def categorize_oc(value):
    return min(thresholds, key=lambda x: abs(thresholds[x] - value))

y_train_category = y_train.apply(categorize_oc)

# Apply MinMax Scaling
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(test_df[common_features])

# Train-Test Split for Model Evaluation
X_train_split, X_val, y_train_split, y_val = train_test_split(X_train_scaled, y_train, test_size=0.2, random_state=42)

# Train XGBoost Model
xgb_model = XGBRegressor(n_estimators=500, max_depth=15, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42)
xgb_model.fit(X_train_split, y_train_split)

# Train Random Forest Model
rf_model = RandomForestRegressor(n_estimators=500, max_depth=20, random_state=42)
rf_model.fit(X_train_split, y_train_split)

# Predict on Validation Set
y_val_pred_xgb = xgb_model.predict(X_val)
y_val_pred_rf = rf_model.predict(X_val)

# Predict on PRISMA Data
test_y_pred_xgb = xgb_model.predict(X_test_scaled)
test_y_pred_rf = rf_model.predict(X_test_scaled)

test_predicted_categories_xgb = [categorize_oc(oc) for oc in test_y_pred_xgb]
test_predicted_categories_rf = [categorize_oc(oc) for oc in test_y_pred_rf]

# Evaluate Model Performance
r2_xgb = r2_score(y_val, y_val_pred_xgb)
mse_xgb = mean_squared_error(y_val, y_val_pred_xgb)

r2_rf = r2_score(y_val, y_val_pred_rf)
mse_rf = mean_squared_error(y_val, y_val_pred_rf)

print(f"\nXGBoost Performance:\nR² Score: {r2_xgb:.4f}\nMSE: {mse_xgb:.4f}")
print(f"\nRandom Forest Performance:\nR² Score: {r2_rf:.4f}\nMSE: {mse_rf:.4f}")

# Save Predictions
test_predictions = pd.DataFrame({
    "Predicted_OC_XGBoost": test_y_pred_xgb,
    "Predicted_Category_XGBoost": test_predicted_categories_xgb,
    "Predicted_OC_RandomForest": test_y_pred_rf,
    "Predicted_Category_RandomForest": test_predicted_categories_rf
})
test_predictions.to_excel("Predicted_OC_PRISMA.xlsx", index=False)
print("Predictions saved as Predicted_OC_PRISMA.xlsx")

# Plot Actual vs Predicted
plt.figure(figsize=(10, 5))
plt.scatter(y_val, y_val_pred_xgb, alpha=0.6, label="XGBoost Predictions")
plt.plot([min(y_val), max(y_val)], [min(y_val), max(y_val)], 'r--', label="Perfect Prediction Line")
plt.xlabel("Actual OC Values")
plt.ylabel("Predicted OC Values")
plt.title("XGBoost: Predicted vs Actual Organic Content")
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.scatter(y_val, y_val_pred_rf, alpha=0.6, label="Random Forest Predictions", color="green")
plt.plot([min(y_val), max(y_val)], [min(y_val), max(y_val)], 'r--', label="Perfect Prediction Line")
plt.xlabel("Actual OC Values")
plt.ylabel("Predicted OC Values")
plt.title("Random Forest: Predicted vs Actual Organic Content")
plt.legend()
plt.show()

# Display first 10 predictions
print("\nFirst 10 Predictions:")
print(test_predictions.head(10))
