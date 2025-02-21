import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.feature_selection import mutual_info_regression

train_file_path = "/home/user/Documents/FYP/SPLIT_N_P_K_LOW_MEDIUM_HIGH.xlsx"
test_file_path = "/home/user/Downloads/PRISMA_DATA.xlsx"

train_df = pd.read_excel(train_file_path, sheet_name="SPLIT_N_P_K_LOW_MEDIUM_HIGH")
test_df = pd.read_excel(test_file_path)

# Ensure Column Names are Strings
train_df.columns = train_df.columns.astype(str)
test_df.columns = test_df.columns.astype(str)

# PRISMA Wavelength Mapping (171 Bands: SWIR only)
prisma_wavelengths = np.linspace(920, 2500, 171).tolist()
test_df.columns = [str(round(w)) for w in prisma_wavelengths]

# Optimal Bands for Each Nutrient
optimal_bands = {
    "OC": [1042, 1065, 1215, 1321, 1539, 1695, 1752, 1788, 2173, 2370, 2416, 969, 1160, 1195, 1227, 1485, 1636, 1685, 1720, 1748, 1789, 2005, 2099, 2143,1251, 1296, 1469, 1501, 1557, 1644, 2017, 2137, 2179, 2187, 2384, 2497],
    "N": [1007, 1207, 1230, 1637, 1759, 923, 929, 1006, 1042, 1045, 1209, 1214, 1225, 1657, 1762],
    "P": [1078, 1204, 1269, 1308, 1636, 1644, 2008, 2064, 2096, 2110, 2194, 2317, 2391,931, 1175, 1623, 1627, 1690, 2011, 2152, 2205, 2269, 2500,978, 1017, 1099, 1453, 1567, 1689, 2257, 2330, 2388, 2429, 2500],
    "K": [933, 1050, 1337, 1516, 1555, 1571, 1651, 1699, 2449, 2500,940, 965, 1195, 1266, 1314, 1571, 2322, 2500]
}

# Find Nearest Bands
def find_best_band(band, available_bands):
    """Find the nearest available band."""
    if band < 920:
        return None
    if str(int(band)) in available_bands:
        return str(int(band))
    available_bands = np.array([float(b) for b in available_bands])
    return str(int(available_bands[np.abs(available_bands - band).argmin()]))

# Get Common Features
def get_common_features(train_df, test_df, optimal_bands, nutrient):
    """Find common bands between training and test dataset."""
    common_features = []
    for band in optimal_bands:
        best_band = find_best_band(band, test_df.columns)
        if best_band and best_band in train_df.columns:
            common_features.append(best_band)
    return list(set(common_features))

# Train Model with Feature Importance Analysis
def train_model(nutrient):
    """Train and Evaluate Models for Each Nutrient."""
    common_features = get_common_features(train_df, test_df, optimal_bands[nutrient], nutrient)

    if len(common_features) == 0:
        raise ValueError(f"No spectral bands found for {nutrient}! Check dataset.")

    print(f"✅ Using {len(common_features)} bands for {nutrient}")

    X_train = train_df[common_features]
    y_train = train_df[nutrient]
    X_test = test_df[common_features]

    # Normalize Data
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_split, X_val, y_train_split, y_val = train_test_split(X_train_scaled, y_train, test_size=0.2, random_state=42)

    # Feature Selection using Mutual Information
    mi_scores = mutual_info_regression(X_train_split, y_train_split)
    mi_ranking = pd.Series(mi_scores, index=common_features).sort_values(ascending=False)
    selected_features = mi_ranking.index[:10]  # Top 10 features

    X_train_split = X_train_split[:, :10]
    X_val = X_val[:, :10]
    X_test_scaled = X_test_scaled[:, :10]

    # Train Models
    xgb_model = XGBRegressor(n_estimators=600, max_depth=8, learning_rate=0.02, subsample=0.75, colsample_bytree=0.85, random_state=42)
    rf_model = RandomForestRegressor(n_estimators=600, max_depth=20, min_samples_split=5, bootstrap=False, random_state=42)

    xgb_model.fit(X_train_split, y_train_split)
    rf_model.fit(X_train_split, y_train_split)

    y_val_pred_xgb = xgb_model.predict(X_val)
    y_val_pred_rf = rf_model.predict(X_val)

    # Print Performance
    print(f"\nXGBoost Performance for {nutrient}: R² Score: {r2_score(y_val, y_val_pred_xgb):.4f}, MSE: {mean_squared_error(y_val, y_val_pred_xgb):.4f}")
    print(f"Random Forest Performance for {nutrient}: R² Score: {r2_score(y_val, y_val_pred_rf):.4f}, MSE: {mean_squared_error(y_val, y_val_pred_rf):.4f}")

# Run Training
for nutrient in ["OC", "N", "P", "K"]:
    train_model(nutrient)