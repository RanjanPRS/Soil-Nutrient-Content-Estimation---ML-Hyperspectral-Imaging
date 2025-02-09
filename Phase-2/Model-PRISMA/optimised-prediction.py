import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.feature_selection import mutual_info_regression

# ✅ File Paths
train_file_path = "/home/user/Documents/FYP/SPLIT_N_P_K_LOW_MEDIUM_HIGH.xlsx"
test_file_path = "/home/user/Downloads/PRISMA_DATA.xlsx"

# ✅ Load Datasets
train_df = pd.read_excel(train_file_path, sheet_name="SPLIT_N_P_K_LOW_MEDIUM_HIGH")
test_df = pd.read_excel(test_file_path)

# ✅ Ensure Column Names are Strings
train_df.columns = train_df.columns.astype(str)
test_df.columns = test_df.columns.astype(str)

# ✅ PRISMA Wavelength Mapping (171 Bands: SWIR only)
prisma_wavelengths = np.linspace(920, 2500, 171).tolist()
test_df.columns = [str(round(w)) for w in prisma_wavelengths]

# ✅ Optimal Bands for Each Nutrient
optimal_bands = {
    "OC": [956, 980, 1008, 1026, 1061, 1242, 1460, 1472, 1585, 1633, 1657, 1730, 1755, 1757, 2064, 2354, 2373],
    "N": [1079, 1269, 1318, 1339, 1474, 1490, 1511, 1523, 1622, 1687, 1691, 1793, 2059, 2267, 2296, 2363, 2436, 2455],
    "P": [960, 1067, 1226, 1234, 1236, 1289, 1296, 1698, 1793, 2032, 2048, 2051, 2091, 2096, 2147, 2336, 2371, 2372, 2374],
    "K": [986, 1033, 1064, 1312, 1340, 1341, 1481, 1524, 2070, 2101, 2196, 2304, 2307, 2321, 2389, 2425, 2448]
}

# ✅ Find Nearest Bands
def find_best_band(band, available_bands):
    """Find the nearest available band."""
    if band < 920:
        return None
    if str(int(band)) in available_bands:
        return str(int(band))
    available_bands = np.array([float(b) for b in available_bands])
    return str(int(available_bands[np.abs(available_bands - band).argmin()]))

# ✅ Get Common Features
def get_common_features(train_df, test_df, optimal_bands, nutrient):
    """Find common bands between training and test dataset."""
    common_features = []
    for band in optimal_bands:
        best_band = find_best_band(band, test_df.columns)
        if best_band and best_band in train_df.columns:
            common_features.append(best_band)
    return list(set(common_features))

# ✅ Train Model with Feature Importance Analysis
def train_model(nutrient):
    """Train and Evaluate Models for Each Nutrient."""
    common_features = get_common_features(train_df, test_df, optimal_bands[nutrient], nutrient)

    if len(common_features) == 0:
        raise ValueError(f"No spectral bands found for {nutrient}! Check dataset.")

    print(f"✅ Using {len(common_features)} bands for {nutrient}")

    X_train = train_df[common_features]
    y_train = train_df[nutrient]
    X_test = test_df[common_features]

    # ✅ Normalize Data
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_split, X_val, y_train_split, y_val = train_test_split(X_train_scaled, y_train, test_size=0.2, random_state=42)

    # ✅ Feature Selection using Mutual Information
    mi_scores = mutual_info_regression(X_train_split, y_train_split)
    mi_ranking = pd.Series(mi_scores, index=common_features).sort_values(ascending=False)
    selected_features = mi_ranking.index[:10]  # Top 10 features

    X_train_split = X_train_split[:, :10]
    X_val = X_val[:, :10]
    X_test_scaled = X_test_scaled[:, :10]

    # ✅ Train Models
    xgb_model = XGBRegressor(n_estimators=600, max_depth=8, learning_rate=0.02, subsample=0.75, colsample_bytree=0.85, random_state=42)
    rf_model = RandomForestRegressor(n_estimators=600, max_depth=20, min_samples_split=5, bootstrap=False, random_state=42)

    xgb_model.fit(X_train_split, y_train_split)
    rf_model.fit(X_train_split, y_train_split)

    y_val_pred_xgb = xgb_model.predict(X_val)
    y_val_pred_rf = rf_model.predict(X_val)

    # ✅ Print Performance
    print(f"\nXGBoost Performance for {nutrient}: R² Score: {r2_score(y_val, y_val_pred_xgb):.4f}, MSE: {mean_squared_error(y_val, y_val_pred_xgb):.4f}")
    print(f"Random Forest Performance for {nutrient}: R² Score: {r2_score(y_val, y_val_pred_rf):.4f}, MSE: {mean_squared_error(y_val, y_val_pred_rf):.4f}")

# ✅ Run Training
for nutrient in ["OC", "N", "P", "K"]:
    train_model(nutrient)
