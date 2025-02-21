import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.feature_selection import RFE
from sklearn.model_selection import RandomizedSearchCV

# File Paths
train_file_path = "/home/user/Documents/FYP/SPLIT_N_P_K_LOW_MEDIUM_HIGH.xlsx"
test_file_path = "/home/user/Downloads/PRISMA_DATA.xlsx"

# Load Data
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
    "OC": [1042, 1065, 1215, 1321, 1539, 1695, 1752, 1788, 2173, 2370, 2416, 969, 1160, 1195, 1227, 1485, 1636, 1685, 1720, 1748, 1789, 2005, 2099, 2143, 1251, 1296, 1469, 1501, 1557, 1644, 2017, 2137, 2179, 2187, 2384, 2497],
    "N": [1007, 1207, 1230, 1637, 1759, 923, 929, 1006, 1042, 1045, 1209, 1214, 1225, 1657, 1762],
    "P": [1078, 1204, 1269, 1308, 1636, 1644, 2008, 2064, 2096, 2110, 2194, 2317, 2391, 931, 1175, 1623, 1627, 1690, 2011, 2152, 2205, 2269, 2500, 978, 1017, 1099, 1453, 1567, 1689, 2257, 2330, 2388, 2429, 2500],
    "K": [933, 1050, 1337, 1516, 1555, 1571, 1651, 1699, 2449, 2500, 940, 965, 1195, 1266, 1314, 1571, 2322, 2500]
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

# Function to Train PLSR Model and Transform Features
def train_plsr_transform(X_train, X_test, y_train):
    """Train PLSR model and transform features before passing to XGBoost."""
    n_components = min(10, X_train.shape[1])  # Limit to 10 components or available features
    pls = PLSRegression(n_components=n_components)
    pls.fit(X_train, y_train)

    X_train_pls = pls.transform(X_train)  # Transform Training Data
    X_test_pls = pls.transform(X_test)  # Transform PRISMA Data

    return X_train_pls, X_test_pls

# Function to Optimize XGBoost using RandomizedSearchCV
def optimize_xgboost(X_train, y_train):
    """Use RandomizedSearchCV to find the best hyperparameters for XGBoost."""
    param_grid = {
        'n_estimators': [200, 400, 600],
        'max_depth': [4, 6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }
    model = XGBRegressor(random_state=42)
    search = RandomizedSearchCV(model, param_grid, n_iter=10, cv=3, scoring='r2', random_state=42, n_jobs=-1)
    search.fit(X_train, y_train)
    return search.best_estimator_

# Function to Train Hybrid Model
def train_hybrid_model(nutrient):
    """Train a hybrid model using PLSR for feature extraction and XGBoost for prediction."""
    common_features = get_common_features(train_df, test_df, optimal_bands[nutrient], nutrient)
    
    if len(common_features) == 0:
        raise ValueError(f"No spectral bands found for {nutrient}! Check dataset.")
    
    print(f"Bands used for {nutrient}: {len(common_features)}")

    X_train = train_df[common_features]
    y_train = train_df[nutrient]
    X_test = test_df[common_features]

    # Normalize Data using MinMaxScaler
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_split, X_val, y_train_split, y_val = train_test_split(X_train_scaled, y_train, test_size=0.2, random_state=42)

    # Transform Features using PLSR
    X_train_pls, X_test_pls = train_plsr_transform(X_train_split, X_test_scaled, y_train_split)

    # Optimize XGBoost Hyperparameters
    best_xgb = optimize_xgboost(X_train_pls, y_train_split)

    # Train XGBoost Model
    best_xgb.fit(X_train_pls, y_train_split)

    y_val_pred = best_xgb.predict(X_train_pls)
    y_test_pred = best_xgb.predict(X_test_pls)

    r2 = r2_score(y_train_split, y_val_pred)
    mse = mean_squared_error(y_train_split, y_val_pred)

    print(f"\nResults for {nutrient}:")
    print(f"   R² Score: {r2:.4f}")
    print(f"   MSE: {mse:.4f} \n")

    return y_test_pred.flatten()

# Run Training and Prediction for PRISMA Data with Hybrid Model
hybrid_predictions = {}
for nutrient in ["OC", "N", "P", "K"]:
    hybrid_predictions[nutrient] = train_hybrid_model(nutrient)

# Convert Predictions to DataFrame and Save
df_hybrid_predictions = pd.DataFrame(hybrid_predictions)
df_hybrid_predictions.to_csv("Hybrid_PLSR_XGBoost_Predictions.csv", index=False)

print("\nFinal Predictions saved as 'Hybrid_PLSR_XGBoost_Predictions.csv'")
print(df_hybrid_predictions.head())  # Print first few predictions
