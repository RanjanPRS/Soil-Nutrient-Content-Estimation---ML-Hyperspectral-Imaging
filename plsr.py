import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset
file_path = '/home/user/Documents/FYP/SPLIT_N_P_K_LOW_MEDIUM_HIGH.xlsx'
df = pd.read_excel(file_path)

# Inspect unique values in the OC column before any processing
print("Unique values in OC before any processing:", df['OC'].unique())

# If OC is already numeric, you may skip the mapping step
# Check if mapping is necessary
if df['OC'].dtype == 'object':  # Only map if OC is of type object (string)
    df['OC'] = df['OC'].map({'LOW': 0, 'MEDIUM': 1, 'HIGH': 2})

# Check for NaN values after mapping
nan_count = df['OC'].isna().sum()
print(f"NaN values in OC after mapping: {nan_count}")

# Drop rows with NaN values in OC
df = df.dropna(subset=['OC'])

# Check the shape of the DataFrame after dropping NaN values
print(f"Shape of DataFrame after dropping NaN: {df.shape}")

# If there are still rows left, proceed with analysis
if df.shape[0] > 0:
    # Relevant wavelength and reflectance columns
    X = df[[927, 2396, 382, 2007, 1453, 2216, 2402, 380, 904, 2213, 2491, 876, 474, 1456, 2005]]  # Wavelength columns

    # Target variable (OC only)
    y = df[['OC']]  # Only target OC

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Apply PLSR with an increased number of components
    n_components = 5  # Increase the number of components
    pls = PLSRegression(n_components=n_components)
    pls.fit(X_train_scaled, y_train)

    # Predict on the test set
    y_pred = pls.predict(X_test_scaled)

    # Evaluate the model (for OC)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")

    # Visualize Results
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel("True OC Values")
    plt.ylabel("Predicted OC Values")
    plt.title("True vs. Predicted OC Values")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)  # Diagonal line
    plt.show()
else:
    print("No valid samples left after processing. Please check your data.")
