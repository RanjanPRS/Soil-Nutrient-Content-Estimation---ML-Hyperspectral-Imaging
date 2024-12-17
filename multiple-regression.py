import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
import matplotlib.pyplot as plt

# Load Dataset
data = pd.read_excel('SPLIT_N_P_K_LOW_MEDIUM_HIGH.xlsx')

# Relevant bands to consider for features
relevant_bands = [899, 2078, 738, 853, 852, 2046, 2047, 737, 736, 2063]

# Extract Features (Wavelengths) and Target (OC values)
feature_columns = [col for col in data.columns if isinstance(col, (int, float))]
X = data[feature_columns].values
y = data['OC'].values

# Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply PCA
n_components = 10  # Number of components to retain
pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Check Explained Variance Ratio
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance_ratio)
print("Explained Variance Ratio:", explained_variance_ratio)
print("Cumulative Explained Variance:", cumulative_variance)

# Initialize Models
models = {
    'Linear Regression': LinearRegression(),
    'Lasso Regression': Lasso(alpha=0.1, random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'PLSR': PLSRegression(n_components=9)
}

# Evaluate Models on PCA-Transformed Data
results = {}
for model_name, model in models.items():
    # Train the model
    model.fit(X_train_pca, y_train)

    # Predict on Test Data
    y_pred = model.predict(X_test_pca)

    # Calculate Metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)

    # Store Results
    results[model_name] = {'MSE': mse, 'R²': r2, 'RMSE': rmse}

# Print Results
for model_name, metrics in results.items():
    print(f"{model_name} Results:")
    print(f"  Mean Squared Error (MSE): {metrics['MSE']:.4f}")
    print(f"  R² Score: {metrics['R²']:.4f}")
    print(f"  Root Mean Squared Error (RMSE): {metrics['RMSE']:.4f}\n")

# Plot Cumulative Explained Variance
plt.figure(figsize=(8, 5))
plt.plot(cumulative_variance, marker='o', linestyle='--', color='b')
plt.title('Cumulative Explained Variance by PCA Components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid()
plt.show()

# Plot Predicted vs. Actual Values for Each Model
plt.figure(figsize=(12, 8))
for i, (model_name, model) in enumerate(models.items(), start=1):
    # Predict on Test Data
    y_pred = model.predict(X_test_pca)

    # Plot Predicted vs. Actual Values
    plt.subplot(2, 2, i)
    plt.scatter(y_test, y_pred, alpha=0.7, color='blue')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    plt.title(f'{model_name} - Predicted vs Actual')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.grid()

plt.tight_layout()
plt.show()
