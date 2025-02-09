import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.cross_decomposition import PLSRegression
import matplotlib.pyplot as plt

# Load Dataset
data = pd.read_excel('/home/user/Documents/FYP/SPLIT_N_P_K_LOW_MEDIUM_HIGH.xlsx')

# Define Bands for K Categories
bands_k = {
    'Medium': [1274,1192,1177,1180,1181,1183,1184,1189,1190,1193],
    'High': [2500,868,852,2060,856,859,2052,2051,881,2021]
}

# Target Variable
k_column = 'K'

# Define Models
param_grid_lasso = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
models = {
    'Linear Regression': LinearRegression(),
    'Lasso Regression': GridSearchCV(Lasso(random_state=42), param_grid_lasso, cv=5, scoring='r2'),
    'Random Forest': GridSearchCV(RandomForestRegressor(random_state=42), param_grid_rf, cv=5, scoring='r2'),
    'PLSR': PLSRegression(n_components=9)
}

# Process Each K Category Separately
for category, bands in bands_k.items():
    print(f"\nProcessing K Category: {category}\n")

    # Subset Data for Relevant Bands
    X = data[bands].values
    y = data[k_column].values

    # Split Data into Training and Testing Sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Polynomial Features
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    # Standardize Features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_poly)
    X_test_scaled = scaler.transform(X_test_poly)

    # Apply PCA
    n_components = min(10, X_train_scaled.shape[1])  # Adjust dynamically based on data
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    # Check Explained Variance Ratio
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    print("Explained Variance Ratio:", explained_variance_ratio)
    print("Cumulative Explained Variance:", cumulative_variance)

    # Evaluate Models
    results = {}
    true_vs_pred = {}
    for model_name, model in models.items():
        # Train the model
        model.fit(X_train_pca, y_train)

        # Predict on Test Data
        y_pred = model.predict(X_test_pca)

        # Calculate Metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        medae = median_absolute_error(y_test, y_pred)

        # Store Results
        results[model_name] = {'MSE': mse, 'R²': r2, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'MedAE': medae}
        true_vs_pred[model_name] = (y_test, y_pred)

    # Plot True vs Predicted Values for Each Model
    plt.figure(figsize=(15, 10))
    for i, (model_name, (y_true, y_pred)) in enumerate(true_vs_pred.items(), 1):
        plt.subplot(2, 2, i)  # Arrange in a 2x2 grid for better spacing
        plt.scatter(y_true, y_pred, color='blue', edgecolors='k', alpha=0.7)
        plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], linestyle='--', color='red', linewidth=2)
        plt.title(f'True vs Predicted for K {category} - {model_name}', fontsize=12, pad=10)
        plt.xlabel('True Values', fontsize=10)
        plt.ylabel('Predicted Values', fontsize=10)
        plt.grid(True)

        # Add text annotation with adjusted font size and position
        metrics = results[model_name]
        textstr = '\n'.join((
            f"R²: {metrics['R²']:.4f}",
            f"RMSE: {metrics['RMSE']:.4f}",
            f"MAE: {metrics['MAE']:.4f}",
        ))
        plt.text(0.05, 0.85, textstr, transform=plt.gca().transAxes, fontsize=9,
                 verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.5))

    # Adjust layout to prevent overlapping
    plt.tight_layout(pad=3.0)
    plt.show()

    # Plot Cumulative Explained Variance
    plt.figure(figsize=(8, 5))
    plt.plot(cumulative_variance, marker='o', linestyle='--', color='b')
    plt.title(f'Cumulative Explained Variance by PCA Components for K {category}', fontsize=14, pad=10)
    plt.xlabel('Number of Principal Components', fontsize=12)
    plt.ylabel('Cumulative Explained Variance', fontsize=12)
    plt.grid()
    plt.tight_layout()
    plt.show()
