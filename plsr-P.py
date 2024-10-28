import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score

# Load your dataset
data = pd.read_excel('/home/user/Documents/FYP/SPLIT_N_P_K_LOW_MEDIUM_HIGH.xlsx')

# Select features (wavelengths) and target (P)
X = data.select_dtypes(include=[np.number])  # Select only numeric columns (wavelengths)
y = data['P']  # target variable = P

# Convert non-numeric values to NaN for error handling
X = X.apply(pd.to_numeric, errors='coerce')  # Convert strings to NaN if any

# Drop any rows with NaN values
X.dropna(inplace=True)
y = y[X.index]  # Align y with the remaining X after dropping NaNs

# Convert column names to strings for manipulation & error handling
X.columns = X.columns.astype(str)

# Check if X is empty before scaling
if not X.empty:  # Ensure X has samples
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
else:
    raise ValueError("X is empty after removing NaNs. Check the input data.")

# Step 2: Outlier Detection using IQR
Q1 = np.percentile(X_scaled, 25, axis=0)
Q3 = np.percentile(X_scaled, 75, axis=0)
IQR = Q3 - Q1
mask = ~((X_scaled < (Q1 - 1.5 * IQR)) | (X_scaled > (Q3 + 1.5 * IQR))).any(axis=1)

# Apply mask to both features and target
X_scaled = X_scaled[mask]
y = y[mask].reset_index(drop=True)  # Reset index for target variable after masking

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Hyperparameter tuning with GridSearchCV
param_grid = {'n_components': range(1, 11)}  # Adjust the range as necessary
pls = PLSRegression()
grid = GridSearchCV(pls, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
grid.fit(X_train, y_train)

# Get the best model
best_pls = grid.best_estimator_
best_n_components = grid.best_params_['n_components']

# Predictions
y_pred = best_pls.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
evs = explained_variance_score(y_test, y_pred)

# Print evaluation metrics
print(f"Best n_components: {best_n_components}")
print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")
print(f"R-squared: {r2}")
print(f"Explained Variance Score: {evs}")

# Plotting results
plt.figure(figsize=(12, 6))

# Scatter plot of actual vs. predicted values
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')  # 45-degree line
plt.xlabel('Actual P Values')
plt.ylabel('Predicted P Values')
plt.title('Actual vs. Predicted P Values')
plt.grid()

# Residual plot
plt.subplot(1, 2, 2)
residuals = y_test - y_pred.flatten()
plt.scatter(y_pred, residuals, color='blue', alpha=0.6)
plt.axhline(0, color='red', linestyle='--')  # Horizontal line at 0
plt.xlabel('Predicted P Values')
plt.grid()
plt.show()
