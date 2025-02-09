import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
import matplotlib.pyplot as plt

# Load the dataset
file_path = '/home/user/Documents/FYP/SPLIT_N_P_K_LOW_MEDIUM_HIGH.xlsx'
df = pd.read_excel(file_path)

# Drop rows with missing 'OC' values
df_cleaned = df.dropna(subset=['OC'])

# Define relevant wavelength columns
selected_columns = [363, 352, 358, 356, 353, 355]

# Prepare features (X) and target (y)
X = df_cleaned[selected_columns]
y = df_cleaned['OC']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply Partial Least Squares Regression (PLSR)
n_components = 2  # Adjust the number of components if necessary
pls = PLSRegression(n_components=n_components)
pls.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = pls.predict(X_test_scaled).flatten()  # Flatten to 1D array

# Evaluate the regression model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Visualize the regression results
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, label="Predictions")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label="Ideal Fit")
plt.xlabel("True OC Values")
plt.ylabel("Predicted OC Values")
plt.title(f"Regression Results\nMSE: {mse:.4f}, R-squared: {r2:.4f}")
plt.legend()
plt.show()

# Define thresholds for classification
def categorize_oc(value):
    if value < 0.33:
        return 0  # LOW
    elif value < 0.66:
        return 1  # MEDIUM
    else:
        return 2  # HIGH

# Apply thresholds to predicted and true values
y_pred_categorical = [categorize_oc(val) for val in y_pred]
y_test_categorical = [categorize_oc(val) for val in y_test]

# Calculate accuracy and generate a classification report
accuracy = accuracy_score(y_test_categorical, y_pred_categorical)
report = classification_report(y_test_categorical, y_pred_categorical, target_names=["LOW", "MEDIUM", "HIGH"])

print(f"Classification Accuracy: {accuracy}")
print("\nClassification Report:\n", report)
