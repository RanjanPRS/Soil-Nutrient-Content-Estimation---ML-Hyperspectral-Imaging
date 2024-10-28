import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
import matplotlib.pyplot as plt

# Loads dataset
file_path = '/home/user/Documents/FYP/SPLIT_N_P_K_LOW_MEDIUM_HIGH.xlsx'
df = pd.read_excel(file_path)

# Mapping and dropping NaNs for 'K'
if df['K'].dtype == 'object': 
    df['K'] = df['K'].map({'MEDIUM': 1, 'HIGH': 2})
df = df.dropna(subset=['K'])

# Selecting relevant features and target variable
X = df[[2215, 2216, 2008, 1456, 908, 393, 2216, 2005, 469, 370]]
y = df['K']

# Splitting and scaling the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Re-train the model with optimal n_components
pls = PLSRegression(n_components=7)
pls.fit(X_train_scaled, y_train)

# Predict and evaluate
y_pred = pls.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
evs = explained_variance_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
print(f"Mean Absolute Error: {mae}")
print(f"Root Mean Squared Error: {rmse}")
print(f"Explained Variance Score: {evs}")

# Plot True vs. Predicted values
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("True K Values")
plt.ylabel("Predicted K Values")
plt.title("True vs. Predicted K Values")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.show()
