import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the dataset
data = pd.ExcelFile('SPLIT_N_P_K_LOW_MEDIUM_HIGH.xlsx')
df = data.parse('SPLIT_N_P_K_LOW_MEDIUM_HIGH')

# Filter data for N_Output values 'LOW' and 'Low1'
filtered_df = df[df['N_Output'].isin(['LOW', 'Low1'])]

# Extract hyperspectral band data and target variable
X = filtered_df.iloc[:, 10:]
X.columns = X.columns.astype(str)  # Ensure column names are strings
band_columns = [col for col in X.columns if col.isdigit()]  # Identify numeric band columns
X = X[band_columns]

# Drop rows with NaN or problematic values
X = X.apply(pd.to_numeric, errors='coerce').dropna()

# Align the target variable 'y' with filtered rows
y = filtered_df.loc[X.index, 'N']

# Standardize the numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply mutual information to rank the importance of bands for Nitrogen
mi_scores = mutual_info_regression(X_scaled, y, random_state=42)

# Rank bands by importance
band_importance = pd.DataFrame({
    'Band': band_columns,
    'MI_Score': mi_scores
}).sort_values(by='MI_Score', ascending=False)

# Select top 10 bands
top_10_bands = band_importance.head(10)['Band'].values
X_top = X[top_10_bands]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_top, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    'Linear Regression': LinearRegression(),
    'Lasso Regression': Lasso(alpha=0.1),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'PLSR': PLSRegression(n_components=2)
}

# Train and evaluate models
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    results.append({'Model': name, 'R²': r2, 'MSE': mse})

# Convert results to DataFrame
results_df = pd.DataFrame(results)
print(results_df)

# Plot performance metrics
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.bar(results_df['Model'], results_df['R²'])
plt.title('R² Scores by Model')
plt.ylabel('R² Score')

plt.subplot(1, 2, 2)
plt.bar(results_df['Model'], results_df['MSE'])
plt.title('MSE Scores by Model')
plt.ylabel('MSE')

plt.tight_layout()
plt.show()