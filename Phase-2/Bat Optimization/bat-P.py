import numpy as np
from scipy.stats import pearsonr
import pandas as pd

# Define the Bat Algorithm
class BatAlgorithm:
    def __init__(self, obj_func, num_bats, num_dims, lower_bound, upper_bound, max_iter):
        self.obj_func = obj_func  # Objective function to maximize
        self.num_bats = num_bats
        self.num_dims = num_dims
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.max_iter = max_iter

        # Initialize bat positions and velocities
        self.positions = np.random.uniform(lower_bound, upper_bound, (num_bats, num_dims))
        self.velocities = np.zeros((num_bats, num_dims))
        self.frequencies = np.zeros(num_bats)
        self.loudness = np.ones(num_bats)
        self.pulse_rate = np.ones(num_bats) * 0.5
        self.best_position = self.positions[0]
        self.best_score = float('-inf')

    def optimize(self):
        for _ in range(self.max_iter):
            for i in range(self.num_bats):
                # Update frequency, velocity, and position
                self.frequencies[i] = np.random.uniform(0, 1)
                self.velocities[i] += (self.positions[i] - self.best_position) * self.frequencies[i]
                new_position = self.positions[i] + self.velocities[i]

                # Apply boundary constraints
                new_position = np.clip(new_position, self.lower_bound, self.upper_bound)

                # Evaluate the new position
                score = self.obj_func(new_position)
                if score > self.best_score and np.random.rand() < self.loudness[i]:
                    self.best_position = new_position
                    self.best_score = score

                # Adjust loudness and pulse rate
                self.loudness[i] *= 0.9
                self.pulse_rate[i] = 0.5 * (1 + np.random.rand())

                self.positions[i] = new_position
        return self.best_position, self.best_score

# Load and preprocess the dataset
file_path = '/home/user/Documents/FYP/SPLIT_N_P_K_LOW_MEDIUM_HIGH.xlsx'
data = pd.ExcelFile(file_path)

sheet_name = 'SPLIT_N_P_K_LOW_MEDIUM_HIGH'  # Update if needed
df = data.parse(sheet_name)

# Filter dataset for P categories
p_column = 'P_Output'  # Update to the column representing P categories
p_categories = ['LOW', 'Medium', 'High']
spectral_band_columns = df.columns[14:]  # Assuming spectral bands start at column index 14 (15th column)

# Ensure only numeric columns are selected
spectral_band_columns = df[spectral_band_columns].select_dtypes(include=[float, int]).columns

# Define a function to optimize and extract top bands for a given subset
def optimize_for_p_category(subset, p_values, num_top_bands=10):
    if len(subset) < 2:
        print(f"Warning: Not enough data for this category. Proceeding with available data.")
        return list(spectral_band_columns[:num_top_bands])  # Return the first few bands as a fallback

    # Remove rows with NaN values in spectral bands before processing
    subset = subset[spectral_band_columns].dropna()

    # Check if there are enough rows after cleaning
    if len(subset) < 2:
        print(f"Warning: Not enough data after cleaning. Proceeding with available data.")
        return list(spectral_band_columns[:num_top_bands])

    spectral_band_values = np.array(subset[spectral_band_columns].values.T, dtype=float)

    # Define the objective function for the subset
    def objective_function(weights):
        weights = np.array(weights, dtype=float)
        weighted_sum = np.dot(weights, spectral_band_values)
        correlation, _ = pearsonr(weighted_sum, p_values)
        return abs(correlation)

    # Set up Bat Algorithm parameters
    num_bats = 30
    num_dims = spectral_band_values.shape[0]  # Number of spectral bands
    lower_bound = np.zeros(num_dims)  # Minimum weights
    upper_bound = np.ones(num_dims)  # Maximum weights
    max_iter = 100

    # Run the Bat Algorithm
    bat_algo = BatAlgorithm(objective_function, num_bats, num_dims, lower_bound, upper_bound, max_iter)
    best_weights, _ = bat_algo.optimize()

    # Identify top bands with highest weights
    num_bands = min(num_top_bands, len(best_weights))  # Adjust for fewer bands if data is limited
    top_band_indices = np.argsort(best_weights)[-num_bands:][::-1]
    top_bands = [spectral_band_columns[i] for i in top_band_indices]

    return top_bands

# Apply the process for each P category
results = {}
for category in p_categories:
    subset = df[df[p_column] == category]
    p_values = np.array(subset['P'], dtype=float)  # Extract P values for correlation
    results[category] = optimize_for_p_category(subset, p_values)

# Print results for each P category
for category, bands in results.items():
    print(f"Top bands for {category} : {bands}")
