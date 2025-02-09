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


# Define the objective function to maximize the correlation between bands and OC
def objective_function(weights):
    weights = np.array(weights, dtype=float)  # Ensure weights are a numpy array
    weighted_sum = np.dot(weights, spectral_band_values)  # Weighted sum
    correlation, _ = pearsonr(weighted_sum, oc_values)  # Compute correlation with OC
    return abs(correlation)

# Load and preprocess the dataset
file_path = '/home/user/Documents/FYP/SPLIT_N_P_K_LOW_MEDIUM_HIGH.xlsx' 
data = pd.ExcelFile(file_path)

# Get the sheet names to confirm
print("Available sheets:", data.sheet_names)

# Load the correct sheet
sheet_name = 'SPLIT_N_P_K_LOW_MEDIUM_HIGH' 
df = data.parse(sheet_name)

# Extract OC column and spectral band columns
oc_column = 'OC'  # Update if needed
spectral_band_columns = df.columns[8:]  # Assuming spectral bands start at column 9
data_for_optimization = df[[oc_column] + list(spectral_band_columns)]

# Check for non-numeric columns in spectral bands
non_numeric_columns = data_for_optimization[spectral_band_columns].select_dtypes(exclude=[np.number]).columns
if not non_numeric_columns.empty:
    print("Non-numeric columns found and removed:", non_numeric_columns)
    spectral_band_columns = data_for_optimization[spectral_band_columns].select_dtypes(include=[np.number]).columns

# Extract OC values and spectral band values as numpy arrays
oc_values = np.array(data_for_optimization[oc_column].values, dtype=float)
spectral_band_values = np.array(data_for_optimization[spectral_band_columns].values.T, dtype=float)

# Set up the Bat Algorithm
num_bats = 30
num_dims = spectral_band_values.shape[0]  # Number of spectral bands
lower_bound = np.zeros(num_dims)  # Minimum weights
upper_bound = np.ones(num_dims)  # Maximum weights
max_iter = 100

# Run the Bat Algorithm
bat_algo = BatAlgorithm(objective_function, num_bats, num_dims, lower_bound, upper_bound, max_iter)
best_weights, best_score = bat_algo.optimize()

# Identify the top 10 bands with the highest weights
top_10_band_indices = np.argsort(best_weights)[-10:][::-1]
top_10_bands = [list(spectral_band_columns)[i] for i in top_10_band_indices]

# Print the top 10 bands
print("Top 10 bands:", top_10_bands)
