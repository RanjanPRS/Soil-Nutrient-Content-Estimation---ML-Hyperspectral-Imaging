import numpy as np
import pandas as pd

# Load the dataset
df = pd.read_excel('/home/user/Documents/FYP/SPLIT_N_P_K_LOW_MEDIUM_HIGH.xlsx')

# List of non-relevant columns to exclude
exclude_columns = ['Sample Code', 'OC_Output', 'N_Output', 'P_Output', 'K_Output', 'Wavelength', 'OC_value', 'N_value', 'k_value', 'p_value']

# Select only numeric columns and exclude the non-relevant ones
df_numeric = df.drop(columns=exclude_columns, errors='ignore')

# Now ensure we're only working with numeric columns that represent wavelengths or OC bands
df_numeric = df_numeric.select_dtypes(include=[np.number])

# Objective function (to minimize)
def objective_function(x):
    return np.sum(x**2)

# Bat Algorithm Functions (initialization, update, local search, etc.)

def initialize_bats(n_bats, dim, lower_bound, upper_bound, f_min, f_max, A0, r0):
    bats = np.random.uniform(lower_bound, upper_bound, (n_bats, dim))
    velocities = np.zeros((n_bats, dim))
    frequencies = np.random.uniform(f_min, f_max, n_bats)  # Initialize frequencies
    pulse_rates = r0 * np.ones(n_bats)  # Initialize pulse rates
    loudness = A0 * np.ones(n_bats)  # Initialize loudness
    return bats, velocities, frequencies, pulse_rates, loudness

def update_position_velocity(bats, velocities, frequencies, best_bat, lower_bound, upper_bound):
    velocities += (bats - best_bat) * frequencies[:, np.newaxis]  # Velocity update
    bats += velocities  # Position update
    # Apply boundaries
    bats = np.clip(bats, lower_bound, upper_bound)
    return bats, velocities

def local_search(bat, best_bat, avg_loudness):
    epsilon = np.random.uniform(-1, 1, bat.shape)
    return best_bat + epsilon * avg_loudness

def bat_algorithm(n_bats, dim, lower_bound, upper_bound, max_iter, f_min=0, f_max=100, alpha=0.9, gamma=0.9, A0=1, r0=0.5):
    bats, velocities, frequencies, pulse_rates, loudness = initialize_bats(n_bats, dim, lower_bound, upper_bound, f_min, f_max, A0, r0)
    fitness = np.array([objective_function(bat) for bat in bats])
    best_bat = bats[np.argmin(fitness)]
    best_fitness = np.min(fitness)

    for t in range(max_iter):
        for i in range(n_bats):
            bats, velocities = update_position_velocity(bats, velocities, frequencies, best_bat, lower_bound, upper_bound)
            if np.random.rand() > pulse_rates[i]:
                avg_loudness = np.mean(loudness)
                new_bat = local_search(bats[i], best_bat, avg_loudness)
            else:
                new_bat = bats[i]
            new_fitness = objective_function(new_bat)

            if np.random.rand() < loudness[i] and new_fitness < fitness[i]:
                bats[i] = new_bat
                fitness[i] = new_fitness
                loudness[i] *= alpha  # Update loudness using At+1 = α * At
                pulse_rates[i] = r0 * (1 - np.exp(-gamma * t))  # Update pulse rate using rt+1 = r0 * (1 - exp(-γ * t))

            if new_fitness < best_fitness:
                best_bat = new_bat
                best_fitness = new_fitness

    return bats, fitness

# Main function to get the 10 best OC bands
def get_best_oc_bands(df_numeric, n_bats=20, dim=5, lower_bound=-10, upper_bound=10, max_iter=1000):
    # Run the bat algorithm
    bats, fitness = bat_algorithm(n_bats, dim, lower_bound, upper_bound, max_iter)

    # Sort the fitness values and select the 10 best OC bands (by column index)
    best_indices = np.argsort(fitness)[:10]  # Get indices of the 10 best OC bands
    
    # Get the column names corresponding to the best indices (wavelengths)
    best_oc_bands = df_numeric.columns[best_indices]

    # Filter out any non-numeric column names that may have been accidentally included
    best_oc_bands = [col for col in best_oc_bands if isinstance(col, (int, float))]

    return best_oc_bands

# Get the top 10 best OC bands
best_oc_bands = get_best_oc_bands(df_numeric)
print("Top 10 Best OC Bands:", best_oc_bands)
