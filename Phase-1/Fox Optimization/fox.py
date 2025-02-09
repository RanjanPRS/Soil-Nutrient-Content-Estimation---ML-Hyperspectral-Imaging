import numpy as np
import pandas as pd

# Load dataset
df = pd.read_excel('/home/user/Documents/FYP/SPLIT_N_P_K_LOW_MEDIUM_HIGH.xlsx')

# Filter only numeric columns
numeric_columns = [col for col in df.columns if str(col).isdigit()]
df = df[numeric_columns]
df.columns = df.columns.astype(int)  # Convert columns to integers for easier handling

# Define the objective function
def objective_function(selected_bands):
    """
    Objective function calculates a weighted score based on desired properties.
    Reflectance is minimized, with adjustments based on selected characteristics.
    """
    # Check for valid bands
    existing_bands = [band for band in selected_bands if band in df.columns]
    if not existing_bands:
        return float('-inf')  # Return a very low score if no valid bands are selected

    # Extract reflectance values, handle non-numeric data
    reflectance_values = df[existing_bands].apply(pd.to_numeric, errors='coerce').fillna(0)
    score = reflectance_values.to_numpy().sum()  # Minimize reflectance sum for dips
    
    # Apply subtle weighting for desired ranges or properties, e.g., bands in higher reflectance zones
    weight = sum(1 for band in selected_bands if 200 < band < 2400)  # Example band range
    return -score + (weight * 10)  # Encourage selections in specific ranges

# FOA parameters
num_foxes = 10
num_iterations = 100
min_band = df.columns.min()
max_band = df.columns.max()

# Initialize fox population with a distribution favoring a desired band range
foxes = [
    np.random.choice(df.columns[df.columns.between(200, 2400)], size=5, replace=False) if i < num_foxes // 2 
    else np.random.choice(df.columns, size=5, replace=False)
    for i in range(num_foxes)
]

# FOA optimization loop
for iteration in range(num_iterations):
    for i in range(num_foxes):
        # Evaluate the objective function for the current fox's selected bands
        score = objective_function(foxes[i])

        # Randomly select another fox and compare scores
        random_fox_index = np.random.randint(0, num_foxes)
        random_fox = foxes[random_fox_index]
        random_fox_score = objective_function(random_fox)

        # Move towards bands of the better-performing fox
        if random_fox_score > score:
            new_selection = set(foxes[i]) | set(random_fox)
            if len(new_selection) > 5:
                new_selection = np.random.choice(list(new_selection), size=5, replace=False)
            foxes[i] = new_selection

# Identify the optimal selection
best_fox = max(foxes, key=objective_function)
best_score = objective_function(best_fox)

# Output the results
print("Optimal Bands Selected:", best_fox)