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
    Objective function to calculate the significance of selected bands.
    Here, we sum reflectance values across selected bands, aiming to minimize it.
    Modify this function as per your project's specific requirements.
    """
    # Filter only existing bands in df to avoid KeyError
    existing_bands = [band for band in selected_bands if band in df.columns]
    
    # Check if there are any valid bands selected
    if not existing_bands:
        return float('-inf')  # Return a very low score if no valid bands are selected

    # Extract reflectance values for existing bands, converting to numeric and handling non-numeric data
    reflectance_values = df[existing_bands].apply(pd.to_numeric, errors='coerce')
    
    # Replace any NaN values (from non-numeric data) with 0 or an appropriate value
    reflectance_values = reflectance_values.fillna(0)

    # Sum reflectance values across the selected bands
    score = reflectance_values.to_numpy().sum()  # Minimize reflectance sum for dips
    return -score  # Lower scores are better

# Initialize FOA parameters
num_foxes = 10
num_iterations = 100
min_band = df.columns.min()  # Minimum numeric band
max_band = df.columns.max()  # Maximum numeric band

# Initialize fox population randomly within the band range
foxes = [np.random.choice(df.columns, size=5, replace=False) for _ in range(num_foxes)]

# Define FOA optimization loop
for iteration in range(num_iterations):
    for i in range(num_foxes):
        # Evaluate the objective function for the current fox's selected bands
        score = objective_function(foxes[i])

        # Randomly select another fox and compare scores
        random_fox_index = np.random.randint(0, num_foxes)
        random_fox = foxes[random_fox_index]
        random_fox_score = objective_function(random_fox)

        # If the random fox has a better score, move towards its bands
        if random_fox_score < score:
            # Move towards the random fox by modifying bands
            new_selection = set(foxes[i]) | set(random_fox)
            if len(new_selection) > 5:  # Ensure it doesn't exceed band selection limit
                new_selection = np.random.choice(list(new_selection), size=5, replace=False)
            foxes[i] = new_selection

# Find the best solution from the final fox population
best_fox = min(foxes, key=objective_function)
best_score = objective_function(best_fox)

# Print the optimal band selection and the corresponding score
print("Optimal Bands Selected:", best_fox)
print("Best Score:", best_score)
