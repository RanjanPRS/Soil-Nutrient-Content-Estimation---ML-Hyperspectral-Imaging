import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

# Dataset Loading
file_path = 'SPLIT_N_P_K_LOW_MEDIUM_HIGH.xlsx'
df = pd.read_excel(file_path)

# Identify the columns that contain categorical values
# Assuming the first 5 columns contain categorical values
categorical_cols = df.select_dtypes(include=['object']).columns  # Automatically identify categorical columns

# Apply Label Encoding to categorical columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Save the encoder for potential inverse transformations

# Extract the target variables (OC, N, P, K)
targets = ['OC', 'N', 'P', 'K']
X = df.iloc[:, 5:].values  # The hyperspectral bands (excluding the first 5 columns)
y = df[targets].values  # The target values for soil nutrients

print(f"Features (bands): {X.shape}, Targets: {y.shape}")

# Initialize Fox Optimization parameters
def initialize_population(pop_size, num_features):
    return np.random.randint(2, size=(pop_size, num_features))

def fitness_function(solution, X, y):
    # Select the bands based on the solution (1 = band selected, 0 = not selected)
    selected_bands = X[:, solution == 1]
    
    # If no bands are selected, return a large fitness value (bad solution)
    if selected_bands.shape[1] == 0:
        return float('inf')
    
    # Train a RandomForest model on the selected bands
    X_train, X_test, y_train, y_test = train_test_split(selected_bands, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    # Evaluate performance using mean squared error
    return mean_squared_error(y_test, predictions)

def move_foxes(foxes, best_fox, exploration_rate):
    # Adjust each fox's position (solution) based on the best solution found so far
    new_population = []
    for fox in foxes:
        # Move towards the best solution with some random noise for exploration
        new_fox = fox + exploration_rate * (best_fox - fox) + np.random.uniform(-0.5, 0.5, len(fox))
        new_fox = np.clip(new_fox, 0, 1)  # Keep values between 0 and 1
        new_fox = (new_fox > 0.5).astype(int)  # Convert back to binary (0 or 1)
        new_population.append(new_fox)
    return np.array(new_population)

def fox_optimization(X, y, num_iterations=100, pop_size=50, exploration_rate=0.1):
    num_features = X.shape[1]
    population = initialize_population(pop_size, num_features)
    best_solution = None
    best_fitness = float('inf')
    
    for iteration in range(num_iterations):
        fitness_values = []
        for fox in population:
            fitness = fitness_function(fox, X, y)
            fitness_values.append(fitness)
            if fitness < best_fitness:
                best_fitness = fitness
                best_solution = fox
        
        # Move foxes towards the best solution
        population = move_foxes(population, best_solution, exploration_rate)
        
        print(f"Iteration {iteration + 1}/{num_iterations}, Best Fitness: {best_fitness}")
    
    return best_solution, best_fitness

# Run the Fox Optimization Algorithm to select the top relevant bands
best_solution, best_fitness = fox_optimization(X, y)

# Print the indices of the selected bands
selected_bands = np.where(best_solution == 1)[0]
print(f"Top Selected Bands: {selected_bands}")
