import pandas as pd
from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt

# Define the file path of your Excel file
file_path = '/home/user/Documents/FYP/SPLIT_N_P_K_LOW_MEDIUM_HIGH.xlsx'

# Load the relevant sheet into a dataframe to inspect the content
df = pd.read_excel(file_path, sheet_name='SPLIT_N_P_K_LOW_MEDIUM_HIGH')

# Display the first few rows of the dataframe to understand its structure
print(df.head())

# Ensure that the column names are strings and drop NaN values
df.columns = df.columns.astype(str)
df.columns = df.columns.str.replace('.', '', regex=False)

# Select only numeric columns (wavelengths)
numeric_columns = df.columns[df.columns.str.isdigit()]

# Check if there are any numeric columns
if len(numeric_columns) == 0:
    print("No numeric columns found.")
else:
    print("Numeric columns:", numeric_columns)

# Use the numeric columns as string
wavelengths = numeric_columns  
# Change the sample code from 'RP-31A' to 'RP-6B'
reflectance_values_cleaned = pd.to_numeric(df.loc[df['Sample Code'] == 'RP-6B', wavelengths].values.flatten(), errors='coerce')

# Process the reflectance values to find dips
inverted_signal = -reflectance_values_cleaned
dips_indices, _ = find_peaks(inverted_signal)

# Extract the corresponding wavelength values for these dips
dips_wavelengths = wavelengths[dips_indices].astype(int)  
dips_values = reflectance_values_cleaned[dips_indices]  

# Debugging: Print key variables
print("Reflectance values for RP-6B:", reflectance_values_cleaned)
print("Dips wavelengths for RP-6B:", dips_wavelengths)
print("Dips values for RP-6B:", dips_values)

# Calculate the average reflectance for each band
average_reflectance = df[numeric_columns].mean()

# Get the top 10 significant bands based on average reflectance
top_bands = average_reflectance.nlargest(10)

# Plotting the reflectance spectrum
plt.figure(figsize=(12, 6))
plt.plot(wavelengths.astype(int), reflectance_values_cleaned, label='Reflectance Spectrum', color='blue')  
plt.scatter(dips_wavelengths.astype(int), dips_values, color='red', label='Dips', zorder=5)  
plt.title('Reflectance Spectrum for RP-6B with Identified Dips')  
plt.xlabel('Wavelength (nm)')  
plt.ylabel('Reflectance')  
plt.legend()  
plt.grid()  

# Highlight the top 10 significant bands on the plot
for band in top_bands.index:
    plt.axvline(x=int(band), color='green', linestyle='--', label=f'Top Band {band}')

# Save the plot
plt.savefig('reflectance_spectrum_RP-6B.png')  # Save the plot as an image file
plt.show()  # Show the plot

# Display the top bands
print("Top 10 significant bands based on average reflectance:\n", top_bands)
