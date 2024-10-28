import pandas as pd
from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt

file_path = '/home/user/Documents/FYP/SPLIT_N_P_K_LOW_MEDIUM_HIGH.xlsx'
df = pd.read_excel(file_path, sheet_name='SPLIT_N_P_K_LOW_MEDIUM_HIGH')

print(df.head())

# Ensure that the column names are strings and drop NaN values
df.columns = df.columns.astype(str)
df.columns = df.columns.str.replace('.', '', regex=False)

# Select only numeric columns (wavelengths)
numeric_columns = df.columns[df.columns.str.isdigit()]

# Use the numeric columns as string
wavelengths = numeric_columns  
reflectance_values_cleaned = pd.to_numeric(df.loc[df['Sample Code'] == 'RP-22A', wavelengths].values.flatten(), errors='coerce')

# Process the reflectance values to find dips
inverted_signal = -reflectance_values_cleaned
dips_indices, _ = find_peaks(inverted_signal)

# Extract the corresponding wavelength values for these dips
dips_wavelengths = wavelengths[dips_indices].astype(int)  
dips_values = reflectance_values_cleaned[dips_indices]  

# Calculate the average reflectance for each band
average_reflectance = df[numeric_columns].mean()

# Get the top 10 significant bands based on average reflectance
top_bands = average_reflectance.nlargest(10)

# Plotting the reflectance spectrum
plt.figure(figsize=(12, 6))
plt.plot(wavelengths.astype(int), reflectance_values_cleaned, label='Reflectance Spectrum', color='blue')  
plt.scatter(dips_wavelengths.astype(int), dips_values, color='red', label='Dips', zorder=5)  
plt.title('Reflectance Spectrum for RP-22A with Identified Dips')  
plt.xlabel('Wavelength (nm)')  
plt.ylabel('Reflectance')  
plt.legend()  
plt.grid()  

# Highlight the top 10 significant bands on the plot
for band in top_bands.index:
    plt.axvline(x=int(band), color='green', linestyle='--', label=f'Top Band {band}')

plt.savefig('reflectance_spectrum_RP-6B.png')
plt.show()  