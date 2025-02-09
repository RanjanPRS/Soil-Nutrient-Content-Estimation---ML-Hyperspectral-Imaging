import rasterio
import numpy as np
import pandas as pd

# Load PRISMA Hyperspectral Image
tif_file = "/home/user/Downloads/PRISMA.tif"

with rasterio.open(tif_file) as src:
    band_count = src.count  # Total bands (spectral channels, now 117)
    height, width = src.shape  # Image dimensions
    print(f"PRISMA Image: {band_count} Bands, {height}x{width} Pixels")

    # Define a small region for testing (100x100 pixels)
    subset_size = 100
    spectral_data = np.zeros((subset_size * subset_size, band_count))

    for i in range(band_count):
        band = src.read(i + 1)[0:subset_size, 0:subset_size].flatten()  # Read a 100x100 portion
        spectral_data[:, i] = band

# Adjust the wavelength list for 117 bands
swir_wavelengths = np.linspace(920, 1800, band_count).tolist()  # Distribute wavelengths evenly

print("Using updated PRISMA SWIR wavelength list:", swir_wavelengths[:10], "...")

# Create a DataFrame with proper column names
df = pd.DataFrame(spectral_data, columns=swir_wavelengths)

# Save to CSV/Excel
df.to_csv("PRISMA_Hyperspectral_Data_Sample.csv", index=False)
df.to_excel("PRISMA_Hyperspectral_Data_Sample.xlsx", index=False)

print("Sample dataset saved as PRISMA_Hyperspectral_Data_Sample.csv and PRISMA_Hyperspectral_Data_Sample.xlsx")
