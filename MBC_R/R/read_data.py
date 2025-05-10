import pandas as pd
import xarray as xr

# Load the CSV files
gcm_c = pd.read_csv('cccma_gcm_c.csv')
gcm_p = pd.read_csv('cccma_gcm_p.csv')

print(gcm_c)

# Convert DataFrames to Xarray Datasets
gcm_c_ds = xr.Dataset.from_dataframe(gcm_c.set_index(['your_index_column']))  # Set your index column if needed
gcm_p_ds = xr.Dataset.from_dataframe(gcm_p.set_index(['your_index_column']))  # Likewise here

# Combine or manipulate datasets as needed
# Assuming you need a combined dataset
combined_ds = xr.merge([gcm_c_ds, gcm_p_ds])

# Save the dataset to a NetCDF file
combined_ds.to_netcdf('cccma_data.nc')

