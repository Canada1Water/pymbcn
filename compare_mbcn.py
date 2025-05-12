import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

# Load the datasets
r_ds = xr.open_dataset('r_corrected_output.nc')
py_ds = xr.open_dataset('python_corrected_output.nc')

# List of MBCn variables to compare
variables = ['pr', 'tas', 'dtr', 'sfcWind', 'ps', 'huss', 'rsds', 'rlds']

def plot_comparison(var, period):
    """Plot comparison of R vs Python for a given variable and period"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Get the data
    r_data = r_ds[f'mbcn_{var}_{period}']
    py_data = py_ds[f'mbcn_{var}_{period}']
    
    # Plot both timeseries
    r_data.plot(ax=ax, label=f'R MBCn {var}', alpha=0.7)
    py_data.plot(ax=ax, label=f'Python MBCn {var}', alpha=0.7)
    
    # Calculate and display correlation
    corr = np.corrcoef(r_data.values, py_data.values)[0,1]
    ax.set_title(f'MBCn {var} {period} comparison (corr={corr:.4f})')
    ax.legend()
    ax.grid(True)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(f'mbcn_{var}_{period}_comparison.png')
    plt.close()

# Compare both control and projection periods for all variables
for var in variables:
    plot_comparison(var, 'c')  # Control period
    plot_comparison(var, 'p')  # Projection period

print("Comparison plots saved as mbcn_*_comparison.png files")
