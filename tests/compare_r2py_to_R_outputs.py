import netCDF4
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# --- Configuration ---
PYTHON_NC_FILE = 'r2py_corrected_output.nc'
R_NC_FILE = '../MBC_R/R/r_corrected_output.nc' # Ensure this file exists and has a similar structure

SETS = ['qdm', 'mbcr', 'mbcp', 'mbcn'] 
VARIABLES = ['pr', 'tas', 'dtr', 'sfcWind', 'ps', 'huss', 'rsds', 'rlds']
PERIOD_SUFFIXES = {
    'control': 'c',
    'projection': 'p'
}
PERIOD_NAMES = ['control', 'projection'] # For page titles and file names

# --- Main Script ---
def main():
    for period_name in PERIOD_NAMES:
        period_suffix = PERIOD_SUFFIXES[period_name]
        
        fig, axes = plt.subplots(nrows=len(SETS), ncols=len(VARIABLES), figsize=(20, 10)) 
        fig.suptitle(f'Comparison of R2PY vs. R Outputs - {period_name.capitalize()} Period', fontsize=16)

        try:
            with netCDF4.Dataset(PYTHON_NC_FILE, 'r') as py_ds, \
                 netCDF4.Dataset(R_NC_FILE, 'r') as r_ds:

                for i, set_name in enumerate(SETS):
                    for j, var_name in enumerate(VARIABLES):
                        ax = axes[i, j]
                        
                        # Set titles and labels for the grid structure
                        if i == 0:  # Top row: Set variable name as title
                            ax.set_title(var_name, fontsize=10)
                        
                        if j == 0:  # First column: Set set name as Y-label (row header)
                            ax.set_ylabel(set_name.upper(), fontsize=11, rotation=90, labelpad=15, va='center')
                        
                        # X-axis label for bottom row plots
                        if i == len(SETS) - 1:
                            ax.set_xlabel('Python', fontsize=9)
                        
                        # Y-axis label for first column plots (R data)
                        if j == 0 and i == len(SETS) -1: # Add "R" label to bottom-left y-axis
                             # The main Y-label is set_name. This adds context to the axis values.
                             pass # ax.set_ylabel(f"{set_name.upper()}\nR", fontsize=11 ...) - can be too busy

                        # Configure ticks and labels
                        ax.tick_params(axis='both', which='major', labelsize=7)
                        ax.xaxis.set_major_locator(plt.MaxNLocator(3, prune='both'))
                        ax.yaxis.set_major_locator(plt.MaxNLocator(3, prune='both'))

                        if i < len(SETS) - 1:  # Remove x-tick labels for non-bottom plots
                            ax.set_xticklabels([])
                        if j > 0:  # Remove y-tick labels for non-first-column plots
                            ax.set_yticklabels([])

                        nc_var_key = f"{set_name}_{var_name}_{period_suffix}"

                        try:
                            py_data = py_ds.variables[nc_var_key][:]
                            r_data = r_ds.variables[nc_var_key][:]

                            if len(py_data) == 0 or len(r_data) == 0:
                                msg = 'Empty data'
                                print(f"Warning: {msg} for {nc_var_key}. Skipping plot.")
                                ax.text(0.5, 0.5, msg, ha='center', va='center', color='orange', fontsize=8)
                                ax.set_xticks([])
                                ax.set_yticks([])
                                continue
                            
                            if len(py_data) != len(r_data):
                                msg = f'Len mismatch:\nPy({len(py_data)}) vs R({len(r_data)})'
                                print(f"Warning: {msg} for {nc_var_key}. Skipping plot.")
                                ax.text(0.5, 0.5, msg, ha='center', va='center', color='brown', fontsize=7)
                                ax.set_xticks([])
                                ax.set_yticks([])
                                continue

                            # Perform linear regression
                            slope, intercept, r_value, p_value, std_err = linregress(py_data, r_data)
                            r_squared = r_value**2

                            # Scatter plot
                            ax.scatter(py_data, r_data, alpha=0.4, s=5)
                            
                            # Regression line
                            line_x_min = np.min(py_data)
                            line_x_max = np.max(py_data)
                            if line_x_min == line_x_max: # Handle case of single point data
                                line_x = np.array([line_x_min - 0.1*abs(line_x_min) if line_x_min != 0 else -0.1, 
                                                   line_x_max + 0.1*abs(line_x_max) if line_x_max != 0 else 0.1])
                                if line_x[0] == line_x[1]: line_x = np.array([line_x_min-1, line_x_max+1]) # if still same (e.g. zero)
                            else:
                                line_x = np.array([line_x_min, line_x_max])
                            
                            line_y = intercept + slope * line_x
                            ax.plot(line_x, line_y, color='red', linestyle='--', linewidth=1)
                            
                            # Annotate with R-squared
                            ax.text(0.05, 0.95, f'$R^2 = {r_squared:.2f}$', 
                                    transform=ax.transAxes, fontsize=8, va='top',
                                    bbox=dict(boxstyle='round,pad=0.2', fc='wheat', alpha=0.6))

                        except KeyError:
                            print(f"Warning: Variable {nc_var_key} not found. Skipping plot.")
                            ax.text(0.5, 0.5, 'Data not found', ha='center', va='center', color='red', fontsize=8)
                            ax.set_xticks([])
                            ax.set_yticks([])
                            continue
                        except Exception as e_plot:
                            print(f"Error plotting {nc_var_key}: {e_plot}")
                            ax.text(0.5, 0.5, 'Plot error', ha='center', va='center', color='purple', fontsize=8)
                            ax.set_xticks([])
                            ax.set_yticks([])
                            continue

        except FileNotFoundError:
            print(f"Error: One or both NetCDF files not found. Searched for '{PYTHON_NC_FILE}' and '{R_NC_FILE}'.")
            fig.clf() 
            ax_err = fig.add_subplot(111)
            ax_err.text(0.5, 0.5, f"Error: NetCDF file(s) not found.\nCannot generate plots for {period_name} period.",
                    ha='center', va='center', color='red', fontsize=12)
            ax_err.set_xticks([])
            ax_err.set_yticks([])
        except Exception as e_outer:
            print(f"An unexpected error occurred while processing {period_name} period: {e_outer}")
            fig.clf()
            ax_err = fig.add_subplot(111)
            ax_err.text(0.5, 0.5, f"An unexpected error occurred:\n{e_outer}\nCannot generate plots for {period_name} period.",
                    ha='center', va='center', color='red', fontsize=12)
            ax_err.set_xticks([])
            ax_err.set_yticks([])

        plt.tight_layout(rect=[0, 0.02, 1, 0.96]) # Adjust rect to make space for suptitle and bottom labels
        output_filename = f'comparison_r2py_vs_R_{period_name}.png'
        plt.savefig(output_filename)
        print(f"Saved plot: {output_filename}")
        plt.close(fig)

if __name__ == '__main__':
    main()
