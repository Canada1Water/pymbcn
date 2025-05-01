import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr
import netCDF4
from mbc_qdm import QDM, MRS, escore, MBCr, MBCp, MBCn

# --- Configuration ---
nc_file_path = 'cccma_output.nc'
variable_names = ['pr', 'tas', 'dtr', 'sfcWind', 'ps', 'huss', 'rsds', 'rlds']
n_vars = len(variable_names)

# --- Load Data from NetCDF ---
try:
    with netCDF4.Dataset(nc_file_path, 'r') as nc_file:
        # Read time lengths
        time_c_len = len(nc_file.dimensions['time_c'])
        time_p_len = len(nc_file.dimensions['time_p'])
        # Use the shorter control period length for alignment
        analysis_len = time_c_len

        # Initialize arrays
        gcm_c_data = np.zeros((analysis_len, n_vars))
        gcm_p_data = np.zeros((analysis_len, n_vars))
        rcm_c_data = np.zeros((analysis_len, n_vars))
        rcm_p_data = np.zeros((analysis_len, n_vars)) # Added for escore

        # Read data for each variable
        for i, var in enumerate(variable_names):
            gcm_c_data[:, i] = nc_file.variables[f'gcm_c_{var}'][:analysis_len]
            # Truncate projection data to control length
            gcm_p_data[:, i] = nc_file.variables[f'gcm_p_{var}'][:analysis_len]
            rcm_c_data[:, i] = nc_file.variables[f'rcm_c_{var}'][:analysis_len]
            rcm_p_data[:, i] = nc_file.variables[f'rcm_p_{var}'][:analysis_len] # Added for escore

        # Read metadata
        ratio_seq = nc_file.variables['ratio_seq'][:].astype(bool)
        trace = nc_file.variables['trace'][:]

except FileNotFoundError:
    print(f"Error: NetCDF file not found at {nc_file_path}")
    print("Please ensure the file exists and the path is correct.")
    # Optionally: Fallback to random data or exit
    np.random.seed(1)
    analysis_len = 100 # Example length if file not found
    gcm_c_data = np.random.rand(analysis_len, n_vars)
    gcm_p_data = np.random.rand(analysis_len, n_vars)
    rcm_c_data = np.random.rand(analysis_len, n_vars)
    rcm_p_data = np.random.rand(analysis_len, n_vars)
    ratio_seq = np.array([True] + [False]*(n_vars-1)) # Example ratio_seq
    trace = np.array([0.05]*n_vars) # Example trace
    print("Using random placeholder data instead.")


# --- Univariate Quantile Mapping (Optional - for comparison/debugging) ---
qdm_c = np.zeros_like(gcm_c_data)
qdm_p = np.zeros_like(gcm_p_data)

for i in range(n_vars):
    fit_qdm = QDM(o_c=rcm_c_data[:, i], m_c=gcm_c_data[:, i],
                 m_p=gcm_p_data[:, i], ratio=ratio_seq[i],
                 trace=trace[i], pp_type='linear')
    qdm_c[:, i] = fit_qdm['mhat_c']
    qdm_p[:, i] = fit_qdm['mhat_p']

# --- Multivariate Bias Corrections ---
print("Running MBCp...")
fit_mbcp = MBCp(o_c=rcm_c_data, m_c=gcm_c_data, m_p=gcm_p_data,
               ratio_seq=ratio_seq, trace=trace, silent=False)
mbcp_c = fit_mbcp['mhat_c']
mbcp_p = fit_mbcp['mhat_p']
print("MBCp finished.")

print("\nRunning MBCr...")
fit_mbcr = MBCr(o_c=rcm_c_data, m_c=gcm_c_data, m_p=gcm_p_data,
               ratio_seq=ratio_seq, trace=trace, silent=False)
mbcr_c = fit_mbcr['mhat_c']
mbcr_p = fit_mbcr['mhat_p']
print("MBCr finished.")

print("\nRunning MBCn...")
fit_mbcn = MBCn(o_c=rcm_c_data, m_c=gcm_c_data, m_p=gcm_p_data,
               ratio_seq=ratio_seq, trace=trace, silent=False, n_escore=100) # Added n_escore example
mbcn_c = fit_mbcn['mhat_c']
mbcn_p = fit_mbcn['mhat_p']
print("MBCn finished.\n")


# --- Analysis Functions ---

# Correlation matrices (Pearson and Spearman)
def plot_correlations(obs, model, title, var_names):
    n_vars = obs.shape[1]
    if n_vars != model.shape[1]:
        raise ValueError("Observation and model must have the same number of variables")

    plt.figure(figsize=(12, 6))

    # Pearson correlation
    plt.subplot(1, 2, 1)
    # Suppress invalid divide warning during correlation calculation
    with np.errstate(invalid='ignore'):
        obs_pearson = np.corrcoef(obs, rowvar=False)
        model_pearson = np.corrcoef(model, rowvar=False)
    # Handle potential NaNs from zero variance
    obs_pearson = np.nan_to_num(obs_pearson)
    model_pearson = np.nan_to_num(model_pearson)
    plt.scatter(obs_pearson.ravel(), model_pearson.ravel(), c='black', alpha=0.5)
    plt.plot([-1, 1], [-1, 1], 'k--')
    plt.title(f'Pearson correlation\n{title}')
    plt.xlabel('RCM Control (Observed)')
    plt.ylabel('Bias Corrected GCM (Model)')
    plt.grid(True)
    plt.axis('equal')
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)

    # Spearman correlation
    plt.subplot(1, 2, 2)
    # spearmanr handles zero variance internally, no errstate needed here
    obs_spearman, _ = spearmanr(obs)
    model_spearman, _ = spearmanr(model)
    # Handle potential NaNs if spearmanr fails for some reason (though less likely)
    if isinstance(obs_spearman, np.ndarray):
        obs_spearman = np.nan_to_num(obs_spearman)
    if isinstance(model_spearman, np.ndarray):
        model_spearman = np.nan_to_num(model_spearman)
    plt.scatter(obs_spearman.ravel(), model_spearman.ravel(), c='black', alpha=0.5)
    plt.plot([-1, 1], [-1, 1], 'k--')
    plt.title(f'Spearman correlation\n{title}')
    plt.xlabel('RCM Control (Observed)')
    plt.ylabel('Bias Corrected GCM (Model)')
    plt.grid(True)
    plt.axis('equal')
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)

    plt.tight_layout()
    # plt.show() # Commented out to avoid showing plots automatically
    plt.savefig(f"{title.replace(' ', '_').lower()}_correlations.png")
    plt.close()

# Pairwise scatterplots
def plot_pairs(data, title, var_names, diagonal='kde'): # Added diagonal argument
    df = pd.DataFrame(data, columns=var_names)
    fig = plt.figure(figsize=(12, 12)) # Adjusted size
    # Use the specified diagonal type ('kde' or 'hist')
    axes = pd.plotting.scatter_matrix(df, diagonal=diagonal, alpha=0.3) # Get axes grid
    plt.suptitle(title)

    # Adjust 'huss' axis limits to be non-negative
    try:
        huss_idx = var_names.index('huss')
        n_vars = len(var_names)
        for i in range(n_vars):
            for j in range(n_vars):
                ax = axes[i, j]
                if i == huss_idx: # Adjust y-axis limits for huss row
                    current_ylim = ax.get_ylim()
                    # Ensure lower bound is >= 0 and upper bound is <= 0.015
                    ax.set_ylim(bottom=max(0, current_ylim[0]), top=min(0.015, current_ylim[1]))
                if j == huss_idx: # Adjust x-axis limits for huss column
                    current_xlim = ax.get_xlim()
                    # Ensure lower bound is >= 0 and upper bound is <= 0.015
                    ax.set_xlim(left=max(0, current_xlim[0]), right=min(0.015, current_xlim[1]))
    except ValueError:
        print("Warning: 'huss' variable not found, skipping axis limit adjustment.")

    # plt.show() # Commented out
    plt.savefig(f"{title.replace(' ', '_').lower()}_pairs.png")
    plt.close()

# --- Plotting Results ---
print("Generating plots...")

# Plot correlations for control period
plot_correlations(rcm_c_data, gcm_c_data, 'Raw GCM Control', variable_names)
plot_correlations(rcm_c_data, qdm_c, 'QDM Control', variable_names)
plot_correlations(rcm_c_data, mbcp_c, 'MBCp Control', variable_names)
plot_correlations(rcm_c_data, mbcr_c, 'MBCr Control', variable_names)
plot_correlations(rcm_c_data, mbcn_c, 'MBCn Control', variable_names)

# Plot pairwise scatterplots for control period
plot_pairs(rcm_c_data, 'RCM Control (Observed)', variable_names)
plot_pairs(gcm_c_data, 'Raw GCM Control', variable_names)
plot_pairs(qdm_c, 'QDM Control', variable_names)
plot_pairs(mbcp_c, 'MBCp Control', variable_names)
# Use histogram for MBCr diagonal due to potential zero variance after rank mapping
plot_pairs(mbcr_c, 'MBCr Control', variable_names, diagonal='hist')
plot_pairs(mbcn_c, 'MBCn Control', variable_names)

print("Plots saved.")

# --- Energy Distance Skill Score (Projection Period) ---
# Using rcm_p_data as the reference for the projection period
print("\nCalculating Energy Scores (using RCM projection as reference)...")

print("Reference (RCM_P) vs QDM_P shapes:", rcm_p_data.shape, qdm_p.shape)
escore_qdm = escore(rcm_p_data, qdm_p, scale_x=True)

print("Reference (RCM_P) vs MBCp_P shapes:", rcm_p_data.shape, mbcp_p.shape)
escore_mbcp = escore(rcm_p_data, mbcp_p, scale_x=True)

print("Reference (RCM_P) vs MBCr_P shapes:", rcm_p_data.shape, mbcr_p.shape)
# print("MBCr values:", mbcr_p[:5])  # Print first few rows (optional)
escore_mbcr = escore(rcm_p_data, mbcr_p, scale_x=True)

print("Reference (RCM_P) vs MBCn_P shapes:", rcm_p_data.shape, mbcn_p.shape)
escore_mbcn = escore(rcm_p_data, mbcn_p, scale_x=True)

print(f'\nESS Base (QDM vs RCM_P): {escore_qdm:.4f}')
if escore_qdm != 0:
    print(f'ESS Improvement (MBCp vs QDM): {1 - escore_mbcp / escore_qdm:.4f}')
    print(f'ESS Improvement (MBCr vs QDM): {1 - escore_mbcr / escore_qdm:.4f}')
    print(f'ESS Improvement (MBCn vs QDM): {1 - escore_mbcn / escore_qdm:.4f}')
else:
    print("Cannot calculate ESS improvement because base QDM score is zero.")

print("\nMBCn Iteration Energy Scores:")
print(fit_mbcn['escore_iter'])

print("\nScript finished.")
