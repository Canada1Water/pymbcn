import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr
import netCDF4
from mbc_qdm import QDM, MRS, escore, MBCr, MBCp, MBCn

# --- Global Random Seed for Reproducibility ---
np.random.seed(1)

# --- Configuration ---
nc_file_path = 'cccma_output.nc'
variable_names = ['pr', 'tas', 'dtr', 'sfcWind', 'ps', 'huss', 'rsds', 'rlds']
n_vars = len(variable_names)

# --- Load Data from NetCDF ---
try:
    with netCDF4.Dataset(nc_file_path, 'r') as nc_file:
        # Read time lengths
        time_c_len = len(nc_file.dimensions['time_c'])
        time_p_len = len(nc_file.dimensions['time_p']) # Use full length for projection

        # Initialize arrays
        # Control period data uses time_c_len
        gcm_c_data = np.zeros((time_c_len, n_vars))
        rcm_c_data = np.zeros((time_c_len, n_vars))
        
        # Projection period data uses time_p_len
        gcm_p_data = np.zeros((time_p_len, n_vars))
        rcm_p_data = np.zeros((time_p_len, n_vars)) 

        # Read data for each variable
        for i, var in enumerate(variable_names):
            gcm_c_data[:, i] = nc_file.variables[f'gcm_c_{var}'][:time_c_len]
            rcm_c_data[:, i] = nc_file.variables[f'rcm_c_{var}'][:time_c_len]
            
            gcm_p_data[:, i] = nc_file.variables[f'gcm_p_{var}'][:time_p_len] 
            rcm_p_data[:, i] = nc_file.variables[f'rcm_p_{var}'][:time_p_len] 

        # Read metadata
        ratio_seq_nc = nc_file.variables['ratio_seq'][:].astype(bool)
        trace_nc = nc_file.variables['trace'][:]

except FileNotFoundError:
    print(f"Error: NetCDF file not found at {nc_file_path}")
    print("Using random placeholder data instead.")
    time_c_len = 100 
    time_p_len = 110 # Example different length for projection
    gcm_c_data = np.random.rand(time_c_len, n_vars)
    rcm_c_data = np.random.rand(time_c_len, n_vars)
    gcm_p_data = np.random.rand(time_p_len, n_vars)
    rcm_p_data = np.random.rand(time_p_len, n_vars)
    ratio_seq_nc = np.array([True] + [False]*(n_vars-1)) 
    trace_nc = np.array([0.05]*n_vars) 


# --- Prepare parameters for MBC functions (as lists/arrays per variable) ---
py_ratio_seq = list(ratio_seq_nc)
py_trace_val = list(trace_nc)
py_jitter_factor = [0.0] * n_vars 

# --- Univariate Quantile Mapping ---
# qdm_c will have time_c_len rows
# qdm_p will have time_p_len rows
qdm_c = np.zeros_like(gcm_c_data) 
qdm_p = np.zeros_like(gcm_p_data) 

print("Running Univariate QDM...")
for i in range(n_vars):
    current_debug_name_py = None
    # Determine if this is the first ratio variable (precipitation)
    # np.where(py_ratio_seq)[0] gives indices of TRUE values
    # We want to debug if py_ratio_seq[i] is TRUE and i is the first such index
    first_ratio_var_idx = -1
    if np.any(py_ratio_seq):
        first_ratio_var_idx = np.where(py_ratio_seq)[0][0]
    
    if py_ratio_seq[i] and i == first_ratio_var_idx:
        current_debug_name_py = "pr_initial_qdm_mp_debug"
    if variable_names[i] == "huss": # Check for huss
        current_debug_name_py = "huss_qdm_debug"

    # QDM for control period:
    # Need a debug name for huss's mhat_c
    debug_name_for_mhat_c = None
    if variable_names[i] == "huss":
        debug_name_for_mhat_c = "huss_qdm_mhat_c_debug"

    fit_qdm_c = QDM(o_c=rcm_c_data[:, i], m_c=gcm_c_data[:, i],
                   m_p=gcm_c_data[:, i],  # m_p is gcm_c_data for control period correction
                   ratio=py_ratio_seq[i], 
                   trace=py_trace_val[i], 
                   jitter_factor=0, 
                   ties='first',   
                   pp_type='linear',
                   debug_name=debug_name_for_mhat_c) # Pass the new debug name
    qdm_c[:, i] = fit_qdm_c['mhat_c']

    # QDM for projection period (mhat_p is desired):
    fit_qdm_p = QDM(o_c=rcm_c_data[:, i], m_c=gcm_c_data[:, i],
                   m_p=gcm_p_data[:, i], 
                   ratio=py_ratio_seq[i], 
                   trace=py_trace_val[i], 
                   jitter_factor=0, 
                   ties='first',    
                   pp_type='linear',
                   debug_name=current_debug_name_py) # Pass debug_name for pr's m_p calculation
    qdm_p[:, i] = fit_qdm_p['mhat_p']
    if variable_names[i] == "huss":
       print('qdm[huss]')
       print(qdm_p[:, i])

print("Univariate QDM finished.")

# --- GCM_P vs QDM_P Histograms and Time Series Plots for all variables ---
print("\nGenerating GCM_P vs QDM_P histograms and time series plots for all variables...")
for plot_var_idx, plot_var_name in enumerate(variable_names):
    gcm_p_var_data = gcm_p_data[:, plot_var_idx]
    qdm_p_var_data = qdm_p[:, plot_var_idx]

    # Determine common range and bins for histograms
    combined_data_for_hist = np.concatenate((gcm_p_var_data, qdm_p_var_data))
    min_val = np.nanmin(combined_data_for_hist)
    max_val = np.nanmax(combined_data_for_hist)
    hist_bins = 30
    hist_range = (min_val, max_val)

    # --- Histograms ---
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(gcm_p_var_data, bins=hist_bins, range=hist_range, color='blue', alpha=0.7)
    plt.title(f'Histogram of Original GCM Projection Data for {plot_var_name.upper()}')
    plt.xlabel(f'{plot_var_name.upper()} Value')
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    plt.hist(qdm_p_var_data, bins=hist_bins, range=hist_range, color='green', alpha=0.7)
    plt.title(f'Histogram of QDM Processed Data for {plot_var_name.upper()} (Projection Period)')
    plt.xlabel(f'{plot_var_name.upper()} Value')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.savefig(f"{plot_var_name}_gcm_p_vs_qdm_p_histograms.png")
    plt.close()
    print(f"{plot_var_name.upper()} GCM_P vs QDM_P histograms saved to {plot_var_name}_gcm_p_vs_qdm_p_histograms.png")

    # --- Time Series Plots ---
    plt.figure(figsize=(12, 8))
    time_axis_p = np.arange(gcm_p_var_data.shape[0]) # Simple time index

    plt.subplot(2, 1, 1)
    plt.plot(time_axis_p, gcm_p_var_data, color='blue', alpha=0.7, linewidth=0.8)
    plt.title(f'Time Series of Original GCM Projection Data for {plot_var_name.upper()}')
    plt.xlabel('Time Index')
    plt.ylabel(f'{plot_var_name.upper()} Value')
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(time_axis_p, qdm_p_var_data, color='green', alpha=0.7, linewidth=0.8)
    plt.title(f'Time Series of QDM Processed Data for {plot_var_name.upper()} (Projection Period)')
    plt.xlabel('Time Index')
    plt.ylabel(f'{plot_var_name.upper()} Value')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"{plot_var_name}_gcm_p_vs_qdm_p_timeseries.png")
    plt.close()
    print(f"{plot_var_name.upper()} GCM_P vs QDM_P time series plots saved to {plot_var_name}_gcm_p_vs_qdm_p_timeseries.png")

# --- Multivariate Bias Corrections ---
print("\nRunning MBCp...")
fit_mbcp = MBCp(o_c=rcm_c_data, m_c=gcm_c_data, m_p=gcm_p_data,
               ratio_seq=py_ratio_seq, trace=py_trace_val, 
               jitter_factor=0, 
               ties='first',    
               silent=False, pp_type='linear')
mbcp_c = fit_mbcp['mhat_c']
mbcp_p = fit_mbcp['mhat_p']
print("MBCp finished.")

print("\nRunning MBCr...")
fit_mbcr = MBCr(o_c=rcm_c_data, m_c=gcm_c_data, m_p=gcm_p_data,
               ratio_seq=py_ratio_seq, trace=py_trace_val,
               jitter_factor=0, 
               ties='first',    
               silent=False, pp_type='linear')
mbcr_c = fit_mbcr['mhat_c']
mbcr_p = fit_mbcr['mhat_p']
print("MBCr finished.")

print("\nRunning MBCn...")
fit_mbcn = MBCn(o_c=rcm_c_data, m_c=gcm_c_data, m_p=gcm_p_data,
               ratio_seq=py_ratio_seq, trace=py_trace_val,
               jitter_factor=0, 
               ties='first',    
               silent=False, n_escore=100, # Changed n_escore to 100
               pp_type='linear') 
mbcn_c = fit_mbcn['mhat_c']
mbcn_p = fit_mbcn['mhat_p']
print("MBCn finished.\n")

# --- Huss Data Summaries for Calibration Plots ---
huss_col_idx = variable_names.index('huss') if 'huss' in variable_names else -1
if huss_col_idx != -1:
    print("--- Python Huss Summaries for Calibration Data ---")
    print(f"Summary of qdm_c[:, 'huss']:\n{pd.Series(qdm_c[:, huss_col_idx]).describe()}")
    print(f"Min: {np.min(qdm_c[:, huss_col_idx]):.6e}, Max: {np.max(qdm_c[:, huss_col_idx]):.6e}, Mean: {np.mean(qdm_c[:, huss_col_idx]):.6e}\n")

    print(f"Summary of mbcp_c[:, 'huss']:\n{pd.Series(mbcp_c[:, huss_col_idx]).describe()}")
    print(f"Min: {np.min(mbcp_c[:, huss_col_idx]):.6e}, Max: {np.max(mbcp_c[:, huss_col_idx]):.6e}, Mean: {np.mean(mbcp_c[:, huss_col_idx]):.6e}\n")

    print(f"Summary of mbcr_c[:, 'huss']:\n{pd.Series(mbcr_c[:, huss_col_idx]).describe()}")
    print(f"Min: {np.min(mbcr_c[:, huss_col_idx]):.6e}, Max: {np.max(mbcr_c[:, huss_col_idx]):.6e}, Mean: {np.mean(mbcr_c[:, huss_col_idx]):.6e}\n")

    print(f"Summary of mbcn_c[:, 'huss']:\n{pd.Series(mbcn_c[:, huss_col_idx]).describe()}")
    print(f"Min: {np.min(mbcn_c[:, huss_col_idx]):.6e}, Max: {np.max(mbcn_c[:, huss_col_idx]):.6e}, Mean: {np.mean(mbcn_c[:, huss_col_idx]):.6e}\n")
else:
    print("Huss variable not found for summary printing.")


# --- Analysis Functions ---
def plot_correlations(obs, model, title_prefix, var_names, period_label):
    # obs_data, model_data, title_suffix (e.g. "calibration" or "evaluation")
    # For calibration: obs=rcm.c, model=corrected gcm.c
    # For evaluation: obs=rcm.p, model=corrected gcm.p
    
    n_vars_plot = obs.shape[1]
    if n_vars_plot != model.shape[1]:
        raise ValueError("Observation and model must have the same number of variables for correlation plot")

    full_title = f'{title_prefix} {period_label}'
    plt.figure(figsize=(12, 6))

    # Pearson correlation
    plt.subplot(1, 2, 1)
    with np.errstate(invalid='ignore', divide='ignore'): 
        obs_pearson = np.corrcoef(obs, rowvar=False) 
        model_pearson = np.corrcoef(model, rowvar=False)
    obs_pearson = np.nan_to_num(obs_pearson)
    model_pearson = np.nan_to_num(model_pearson)
    plt.scatter(obs_pearson.ravel(), model_pearson.ravel(), c='black', alpha=0.5, label='Corrected')
    # Add QDM points if qdm_model is provided (for comparison)
    # This part is tricky as the R code plots QDM vs MBC, not obs vs QDM then obs vs MBC.
    # R: plot(c(cor(cccma$rcm.c)), c(cor(qdm.c))) ... points(c(cor(cccma$rcm.c)), c(cor(mbcp.c)))
    # So, x-axis is always cor(obs_ref), y-axis is cor(model_type_A) then cor(model_type_B)
    # This function is structured for obs_ref vs model_type_X.
    # To replicate R plot: call this with (obs_ref, qdm_model) then add points from (obs_ref, mbc_model)
    # For now, this plots obs_ref vs model.
    plt.plot([-1, 1], [-1, 1], 'k--')
    plt.title(f'Pearson correlation\n{full_title}')
    plt.xlabel('CanRCM4 (Observed Reference)') # R: 'CanRCM4'
    plt.ylabel('CanESM2 Corrected (Model)')   # R: 'CanESM2 MBC<method>'
    plt.grid(True); plt.axis('equal'); plt.xlim(-1, 1); plt.ylim(-1, 1)

    # Spearman correlation
    plt.subplot(1, 2, 2)
    try:
        obs_spearman_matrix, _ = spearmanr(obs, axis=0, nan_policy='propagate') # axis=0 for column-wise
        if obs.shape[1] == 1: obs_spearman_matrix = np.array([[1.0]]) # spearmanr returns scalar for 1 var
        elif np.isscalar(obs_spearman_matrix): # Should be matrix if n_vars > 1
            obs_spearman_matrix = np.diag(np.full(n_vars_plot, obs_spearman_matrix))


        model_spearman_matrix, _ = spearmanr(model, axis=0, nan_policy='propagate')
        if model.shape[1] == 1: model_spearman_matrix = np.array([[1.0]])
        elif np.isscalar(model_spearman_matrix):
             model_spearman_matrix = np.diag(np.full(n_vars_plot, model_spearman_matrix))

    except Exception as e:
        print(f"Warning: Spearman calculation failed for {full_title}: {e}")
        obs_spearman_matrix = np.full((n_vars_plot, n_vars_plot), np.nan)
        model_spearman_matrix = np.full((n_vars_plot, n_vars_plot), np.nan)

    obs_spearman = np.nan_to_num(obs_spearman_matrix)
    model_spearman = np.nan_to_num(model_spearman_matrix)
    plt.scatter(obs_spearman.ravel(), model_spearman.ravel(), c='black', alpha=0.5)
    plt.plot([-1, 1], [-1, 1], 'k--')
    plt.title(f'Spearman correlation\n{full_title}')
    plt.xlabel('CanRCM4 (Observed Reference)')
    plt.ylabel('CanESM2 Corrected (Model)')
    plt.grid(True); plt.axis('equal'); plt.xlim(-1, 1); plt.ylim(-1, 1)

    plt.tight_layout()
    plt.savefig(f"{title_prefix.replace(' ', '_').lower()}_{period_label.lower()}_correlations.png")
    plt.close()

def plot_r_style_correlations(obs_ref_c, obs_ref_p, qdm_c_data, qdm_p_data, mbc_c_data, mbc_p_data, mbc_method_name):
    """Replicates the R script's correlation plot structure."""
    plt.figure(figsize=(12, 12)) # R uses par(mfrow=c(2,2))

    # Pearson Calibration
    plt.subplot(2, 2, 1)
    with np.errstate(invalid='ignore', divide='ignore'):
        cor_obs_ref_c = np.corrcoef(obs_ref_c, rowvar=False)
        cor_qdm_c = np.corrcoef(qdm_c_data, rowvar=False)
        cor_mbc_c = np.corrcoef(mbc_c_data, rowvar=False)
    plt.scatter(np.nan_to_num(cor_obs_ref_c).ravel(), np.nan_to_num(cor_qdm_c).ravel(), color='black', marker='o', label='QDM')
    plt.scatter(np.nan_to_num(cor_obs_ref_c).ravel(), np.nan_to_num(cor_mbc_c).ravel(), color='red', marker='x', label=mbc_method_name)
    plt.plot([-1, 1], [-1, 1], 'k--'); plt.grid(True); plt.axis('equal'); plt.xlim(-1, 1); plt.ylim(-1, 1)
    plt.title(f'Pearson correlation\n{mbc_method_name} calibration'); plt.xlabel('CanRCM4'); plt.ylabel(f'CanESM2 {mbc_method_name}')

    # Pearson Evaluation
    plt.subplot(2, 2, 2)
    with np.errstate(invalid='ignore', divide='ignore'):
        cor_obs_ref_p = np.corrcoef(obs_ref_p, rowvar=False)
        cor_qdm_p = np.corrcoef(qdm_p_data, rowvar=False)
        cor_mbc_p = np.corrcoef(mbc_p_data, rowvar=False)
    plt.scatter(np.nan_to_num(cor_obs_ref_p).ravel(), np.nan_to_num(cor_qdm_p).ravel(), color='black', marker='o')
    plt.scatter(np.nan_to_num(cor_obs_ref_p).ravel(), np.nan_to_num(cor_mbc_p).ravel(), color='red', marker='x')
    plt.plot([-1, 1], [-1, 1], 'k--'); plt.grid(True); plt.axis('equal'); plt.xlim(-1, 1); plt.ylim(-1, 1)
    plt.title(f'Pearson correlation\n{mbc_method_name} evaluation'); plt.xlabel('CanRCM4'); plt.ylabel(f'CanESM2 {mbc_method_name}')

    # Spearman Calibration
    plt.subplot(2, 2, 3)
    def get_spearman_cor(data):
        mat, _ = spearmanr(data, axis=0, nan_policy='propagate')
        if data.shape[1] == 1: return np.array([[1.0]])
        if np.isscalar(mat): # Handle case where spearmanr might return scalar for multiple identical columns
            return np.diag(np.full(data.shape[1], mat))
        return np.nan_to_num(mat)
    
    cor_obs_ref_c_s = get_spearman_cor(obs_ref_c)
    cor_qdm_c_s = get_spearman_cor(qdm_c_data)
    cor_mbc_c_s = get_spearman_cor(mbc_c_data)
    plt.scatter(cor_obs_ref_c_s.ravel(), cor_qdm_c_s.ravel(), color='black', marker='o')
    plt.scatter(cor_obs_ref_c_s.ravel(), cor_mbc_c_s.ravel(), color='red', marker='x')
    plt.plot([-1, 1], [-1, 1], 'k--'); plt.grid(True); plt.axis('equal'); plt.xlim(-1, 1); plt.ylim(-1, 1)
    plt.title(f'Spearman correlation\n{mbc_method_name} calibration'); plt.xlabel('CanRCM4'); plt.ylabel(f'CanESM2 {mbc_method_name}')

    # Spearman Evaluation
    plt.subplot(2, 2, 4)
    cor_obs_ref_p_s = get_spearman_cor(obs_ref_p)
    cor_qdm_p_s = get_spearman_cor(qdm_p_data)
    cor_mbc_p_s = get_spearman_cor(mbc_p_data)
    plt.scatter(cor_obs_ref_p_s.ravel(), cor_qdm_p_s.ravel(), color='black', marker='o')
    plt.scatter(cor_obs_ref_p_s.ravel(), cor_mbc_p_s.ravel(), color='red', marker='x')
    plt.plot([-1, 1], [-1, 1], 'k--'); plt.grid(True); plt.axis('equal'); plt.xlim(-1, 1); plt.ylim(-1, 1)
    plt.title(f'Spearman correlation\n{mbc_method_name} evaluation'); plt.xlabel('CanRCM4'); plt.ylabel(f'CanESM2 {mbc_method_name}')

    plt.tight_layout()
    plt.savefig(f"{mbc_method_name.lower()}_r_style_correlations.png")
    plt.close()


def plot_pairs(data, title, var_names_list, diagonal='kde', color_hex='#0000001A'):
    if not np.all(np.isfinite(data)):
        # print(f"Warning: Non-finite values in data for '{title}'. Skipping pair plot.")
        # Create an empty plot or save a message
        fig, ax = plt.subplots(); ax.text(0.5, 0.5, "Pair plot skipped (non-finite data)", ha='center', va='center')
        plt.savefig(f"{title.replace(' ', '_').lower()}_pairs_skipped.png"); plt.close()
        return

    df = pd.DataFrame(data, columns=var_names_list)
    if df.empty:
        # print(f"Warning: DataFrame empty for '{title}'. Skipping pair plot.")
        fig, ax = plt.subplots(); ax.text(0.5, 0.5, "Pair plot skipped (empty data)", ha='center', va='center')
        plt.savefig(f"{title.replace(' ', '_').lower()}_pairs_empty.png"); plt.close()
        return

    fig = plt.figure(figsize=(12, 12)) 
    try:
        # Ensure color for hist_kwds is just the hex color, not with alpha
        hist_color = color_hex[:-2] if len(color_hex) == 9 and color_hex[0] == '#' else color_hex
        
        axes = pd.plotting.scatter_matrix(df, diagonal=diagonal, alpha=float(int(color_hex[-2:], 16))/255.0 if len(color_hex) == 9 else 0.5, 
                                          c=color_hex[:-2] if len(color_hex) == 9 else color_hex, 
                                          s=5, 
                                          hist_kwds={'color': hist_color, 'bins': 20}, # Added bins for better hist
                                          density_kwds={'color': hist_color}) # Color for KDE
        plt.suptitle(title)
        # Axis adjustments (e.g., for 'huss') can be added here if needed
        # Based on R code, no specific axis limits, but huss might need non-negative.
        # Pandas scatter_matrix usually handles ranges well.
        plt.savefig(f"{title.replace(' ', '_').lower()}_pairs.png")
    except Exception as e:
        print(f"Error in scatter_matrix for {title}: {e}")
        fig = plt.figure(figsize=(3,3)) 
        plt.text(0.5, 0.5, f"Error in scatter_matrix for {title}", ha='center', va='center')
        plt.savefig(f"{title.replace(' ', '_').lower()}_pairs_error.png")
    finally:
        plt.close(fig)


# --- Plotting Results ---
print("Generating plots...")

# R-style correlation plots (Obs_Ref vs QDM, Obs_Ref vs MBC_method)
plot_r_style_correlations(rcm_c_data, rcm_p_data, qdm_c, qdm_p, mbcp_c, mbcp_p, "MBCp")
# Similar plots for MBCr and MBCn would be:
plot_r_style_correlations(rcm_c_data, rcm_p_data, qdm_c, qdm_p, mbcr_c, mbcr_p, "MBCr")
plot_r_style_correlations(rcm_c_data, rcm_p_data, qdm_c, qdm_p, mbcn_c, mbcn_p, "MBCn")


# Pairwise scatterplots (R colors: #0000001A, #FF00001A, #0000FF1A, #FFA5001A)
plot_pairs(gcm_c_data, 'CanESM2 calibration', variable_names, color_hex='#0000003A') # Darker alpha for visibility
plot_pairs(rcm_c_data, 'CanRCM4 calibration', variable_names, color_hex='#0000003A')
plot_pairs(qdm_c,      'QDM calibration',     variable_names, color_hex='#0000003A', diagonal='hist') # Changed diagonal to hist
plot_pairs(mbcp_c,     'MBCp calibration',    variable_names, color_hex='#FF00003A', diagonal='hist') # MBCp often results in spiky KDEs
plot_pairs(mbcr_c,     'MBCr calibration',    variable_names, color_hex='#0000FF3A', diagonal='hist') # MBCr uses ranks, hist is better
plot_pairs(mbcn_c,     'MBCn calibration',    variable_names, color_hex='#FFA5003A', diagonal='hist') # MBCn can also be spiky

print("Plots saved.")

# --- Energy Distance Skill Score (Projection Period) ---
# R: escore(cccma$rcm.p, qdm.p, scale.x = TRUE)
# Python: escore(rcm_p_data, qdm_p, scale_x=True)
# This uses rcm_p_data as the "observation" for the projection period.
print("\nCalculating Energy Scores (using RCM projection as reference)...")

# Ensure all inputs to escore are 2D
def ensure_2d(arr):
    return arr if arr.ndim == 2 else arr.reshape(-1, 1)

# Use a small subset for energy score calculation to match R behavior
n_escore_sample = 100  # Similar to what R might use internally
escore_qdm_p = escore(ensure_2d(rcm_p_data), ensure_2d(qdm_p), scale_x=True, n_cases=n_escore_sample)
escore_mbcp_p = escore(ensure_2d(rcm_p_data), ensure_2d(mbcp_p), scale_x=True, n_cases=n_escore_sample)
escore_mbcr_p = escore(ensure_2d(rcm_p_data), ensure_2d(mbcr_p), scale_x=True, n_cases=n_escore_sample)
escore_mbcn_p = escore(ensure_2d(rcm_p_data), ensure_2d(mbcn_p), scale_x=True, n_cases=n_escore_sample)

print(f'\nRaw Escore QDM (vs RCM_P): {escore_qdm_p:.6f}') # R output has no label for this
# R: cat('ESS (MBCp):', 1 - escore.mbcp / escore.qdm, '\n')
if escore_qdm_p != 0 and not np.isnan(escore_qdm_p):
    ess_mbcp = 1 - escore_mbcp_p / escore_qdm_p if not np.isnan(escore_mbcp_p) else np.nan
    ess_mbcr = 1 - escore_mbcr_p / escore_qdm_p if not np.isnan(escore_mbcr_p) else np.nan
    ess_mbcn = 1 - escore_mbcn_p / escore_qdm_p if not np.isnan(escore_mbcn_p) else np.nan
    print(f'ESS (MBCp): {ess_mbcp:.6f}')
    print(f'ESS (MBCr): {ess_mbcr:.6f}')
    print(f'ESS (MBCn): {ess_mbcn:.6f}')
else:
    print("Cannot calculate ESS improvement because base QDM escore is zero or NaN.")
    print(f'Raw escore_mbcp_p: {escore_mbcp_p:.6f}')
    print(f'Raw escore_mbcr_p: {escore_mbcr_p:.6f}')
    print(f'Raw escore_mbcn_p: {escore_mbcn_p:.6f}')


print("\nMBCn Iteration Energy Scores (if calculated):")
if 'escore_iter' in fit_mbcn and isinstance(fit_mbcn['escore_iter'], dict):
    # Filter out NaN scores for cleaner printing if n_escore was 0 for RAW/QM
    filtered_scores = {k: v for k, v in fit_mbcn['escore_iter'].items() if not (isinstance(v, float) and np.isnan(v))}
    if filtered_scores:
        for k, v_score in filtered_scores.items():
            print(f"  {k}: {v_score:.6g}" if isinstance(v_score, (float, np.floating)) else f"  {k}: {v_score}")
    else:
        print("  (No energy scores were calculated or all were NaN)")
else:
    print("  (No iteration energy scores available in fit_mbcn)")


print("\nScript finished.")
