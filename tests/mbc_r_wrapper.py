import os
import time
import numpy as np
import netCDF4
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
import subprocess
import sys

# --- Global Configuration (mirroring testcode.py) ---
VARIABLE_NAMES = ['pr', 'tas', 'dtr', 'sfcWind', 'ps', 'huss', 'rsds', 'rlds']
N_ITER_MBCN = 30
N_ITER_MBCPR = 20
RANDOM_SEED = 1

# Initialize R interface
numpy2ri.activate()
base = importr('base')
utils = importr('utils')

def install_r_packages():
    """Install required R packages"""
    print("Checking/installing R packages...")
    # Set custom library path and suppress warnings
    r_lib_path = '/home/guido/R/x86_64-pc-linux-gnu-library/4.4'
    
    # First verify R library path exists
    if not os.path.exists(r_lib_path):
        print(f"Creating R library directory: {r_lib_path}")
        os.makedirs(r_lib_path, exist_ok=True)
    
    # Set library path and suppress all messages/warnings
    ro.r(f'''
    options(warn=-1)
    suppressPackageStartupMessages({{
        suppressMessages({{
            if(!dir.exists("{r_lib_path}")) dir.create("{r_lib_path}", recursive=TRUE)
            # Only keep existing library paths that contain packages
            current_paths <- .libPaths()
            valid_paths <- current_paths[sapply(current_paths, function(p) length(list.files(p)) > 0)]
            .libPaths(c("{r_lib_path}", valid_paths))
        }})
    }})
    options(warn=0)
    ''')
    
    # Check and install packages with better error handling
    required_pkgs = ['MBC', 'Matrix', 'energy', 'FNN']
    
    for pkg in required_pkgs:
        try:
            # Check if package is installed by trying to import it
            try:
                importr(pkg)
                print(f"{pkg} is already installed")
                continue
            except:
                print(f"Installing {pkg}...")
                utils.install_packages(pkg, lib=r_lib_path, repos="https://cloud.r-project.org")
                print(f"Installing {pkg}...")
                utils.install_packages(pkg, lib=r_lib_path, repos="https://cloud.r-project.org")
                # Verify installation
                if pkg not in ro.packages.installed_packages():
                    raise RuntimeError(f"Failed to verify {pkg} installation")
                print(f"Successfully installed {pkg}")
            else:
                print(f"{pkg} is already installed")
        except Exception as e:
            print(f"Critical error installing {pkg}: {str(e)}")
            sys.exit(1)

def load_netcdf_data(nc_file_path):
    """Load data from NetCDF file into numpy arrays using a fixed variable order."""
    print(f"Loading data from {nc_file_path} using fixed variable order: {VARIABLE_NAMES}...")
    n_vars = len(VARIABLE_NAMES)
    
    with netCDF4.Dataset(nc_file_path) as nc:
        time_c_len = len(nc.dimensions['time_c'])
        time_p_len = len(nc.dimensions['time_p'])

        gcm_c = np.zeros((time_c_len, n_vars))
        rcm_c = np.zeros((time_c_len, n_vars))
        gcm_p = np.zeros((time_p_len, n_vars))
        
        var_units = {}

        for i, var_name in enumerate(VARIABLE_NAMES):
            try:
                gcm_c[:, i] = nc.variables[f'gcm_c_{var_name}'][:time_c_len]
                rcm_c[:, i] = nc.variables[f'rcm_c_{var_name}'][:time_c_len]
                gcm_p[:, i] = nc.variables[f'gcm_p_{var_name}'][:time_p_len]
                var_units[var_name] = nc.variables[f'gcm_c_{var_name}'].units
            except KeyError:
                print(f"Warning: Variable {var_name} not found in NetCDF. Filling with NaNs.")
                gcm_c[:, i] = np.nan
                rcm_c[:, i] = np.nan
                gcm_p[:, i] = np.nan
                var_units[var_name] = 'unknown'
            except AttributeError: # For units
                 var_units[var_name] = 'unknown'
        
        # Get metadata - assuming their order in NetCDF matches VARIABLE_NAMES
        # Or, if they are named per variable, load them accordingly.
        # For now, assume they are 1D arrays matching VARIABLE_NAMES order.
        try:
            ratio_seq = np.ma.filled(nc.variables['ratio_seq'][:], False).astype(bool)
            trace = np.ma.filled(nc.variables['trace'][:], np.nan)
            if len(ratio_seq) != n_vars or len(trace) != n_vars:
                raise ValueError("Metadata array lengths do not match number of variables.")
        except Exception as e:
            print(f"Error loading metadata (ratio_seq, trace): {e}. Using defaults.")
            ratio_seq = np.array([False]*n_vars) # Default: no ratio variables
            trace = np.array([0.05]*n_vars)    # Default: 0.05 trace for all

    return {
        'gcm_c': gcm_c,
        'rcm_c': rcm_c,
        'gcm_p': gcm_p,
        'var_names': list(VARIABLE_NAMES), # Pass a copy
        'ratio_seq': ratio_seq,
        'trace': trace,
        'var_units': var_units
    }

def run_mbc_methods(data):
    """Run MBC methods using rpy2 interface"""
    print("Running MBC methods through R...")
    mbc = importr('MBC')
    r = ro.r
    
    # Validate and convert numpy arrays to R matrices
    print("Validating input data...")
    for arr_name in ['gcm_c', 'rcm_c', 'gcm_p']:
        arr = data[arr_name]
        # Ensure proper float64 type
        if not np.issubdtype(arr.dtype, np.floating):
            print(f"Converting {arr_name} to float64")
            arr = arr.astype(np.float64)
        # Check for invalid values
        if np.any(np.isnan(arr)):
            print(f"Warning: {arr_name} contains NaN values")
        if np.any(np.isinf(arr)):
            print(f"Warning: {arr_name} contains Inf values")
        # Ensure 2D shape
        if arr.ndim != 2:
            print(f"Reshaping {arr_name} to 2D")
            arr = arr.reshape(-1, 1)
        data[arr_name] = arr
    
    # Convert to R matrices with explicit type checking
    def to_r_matrix(arr):
        if not isinstance(arr, np.ndarray):
            arr = np.array(arr)
        if arr.ndim != 2:
            arr = arr.reshape(-1, 1)
        return r.matrix(
            arr.astype(np.float64), 
            nrow=arr.shape[0], 
            ncol=arr.shape[1]
        )
    
    gcm_c_r = to_r_matrix(data['gcm_c'])
    rcm_c_r = to_r_matrix(data['rcm_c'])
    gcm_p_r = to_r_matrix(data['gcm_p'])
    
    # Convert parameters to R format
    ratio_seq_r = ro.BoolVector(data['ratio_seq'])
    trace_r = ro.FloatVector(data['trace'])
    trace_calc_r = ro.FloatVector(data['trace'] * 0.5)  # Pre-calculate trace_calc
    
    # Run MBC methods
    results = {}
    
    print("\nRunning QDM for each variable...")
    qdm_c = np.zeros_like(data['gcm_c'])
    qdm_p = np.zeros_like(data['gcm_p'])
    
    # Make a mutable copy of trace values for potential modification
    current_trace_values = list(data['trace']) # Python list of floats

    for i in range(len(data['var_names'])):
        var_name_current = data['var_names'][i]
        print(f"  Variable {i+1}/{len(data['var_names'])}: {var_name_current}")
        
        o_c_col = ro.FloatVector(data['rcm_c'][:, i])
        m_c_col = ro.FloatVector(data['gcm_c'][:, i])
        m_p_col = ro.FloatVector(data['gcm_p'][:, i])
        
        # Initial trace for this variable
        current_trace_for_var = current_trace_values[i]
        current_trace_calc_for_var = 0.5 * current_trace_for_var

        try:
            # print(f"Running QDM for {var_name_current} with trace={current_trace_for_var}, ratio={data['ratio_seq'][i]}")
            qdm_result = mbc.QDM(
                o_c=o_c_col,
                m_c=m_c_col,
                m_p=m_p_col, # This is gcm_p for getting both mhat_c and mhat_p
                ratio=data['ratio_seq'][i], # Use boolean directly
                trace=current_trace_for_var,
                trace_calc=current_trace_calc_for_var,
                jitter_factor=0,
                ties="first",
                pp_type="linear" # Added pp_type
            )
            # print(f"QDM completed for {var_name_current}")
            
            mhat_c_res = qdm_result.rx2('mhat_c')
            mhat_p_res = qdm_result.rx2('mhat_p')
            
            if mhat_c_res == ro.NULL:
                print(f"Warning: QDM mhat_c is NULL for {var_name_current}, using original gcm_c.")
                qdm_c[:, i] = data['gcm_c'][:, i]
            else:
                qdm_c[:, i] = np.asarray(mhat_c_res, dtype=np.float64).reshape(-1)

            if mhat_p_res == ro.NULL:
                print(f"Warning: QDM mhat_p is NULL for {var_name_current}, using original gcm_p.")
                qdm_p[:, i] = data['gcm_p'][:, i]
            else:
                qdm_p[:, i] = np.asarray(mhat_p_res, dtype=np.float64).reshape(-1)

            # Adaptive thresholding for ratio variables (modifies qdm_p and current_trace_values[i])
            if data['ratio_seq'][i]:
                original_gcm_p_series = data['gcm_p'][:, i]
                # qdm_p[:, i] currently holds result from QDM with initial trace
                
                if np.std(original_gcm_p_series) > 1e-9 and np.std(qdm_p[:, i]) > 1e-9:
                    correlation = np.corrcoef(original_gcm_p_series, qdm_p[:, i])[0, 1]
                    if not np.isnan(correlation) and correlation < 0.8:
                        print(f"R WRAPPER UNIQDM LOOP - Var: {var_name_current}, Low correlation ({correlation:.2f}) with original trace {current_trace_for_var:.4f}. Adjusting trace.")
                        adjusted_trace_val = current_trace_for_var * 2.0
                        adjusted_trace_calc_val = 0.5 * adjusted_trace_val
                        print(f"R WRAPPER UNIQDM LOOP - Var: {var_name_current}, New trace: {adjusted_trace_val:.4f}, New trace_calc: {adjusted_trace_calc_val:.4f}")

                        qdm_adjusted_result = mbc.QDM(
                            o_c=o_c_col, m_c=m_c_col, m_p=m_p_col,
                            ratio=data['ratio_seq'][i],
                            trace=adjusted_trace_val,
                            trace_calc=adjusted_trace_calc_val,
                            jitter_factor=0, ties="first", pp_type="linear"
                        )
                        mhat_p_adjusted_res = qdm_adjusted_result.rx2('mhat_p')
                        if mhat_p_adjusted_res != ro.NULL:
                            qdm_p[:, i] = np.asarray(mhat_p_adjusted_res, dtype=np.float64).reshape(-1)
                        
                        current_trace_values[i] = adjusted_trace_val # Update for subsequent MBC calls
                elif np.std(original_gcm_p_series) <= 1e-9 or np.std(qdm_p[:, i]) <= 1e-9:
                     print(f"R WRAPPER UNIQDM LOOP - Var: {var_name_current}, Skipping correlation check due to zero variance in time series.")
            
            if qdm_c[:, i].shape[0] != data['gcm_c'].shape[0] or qdm_p[:, i].shape[0] != data['gcm_p'].shape[0]:
                raise ValueError(f"QDM output length mismatch for {var_name_current}")

        except Exception as e:
            print(f"Error processing QDM for variable {var_name_current}: {str(e)}")
            qdm_c[:, i] = data['gcm_c'][:, i] # Fallback
            qdm_p[:, i] = data['gcm_p'][:, i] # Fallback
    
    results['qdm'] = {'mhat_c': qdm_c, 'mhat_p': qdm_p}
    
    # Convert final trace values (potentially modified) to R vector for MBCp/r/n
    final_trace_r = ro.FloatVector(current_trace_values)

    print("\nRunning MBCp...")
    try:
        mbcp = mbc.MBCp(
            o_c=rcm_c_r,
            m_c=gcm_c_r,
            m_p=gcm_p_r,
            ratio_seq=ratio_seq_r, # Original ratio_seq R vector
            trace=final_trace_r,   # Potentially modified trace R vector
            jitter_factor=0,
            ties="first",
            silent=False,
            pp_type="linear",      # Added pp_type
            iter=N_ITER_MBCPR      # Added iter
        )
        
        # Handle potential NULL returns
        mhat_c = mbcp.rx2('mhat_c')
        mhat_p = mbcp.rx2('mhat_p')
        
        if mhat_c == ro.NULL or mhat_p == ro.NULL:
            print("Warning: MBCp returned NULL values, using original data")
            results['mbcp'] = {
                'mhat_c': data['gcm_c'],
                'mhat_p': data['gcm_p']
            }
        else:
            results['mbcp'] = {
                'mhat_c': np.atleast_1d(np.array(mhat_c)),
                'mhat_p': np.atleast_1d(np.array(mhat_p))
            }
    except Exception as e:
        print(f"Error running MBCp: {str(e)}")
        print("Using original data as fallback")
        results['mbcp'] = {
            'mhat_c': data['gcm_c'],
            'mhat_p': data['gcm_p']
        }
    
    print("\nRunning MBCr...")
    try:
        mbcr = mbc.MBCr(
            o_c=rcm_c_r,
            m_c=gcm_c_r,
            m_p=gcm_p_r,
            ratio_seq=ratio_seq_r, # Original ratio_seq R vector
            trace=final_trace_r,   # Potentially modified trace R vector
            jitter_factor=0,
            ties="first",
            silent=False,
            pp_type="linear",      # Added pp_type
            iter=N_ITER_MBCPR      # Added iter
        )
        
        # Handle potential NULL returns
        mhat_c = mbcr.rx2('mhat_c')
        mhat_p = mbcr.rx2('mhat_p')
        
        if mhat_c == ro.NULL or mhat_p == ro.NULL:
            print("Warning: MBCr returned NULL values, using original data")
            results['mbcr'] = {
                'mhat_c': data['gcm_c'],
                'mhat_p': data['gcm_p']
            }
        else:
            results['mbcr'] = {
                'mhat_c': np.atleast_1d(np.array(mhat_c)),
                'mhat_p': np.atleast_1d(np.array(mhat_p))
            }
    except Exception as e:
        print(f"Error running MBCr: {str(e)}")
        print("Using original data as fallback")
        results['mbcr'] = {
            'mhat_c': data['gcm_c'],
            'mhat_p': data['gcm_p']
        }
    
    print("\nRunning MBCn...")
    try:
        # Set R's seed for reproducibility of rotation matrices if generated internally by R
        r(f'set.seed({RANDOM_SEED})')

        # Generate rotation sequence in R
        n_vars_for_rot = data['gcm_c'].shape[1]
        rot_random_r_func = mbc.rot_random # Get R's rot.random function from MBC package
        
        # Create a list of R matrix objects
        rot_seq_list_r_pyobjects = [rot_random_r_func(n_vars_for_rot) for _ in range(N_ITER_MBCN)]
        rot_seq_r_vector = ro.ListVector(rot_seq_list_r_pyobjects)

        mbcn = mbc.MBCn(
            o_c=rcm_c_r,
            m_c=gcm_c_r,
            m_p=gcm_p_r,
            ratio_seq=ratio_seq_r,    # Original ratio_seq R vector
            trace=final_trace_r,      # Potentially modified trace R vector
            jitter_factor=0,
            ties="first",
            silent=False,
            n_escore=100,
            pp_type="linear",         # Added pp_type
            iter=N_ITER_MBCN,         # Added iter
            rot_seq=rot_seq_r_vector  # Added rot_seq
        )
        
        # Handle potential NULL returns
        mhat_c = mbcn.rx2('mhat_c')
        mhat_p = mbcn.rx2('mhat_p')
        
        if mhat_c == ro.NULL or mhat_p == ro.NULL:
            print("Warning: MBCn returned NULL values, using original data")
            results['mbcn'] = {
                'mhat_c': data['gcm_c'],
                'mhat_p': data['gcm_p'],
                'escore_iter': {'RAW': np.nan, 'QM': np.nan}
            }
        else:
            results['mbcn'] = {
                'mhat_c': np.atleast_1d(np.array(mhat_c)),
                'mhat_p': np.atleast_1d(np.array(mhat_p)),
                'escore_iter': dict(zip(
                    mbcn.names,
                    [np.array(x) if hasattr(x, '__len__') else x for x in mbcn]
                )) if hasattr(mbcn, 'names') else {}
            }
    except Exception as e:
        print(f"Error running MBCn: {str(e)}")
        print("Using original data as fallback")
        results['mbcn'] = {
            'mhat_c': data['gcm_c'],
            'mhat_p': data['gcm_p'],
            'escore_iter': {'error': str(e)}
        }
    
    return results

def save_results_to_netcdf(results, var_names, var_units, output_file):
    """Save results to NetCDF file"""
    print(f"\nSaving results to {output_file}...")
    with netCDF4.Dataset(output_file, 'w') as nc:
        # Create dimensions
        nc.createDimension('time_c', results['qdm']['mhat_c'].shape[0])
        nc.createDimension('time_p', results['qdm']['mhat_p'].shape[0])
        
        # Save variables for each method
        for method in ['qdm', 'mbcp', 'mbcr', 'mbcn']:
            for i, var in enumerate(var_names):
                # Ensure data is properly shaped
                mhat_c = np.atleast_1d(results[method]['mhat_c'])
                mhat_p = np.atleast_1d(results[method]['mhat_p'])
                
                # Control period
                nc_var_c = nc.createVariable(
                    f"{method}_{var}_c", 'f4', ('time_c',))
                try:
                    if mhat_c.ndim > 1:
                        nc_var_c[:] = np.nan_to_num(mhat_c[:, i])
                    else:
                        nc_var_c[:] = np.nan_to_num(mhat_c)
                    nc_var_c.units = var_units[var]
                except Exception as e:
                    print(f"Error saving {method}_{var}_c: {str(e)}")
                    nc_var_c[:] = np.nan
                    nc_var_c.units = 'unknown'
                
                # Projection period
                nc_var_p = nc.createVariable(
                    f"{method}_{var}_p", 'f4', ('time_p',))
                try:
                    if mhat_p.ndim > 1:
                        nc_var_p[:] = np.nan_to_num(mhat_p[:, i])
                    else:
                        nc_var_p[:] = np.nan_to_num(mhat_p)
                    nc_var_p.units = var_units[var]
                except Exception as e:
                    print(f"Error saving {method}_{var}_p: {str(e)}")
                    nc_var_p[:] = np.nan
                    nc_var_p.units = 'unknown'
        
        # Save energy scores if available
        if 'escore_iter' in results['mbcn']:
            for k, v in results['mbcn']['escore_iter'].items():
                if isinstance(v, (int, float)):
                    nc.setncattr(f"mbcn_escore_{k}", v)


def main():
    # Setup R environment
    install_r_packages()
    
    # Load data - construct path relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    nc_file = os.path.abspath(os.path.join(script_dir, '..', 'MBC_R', 'data', 'cccma_output.nc'))
    
    if not os.path.exists(nc_file):
        raise FileNotFoundError(
            f"NetCDF data file not found at: {nc_file}\n"
            "Please ensure the file exists and the path is correct."
        )
    
    print(f"Loading data from: {nc_file}")
    data = load_netcdf_data(nc_file)
    
    # Run MBC methods through R
    results = run_mbc_methods(data)
    
    # Save results
    output_file = 'r2py_corrected_output.nc'
    save_results_to_netcdf(results, data['var_names'], data['var_units'], output_file)
    

if __name__ == '__main__':
    main()
