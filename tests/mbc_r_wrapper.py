import os
import time
import numpy as np
import netCDF4
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
import subprocess
import sys

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
    """Load data from NetCDF file into numpy arrays"""
    print(f"Loading data from {nc_file_path}...")
    with netCDF4.Dataset(nc_file_path) as nc:
        # Get variable names (assuming standard CCCMA format)
        var_names = [v.split('_')[-1] for v in nc.variables 
                    if v.startswith('gcm_c_') or v.startswith('rcm_c_')]
        var_names = sorted(list(set(var_names)))  # Get unique sorted vars
        
        # Load data and units into arrays
        gcm_c = np.column_stack([nc.variables[f'gcm_c_{var}'][:] for var in var_names])
        rcm_c = np.column_stack([nc.variables[f'rcm_c_{var}'][:] for var in var_names])
        gcm_p = np.column_stack([nc.variables[f'gcm_p_{var}'][:] for var in var_names])
        
        # Store units for each variable
        var_units = {}
        for var in var_names:
            try:
                var_units[var] = nc.variables[f'gcm_c_{var}'].units
            except AttributeError:
                var_units[var] = 'unknown'
        
        # Get metadata and handle masked values
        ratio_seq = np.ma.filled(nc.variables['ratio_seq'][:], False).astype(bool)
        trace = np.ma.filled(nc.variables['trace'][:], np.nan)
        
    return {
        'gcm_c': gcm_c,
        'rcm_c': rcm_c,
        'gcm_p': gcm_p,
        'var_names': var_names,
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
        if np.any(np.isnan(arr)):
            print(f"Warning: {arr_name} contains NaN values")
        if np.any(np.isinf(arr)):
            print(f"Warning: {arr_name} contains Inf values")
    
    gcm_c_r = r.matrix(data['gcm_c'], nrow=data['gcm_c'].shape[0], ncol=data['gcm_c'].shape[1])
    rcm_c_r = r.matrix(data['rcm_c'], nrow=data['rcm_c'].shape[0], ncol=data['rcm_c'].shape[1])
    gcm_p_r = r.matrix(data['gcm_p'], nrow=data['gcm_p'].shape[0], ncol=data['gcm_p'].shape[1])
    
    # Convert parameters to R format
    ratio_seq_r = ro.BoolVector(data['ratio_seq'])
    trace_r = ro.FloatVector(data['trace'])
    trace_calc_r = ro.FloatVector(data['trace'] * 0.5)  # Pre-calculate trace_calc
    
    # Run MBC methods
    results = {}
    
    print("\nRunning QDM for each variable...")
    qdm_c = np.zeros_like(data['gcm_c'])
    qdm_p = np.zeros_like(data['gcm_p'])
    
    for i in range(len(data['var_names'])):
        print(f"  Variable {i+1}/{len(data['var_names'])}: {data['var_names'][i]}")
        # Extract columns by converting to R vectors first
        o_c_col = ro.FloatVector(data['rcm_c'][:, i])
        m_c_col = ro.FloatVector(data['gcm_c'][:, i])
        m_p_col = ro.FloatVector(data['gcm_p'][:, i])
        
        try:
            print(f"Running QDM for {data['var_names'][i]} with trace={trace_r[i]}, ratio={ratio_seq_r[i]}")
            qdm = mbc.QDM(
                o_c=o_c_col,
                m_c=m_c_col,
                m_p=m_p_col,
                ratio=ratio_seq_r[i],
                trace=trace_r[i],
                trace_calc=trace_calc_r[i],
                jitter_factor=0,
                ties="first"
            )
            print(f"QDM completed for {data['var_names'][i]}")
            
            # Check for NULL returns and handle empty results
            mhat_c = qdm.rx2('mhat_c')
            mhat_p = qdm.rx2('mhat_p')
            
            if mhat_c == ro.NULL or mhat_p == ro.NULL:
                print(f"Warning: QDM returned NULL for variable {data['var_names'][i]}, using original values")
                qdm_c[:, i] = data['gcm_c'][:, i]
                qdm_p[:, i] = data['gcm_p'][:, i]
            else:
                # Ensure results are properly shaped arrays
                qdm_c[:, i] = np.atleast_1d(np.array(mhat_c))
                qdm_p[:, i] = np.atleast_1d(np.array(mhat_p))
            
        except Exception as e:
            print(f"Error processing variable {data['var_names'][i]}: {str(e)}")
            # Fill with original values if QDM fails
            qdm_c[:, i] = data['gcm_c'][:, i]
            qdm_p[:, i] = data['gcm_p'][:, i]
    
    results['qdm'] = {
        'mhat_c': qdm_c,
        'mhat_p': qdm_p
    }
    
    print("\nRunning MBCp...")
    try:
        mbcp = mbc.MBCp(
            o_c=rcm_c_r,
            m_c=gcm_c_r,
            m_p=gcm_p_r,
            ratio_seq=ratio_seq_r,
            trace=trace_r,
            jitter_factor=0,
            ties="first",
            silent=False
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
            ratio_seq=ratio_seq_r,
            trace=trace_r,
            jitter_factor=0,
            ties="first",
            silent=False
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
        mbcn = mbc.MBCn(
            o_c=rcm_c_r,
            m_c=gcm_c_r,
            m_p=gcm_p_r,
            ratio_seq=ratio_seq_r,
            trace=trace_r,
            jitter_factor=0,
            ties="first",
            silent=False,
            n_escore=100
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

def run_comparison():
    """Run the comparison script"""
    print("\nRunning comparison between R and Python results...")
    
    # Wait for files to exist
    max_attempts = 5
    wait_seconds = 1
    files_exist = False
    
    for attempt in range(max_attempts):
        if (os.path.exists('python_corrected_output.nc') and 
            os.path.exists('r2py_corrected_output.nc')):
            files_exist = True
            break
        time.sleep(wait_seconds)
    
    if not files_exist:
        print("Error: Required NetCDF files not found after waiting")
        return
        
    try:
        # Get absolute path to comparison script (it's in the same directory)
        script_path = os.path.join(os.path.dirname(__file__), "compare_r2py_to_R_outputs.py")
        subprocess.run([sys.executable, script_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running comparison: {e}")
    except Exception as e:
        print(f"Unexpected error running comparison: {e}")

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
    
    # Run comparison with Python results
    run_comparison()

if __name__ == '__main__':
    main()
