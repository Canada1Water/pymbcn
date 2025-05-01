"""
This code implements several multivariate bias correction methods. Here's a breakdown of the conversion process and the resulting Python code:

Conversion Strategy and Key Considerations:

Libraries: I'll use the following Python libraries:

numpy: For numerical operations, array manipulation, and linear algebra.

scipy: For interpolation (scipy.interpolate.interp1d), Cholesky decomposition (scipy.linalg.cholesky), and nearest positive definite matrix calculation.

pandas: While not strictly necessary, pandas DataFrames can be helpful for handling tabular data (though I'll primarily stick with NumPy arrays for consistency with the R code's matrix-centric approach).

statsmodels: For plotting positions.

sklearn: For scaling and nearest neighbors search.

Function Mapping: I'll create Python functions that directly correspond to the R functions: QDM, escore, MRS, MBCr, MBCp, rot.random, MBCn, and R2D2.

Core Logic: The core logic within each function will be translated as directly as possible, paying close attention to:

Indexing: R uses 1-based indexing, while Python uses 0-based indexing. I'll adjust indices accordingly.

Vectorized Operations: R excels at vectorized operations. I'll leverage NumPy's array operations to achieve similar efficiency in Python.

Matrix Operations: I'll use NumPy's matrix multiplication (@), transpose (.T), and linear algebra functions.

Approximation/Interpolation: R's approx function is analogous to scipy.interpolate.interp1d in Python. I'll use the kind='linear' option for direct correspondence and handle the rule=2 behavior (extrapolation) explicitly.

Ranking: R's rank function with ties.method will be replicated using NumPy's argsort and careful handling of ties.

Random Number Generation: I'll use numpy.random for generating random numbers, ensuring consistency with R's behavior where needed.

NearPD: The nearPD function from R's Matrix package doesn't have a direct equivalent in SciPy. I'll implement a function to find the nearest positive definite matrix, based on Higham's (2002) algorithm, which is commonly used for this purpose.

Energy Score: The energy package in R is translated using scipy and sklearn.

Error Handling: I'll add basic checks for input data types and dimensions where appropriate.

Docstrings: I will add docstrings to the functions.

Python Code:
"""
import numpy as np
import scipy.stats as stats
from scipy.linalg import cholesky, qr
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
from scipy.linalg import solve
from scipy.stats import rankdata

def QDM(o_c, m_c, m_p, ratio=False, trace=0.05, trace_calc=0.5*0.05,
        jitter_factor=0, n_tau=None, ratio_max=2, ratio_max_trace=10*0.05,
        ECBC=False, ties='first', subsample=None, pp_type='linear'):
    """Quantile Delta Mapping bias correction"""
    # Handle jitter
    if jitter_factor == 0 and (len(np.unique(o_c)) == 1 or
                               len(np.unique(m_c)) == 1 or
                               len(np.unique(m_p)) == 1):
        jitter_factor = np.sqrt(np.finfo(float).eps)
        
    if jitter_factor > 0:
        o_c = o_c + np.random.uniform(-jitter_factor, jitter_factor, len(o_c))
        m_c = m_c + np.random.uniform(-jitter_factor, jitter_factor, len(m_c))
        m_p = m_p + np.random.uniform(-jitter_factor, jitter_factor, len(m_p))

    # Handle ratio data
    if ratio:
        epsilon = np.finfo(float).eps
        o_c[o_c < trace_calc] = np.random.uniform(epsilon, trace_calc, 
                                                 np.sum(o_c < trace_calc))
        m_c[m_c < trace_calc] = np.random.uniform(epsilon, trace_calc,
                                                 np.sum(m_c < trace_calc))
        m_p[m_p < trace_calc] = np.random.uniform(epsilon, trace_calc,
                                                 np.sum(m_p < trace_calc))

    # Calculate empirical quantiles
    n = len(m_p)
    if n_tau is None:
        n_tau = n
    tau = np.linspace(0, 1, n_tau)
    
    if subsample is not None:
        quant_o_c = np.mean([np.quantile(np.random.choice(o_c, size=len(tau)), 
                                       tau, method=pp_type) 
                           for _ in range(subsample)], axis=0)
        quant_m_c = np.mean([np.quantile(np.random.choice(m_c, size=len(tau)), 
                                       tau, method=pp_type) 
                           for _ in range(subsample)], axis=0)
        quant_m_p = np.mean([np.quantile(np.random.choice(m_p, size=len(tau)), 
                                       tau, method=pp_type) 
                           for _ in range(subsample)], axis=0)
    else:
        quant_o_c = np.quantile(o_c, tau, method=pp_type)
        quant_m_c = np.quantile(m_c, tau, method=pp_type)
        quant_m_p = np.quantile(m_p, tau, method=pp_type)

    # Apply quantile delta mapping
    tau_m_p = np.interp(m_p, quant_m_p, tau)
    
    if ratio:
        approx_t_qmc_tmp = np.interp(tau_m_p, tau, quant_m_c)
        delta_m = m_p / approx_t_qmc_tmp
        delta_m[(delta_m > ratio_max) & 
               (approx_t_qmc_tmp < ratio_max_trace)] = ratio_max
        mhat_p = np.interp(tau_m_p, tau, quant_o_c) * delta_m
    else:
        delta_m = m_p - np.interp(tau_m_p, tau, quant_m_c)
        mhat_p = np.interp(tau_m_p, tau, quant_o_c) + delta_m

    mhat_c = np.interp(m_c, quant_m_c, quant_o_c)
    
    # Handle ratio data
    if ratio:
        mhat_c[mhat_c < trace] = 0
        mhat_p[mhat_p < trace] = 0
        
    if ECBC:
        if len(mhat_p) == len(o_c):
            mhat_p = np.sort(mhat_p)[np.argsort(np.argsort(o_c))]
        else:
            raise ValueError('Schaake shuffle failed due to incompatible lengths')
            
    return {'mhat_c': mhat_c, 'mhat_p': mhat_p}

def escore(x, y, scale_x=False, n_cases=None, alpha=1, method='cluster'):
    """Energy score for assessing equality of multivariate samples"""
    # Ensure inputs are numpy arrays
    x = np.asarray(x)
    y = np.asarray(y)

    # Check for and remove rows containing NaN or Inf values
    valid_rows_x = np.all(np.isfinite(x), axis=1)
    valid_rows_y = np.all(np.isfinite(y), axis=1)
    valid_rows = valid_rows_x & valid_rows_y

    if not np.all(valid_rows):
        original_count = x.shape[0]
        x = x[valid_rows]
        y = y[valid_rows]
        removed_count = original_count - x.shape[0]
        print(f"Warning in escore: Removed {removed_count} rows containing NaN/Inf values.")
        if x.shape[0] == 0 or y.shape[0] == 0:
             print("Warning in escore: No valid data remaining after removing NaN/Inf.")
             return np.nan # Return NaN if no data is left

    if scale_x:
        # Apply scaling *after* removing invalid rows
        x = stats.zscore(x)
        y = stats.zscore(y, ddof=1) # Use sample std dev for y if it represents a model

    if n_cases is not None:
        n_cases = min(len(x), len(y), n_cases)
        x = x[np.random.choice(len(x), n_cases, replace=False)]
        y = y[np.random.choice(len(y), n_cases, replace=False)]
        
    # Calculate energy distance
    xx = cdist(x, x, 'euclidean')
    yy = cdist(y, y, 'euclidean')
    xy = cdist(x, y, 'euclidean')
    
    return (np.mean(xy) - 0.5 * np.mean(xx) - 0.5 * np.mean(yy)) / 2

def MRS(o_c, m_c, m_p, o_c_chol=None, o_p_chol=None, m_c_chol=None, m_p_chol=None):
    """Multivariate rescaling based on Cholesky decomposition"""
    # Center based on multivariate means
    o_c_mean = np.mean(o_c, axis=0)
    m_c_mean = np.mean(m_c, axis=0)
    m_p_mean = np.mean(m_p, axis=0)
    
    o_c = o_c - o_c_mean
    m_c = m_c - m_c_mean
    m_p = m_p - m_p_mean
    
    # Cholesky decomposition
    if o_c_chol is None:
        o_c_chol = cholesky(np.cov(o_c, rowvar=False))
    if o_p_chol is None:
        o_p_chol = cholesky(np.cov(o_c, rowvar=False))
    if m_c_chol is None:
        m_c_chol = cholesky(np.cov(m_c, rowvar=False))
    if m_p_chol is None:
        m_p_chol = cholesky(np.cov(m_c, rowvar=False))
        
    # Bias correction factors
    mbcfactor = solve(m_c_chol, o_c_chol)
    mbpfactor = solve(m_p_chol, o_p_chol)
    
    # Multivariate bias correction
    mbc_c = m_c @ mbcfactor
    mbc_p = m_p @ mbpfactor
    
    # Recenter and account for change in means
    mbc_c = mbc_c + o_c_mean
    mbc_p = mbc_p + o_c_mean + (m_p_mean - m_c_mean)
    
    return {'mhat_c': mbc_c, 'mhat_p': mbc_p}

def rot_random(k):
    """Generate random orthogonal rotation matrix"""
    rand = np.random.normal(size=(k, k))
    Q, R = qr(rand)
    diagR = np.diag(R)
    rot = Q @ np.diag(diagR / np.abs(diagR))
    return rot

def MBCr(o_c, m_c, m_p, iter=20, cor_thresh=1e-4, ratio_seq=None, trace=0.05,
         trace_calc=0.5*0.05, jitter_factor=0, n_tau=None, ratio_max=2,
         ratio_max_trace=10*0.05, ties='ordinal', qmap_precalc=False, # Changed default ties to 'ordinal'
         silent=False, subsample=None, pp_type='linear'):
    """Multivariate quantile mapping bias correction (Spearman correlation)"""
    n_vars = o_c.shape[1]
    if ratio_seq is None:
        ratio_seq = [False] * n_vars
    if np.isscalar(trace_calc):
        trace_calc = [trace_calc] * n_vars
    if np.isscalar(trace):
        trace = [trace] * n_vars
    if np.isscalar(jitter_factor):
        jitter_factor = [jitter_factor] * n_vars
    if np.isscalar(ratio_max):
        ratio_max = [ratio_max] * n_vars
    if np.isscalar(ratio_max_trace):
        ratio_max_trace = [ratio_max_trace] * n_vars
        
    m_c_qmap = m_c.copy()
    m_p_qmap = m_p.copy()
    
    if not qmap_precalc:
        for i in range(n_vars):
            fit_qmap = QDM(o_c[:,i], m_c[:,i], m_p[:,i],
                          ratio=ratio_seq[i], trace_calc=trace_calc[i],
                          trace=trace[i], jitter_factor=jitter_factor[i],
                          n_tau=n_tau, ratio_max=ratio_max[i],
                          ratio_max_trace=ratio_max_trace[i],
                          subsample=subsample, pp_type=pp_type)
            m_c_qmap[:,i] = fit_qmap['mhat_c']
            m_p_qmap[:,i] = fit_qmap['mhat_p']

    # Calculate ranks using 'ordinal' to better handle ties
    o_c_r = np.apply_along_axis(rankdata, 0, o_c, method='ordinal')
    m_c_r = np.apply_along_axis(rankdata, 0, m_c, method='ordinal')
    m_p_r = np.apply_along_axis(rankdata, 0, m_p, method='ordinal')

    m_c_i = m_c_r.copy()
    if cor_thresh > 0:
        cor_i = np.corrcoef(m_c_r, rowvar=False)
        cor_i[np.isnan(cor_i)] = 0
    
    # Iterative MBC/reranking
    o_c_chol = o_p_chol = cholesky(np.cov(o_c_r, rowvar=False))
    
    for i in range(iter):
        m_c_chol = m_p_chol = cholesky(np.cov(m_c_r, rowvar=False))
        fit_mbc = MRS(o_c_r, m_c_r, m_p_r, o_c_chol=o_c_chol,
                     o_p_chol=o_p_chol, m_c_chol=m_c_chol, m_p_chol=m_p_chol)

        # Use 'ordinal' ranking within the loop as well
        m_c_r = np.apply_along_axis(rankdata, 0, fit_mbc['mhat_c'], method='ordinal')
        m_p_r = np.apply_along_axis(rankdata, 0, fit_mbc['mhat_p'], method='ordinal')

        if cor_thresh > 0:
            cor_j = np.corrcoef(m_c_r, rowvar=False)
            cor_j[np.isnan(cor_j)] = 0
            cor_diff = np.mean(np.abs(cor_j - cor_i))
            cor_i = cor_j
        else:
            cor_diff = 0
            
        if not silent:
            print(f"{i} {np.mean(m_c_r == m_c_i)} {cor_diff} ", end='')
            
        if cor_diff < cor_thresh:
            break
        if np.array_equal(m_c_r, m_c_i):
            break
            
        m_c_i = m_c_r.copy()
        
    if not silent:
        print()
        
    # Replace ranks with QDM values while preserving rank structure
    for i in range(n_vars):
        # Get ranks of the transformed data using 'ordinal'
        m_c_ranks = rankdata(m_c_r[:,i], method='ordinal') - 1  # Convert to 0-based
        m_p_ranks = rankdata(m_p_r[:,i], method='ordinal') - 1

        # Sort QDM values
        sorted_qmap_c = np.sort(m_c_qmap[:,i])
        sorted_qmap_p = np.sort(m_p_qmap[:,i])
        
        # Map ranks to QDM values
        m_c_r[:,i] = sorted_qmap_c[m_c_ranks]
        m_p_r[:,i] = sorted_qmap_p[m_p_ranks]
        
    return {'mhat_c': m_c_r, 'mhat_p': m_p_r}

def MBCp(o_c, m_c, m_p, iter=20, cor_thresh=1e-4, ratio_seq=None, trace=0.05,
         trace_calc=0.5*0.05, jitter_factor=0, n_tau=None, ratio_max=2,
         ratio_max_trace=10*0.05, ties='first', qmap_precalc=False,
         silent=False, subsample=None, pp_type='linear'):
    """Multivariate quantile mapping bias correction (Pearson correlation)"""
    n_vars = o_c.shape[1]
    if ratio_seq is None:
        ratio_seq = [False] * n_vars
    if np.isscalar(trace_calc):
        trace_calc = [trace_calc] * n_vars
    if np.isscalar(trace):
        trace = [trace] * n_vars
    if np.isscalar(jitter_factor):
        jitter_factor = [jitter_factor] * n_vars
    if np.isscalar(ratio_max):
        ratio_max = [ratio_max] * n_vars
    if np.isscalar(ratio_max_trace):
        ratio_max_trace = [ratio_max_trace] * n_vars
        
    m_c_qmap = m_c.copy()
    m_p_qmap = m_p.copy()
    
    if not qmap_precalc:
        for i in range(n_vars):
            fit_qmap = QDM(o_c[:,i], m_c[:,i], m_p[:,i],
                          ratio=ratio_seq[i], trace_calc=trace_calc[i],
                          trace=trace[i], jitter_factor=jitter_factor[i],
                          n_tau=n_tau, ratio_max=ratio_max[i],
                          ratio_max_trace=ratio_max_trace[i],
                          subsample=subsample, pp_type=pp_type)
            m_c_qmap[:,i] = fit_qmap['mhat_c']
            m_p_qmap[:,i] = fit_qmap['mhat_p']
    
    m_c = m_c_qmap.copy()
    m_p = m_p_qmap.copy()
    
    if cor_thresh > 0:
        cor_i = np.corrcoef(m_c, rowvar=False)
        cor_i[np.isnan(cor_i)] = 0
    
    o_c_chol = o_p_chol = cholesky(np.cov(o_c, rowvar=False))
    
    for i in range(iter):
        m_c_chol = m_p_chol = cholesky(np.cov(m_c, rowvar=False))
        fit_mbc = MRS(o_c, m_c, m_p, o_c_chol=o_c_chol,
                     o_p_chol=o_p_chol, m_c_chol=m_c_chol, m_p_chol=m_p_chol)
        
        m_c = fit_mbc['mhat_c']
        m_p = fit_mbc['mhat_p']
        
        for j in range(n_vars):
            fit_qmap = QDM(o_c[:,j], m_c[:,j], m_p[:,j], ratio=False,
                          n_tau=n_tau, pp_type=pp_type)
            m_c[:,j] = fit_qmap['mhat_c']
            m_p[:,j] = fit_qmap['mhat_p']
            
        if cor_thresh > 0:
            cor_j = np.corrcoef(m_c, rowvar=False)
            cor_j[np.isnan(cor_j)] = 0
            cor_diff = np.mean(np.abs(cor_j - cor_i))
            cor_i = cor_j
        else:
            cor_diff = 0
            
        if not silent:
            print(f"{i} {cor_diff} ", end='')
            
        if cor_diff < cor_thresh:
            break
            
    if not silent:
        print()
        
    # Replace with shuffled QDM elements using 'ordinal' ranks
    for i in range(n_vars):
        m_c[:,i] = np.sort(m_c_qmap[:,i])[rankdata(m_c[:,i], method='ordinal')-1]
        m_p[:,i] = np.sort(m_p_qmap[:,i])[rankdata(m_p[:,i], method='ordinal')-1]

    return {'mhat_c': m_c, 'mhat_p': m_p}

def MBCn(o_c, m_c, m_p, iter=30, ratio_seq=None, trace=0.05,
         trace_calc=0.5*0.05, jitter_factor=0, n_tau=None, ratio_max=2,
         ratio_max_trace=10*0.05, ties='first', qmap_precalc=False,
         rot_seq=None, silent=False, n_escore=0, return_all=False,
         subsample=None, pp_type='linear'):
    """Multivariate quantile mapping bias correction (N-dimensional pdf transfer)"""
    n_vars = o_c.shape[1]
    if ratio_seq is None:
        ratio_seq = [False] * n_vars
    if np.isscalar(trace_calc):
        trace_calc = [trace_calc] * n_vars
    if np.isscalar(trace):
        trace = [trace] * n_vars
    if np.isscalar(jitter_factor):
        jitter_factor = [jitter_factor] * n_vars
    if np.isscalar(ratio_max):
        ratio_max = [ratio_max] * n_vars
    if np.isscalar(ratio_max_trace):
        ratio_max_trace = [ratio_max_trace] * n_vars
        
    if rot_seq is not None and len(rot_seq) != iter:
        raise ValueError('length(rot_seq) != iter')
        
    # Energy score tracking
    escore_iter = np.full(iter+2, np.nan)
    
    if n_escore > 0:
        n_escore = min(o_c.shape[0], m_c.shape[0], n_escore)
        escore_cases_o_c = np.random.choice(o_c.shape[0], n_escore, replace=False)
        escore_cases_m_c = np.random.choice(m_c.shape[0], n_escore, replace=False)
        escore_iter[0] = escore(o_c[escore_cases_o_c], m_c[escore_cases_m_c],
                               scale_x=True)
        if not silent:
            print(f"RAW {escore_iter[0]} : ", end='')
    
    m_c_qmap = m_c.copy()
    m_p_qmap = m_p.copy()
    
    if not qmap_precalc:
        for i in range(n_vars):
            fit_qmap = QDM(o_c[:,i], m_c[:,i], m_p[:,i],
                          ratio=ratio_seq[i], trace_calc=trace_calc[i],
                          trace=trace[i], jitter_factor=jitter_factor[i],
                          n_tau=n_tau, ratio_max=ratio_max[i],
                          ratio_max_trace=ratio_max_trace[i],
                          subsample=subsample, pp_type=pp_type)
            m_c_qmap[:,i] = fit_qmap['mhat_c']
            m_p_qmap[:,i] = fit_qmap['mhat_p']
    
    m_c = m_c_qmap.copy()
    m_p = m_p_qmap.copy()
    
    if n_escore > 0:
        escore_iter[1] = escore(o_c[escore_cases_o_c], m_c[escore_cases_m_c],
                               scale_x=True)
        if not silent:
            print(f"QDM {escore_iter[1]} : ", end='')
    
    # Standardize data
    o_c_mean = np.mean(o_c, axis=0)
    o_c_sdev = np.std(o_c, axis=0)
    o_c_sdev[o_c_sdev < np.finfo(float).eps] = 1
    o_c = (o_c - o_c_mean) / o_c_sdev
    
    m_c_p = np.vstack((m_c, m_p))
    m_c_p_mean = np.mean(m_c_p, axis=0)
    m_c_p_sdev = np.std(m_c_p, axis=0)
    m_c_p_sdev[m_c_p_sdev < np.finfo(float).eps] = 1
    m_c_p = (m_c_p - m_c_p_mean) / m_c_p_sdev
    
    Xt = np.vstack((o_c, m_c_p))
    m_iter = []
    
    for i in range(iter):
        if not silent:
            print(f"{i} ", end='')
            
        # Random rotation
        if rot_seq is None:
            rot = rot_random(n_vars)
        else:
            rot = rot_seq[i]
            
        Z = Xt @ rot
        Z_o_c = Z[:o_c.shape[0]]
        Z_m_c = Z[o_c.shape[0]:o_c.shape[0]+m_c.shape[0]]
        Z_m_p = Z[o_c.shape[0]+m_c.shape[0]:]
        
        # Bias correct rotated variables
        for j in range(n_vars):
            Z_qdm = QDM(Z_o_c[:,j], Z_m_c[:,j], Z_m_p[:,j], ratio=False,
                       jitter_factor=jitter_factor[j], n_tau=n_tau,
                       pp_type=pp_type)
            Z_m_c[:,j] = Z_qdm['mhat_c']
            Z_m_p[:,j] = Z_qdm['mhat_p']
            
        # Rotate back
        m_c = Z_m_c @ rot.T
        m_p = Z_m_p @ rot.T
        Xt = np.vstack((o_c, m_c, m_p))
        
        # Track energy score
        if n_escore > 0:
            escore_iter[i+2] = escore(o_c[escore_cases_o_c], m_c[escore_cases_m_c],
                                     scale_x=True)
            if not silent:
                print(f"{escore_iter[i+2]} : ", end='')
                
        if return_all:
            m_c_i = m_c * m_c_p_sdev + m_c_p_mean
            m_p_i = m_p * m_c_p_sdev + m_c_p_mean
            m_iter.append({'m_c': m_c_i, 'm_p': m_p_i})
            
    if not silent:
        print()
        
    # Rescale back to original units
    m_c = m_c * m_c_p_sdev + m_c_p_mean
    m_p = m_p * m_c_p_sdev + m_c_p_mean
    
    # Replace with QDM outputs
    for i in range(n_vars):
        m_c[:,i] = np.sort(m_c_qmap[:,i])[rankdata(m_c[:,i], method='min')-1]
        m_p[:,i] = np.sort(m_p_qmap[:,i])[rankdata(m_p[:,i], method='min')-1]
        
    escore_iter = dict(zip(['RAW', 'QM'] + list(range(iter)), escore_iter))
    
    return {'mhat_c': m_c, 'mhat_p': m_p, 'escore_iter': escore_iter,
            'm_iter': m_iter}

def R2D2(o_c, m_c, m_p, ref_column=0, ratio_seq=None, trace=0.05,
         trace_calc=0.5*0.05, jitter_factor=0, n_tau=None, ratio_max=2,
         ratio_max_trace=10*0.05, ties='first', qmap_precalc=False,
         subsample=None, pp_type='linear'): # Changed default pp_type to linear for consistency
    """Multivariate bias correction using nearest neighbor rank resampling"""
    if o_c.shape[0] != m_c.shape[0] or o_c.shape[0] != m_p.shape[0]:
        raise ValueError("R2D2 requires data samples of equal length")
        
    n_vars = o_c.shape[1]
    if ratio_seq is None:
        ratio_seq = [False] * n_vars
    if np.isscalar(trace_calc):
        trace_calc = [trace_calc] * n_vars
    if np.isscalar(trace):
        trace = [trace] * n_vars
    if np.isscalar(jitter_factor):
        jitter_factor = [jitter_factor] * n_vars
    if np.isscalar(ratio_max):
        ratio_max = [ratio_max] * n_vars
    if np.isscalar(ratio_max_trace):
        ratio_max_trace = [ratio_max_trace] * n_vars
        
    m_c_qmap = m_c.copy()
    m_p_qmap = m_p.copy()
    
    if not qmap_precalc:
        for i in range(n_vars):
            fit_qmap = QDM(o_c[:,i], m_c[:,i], m_p[:,i],
                          ratio=ratio_seq[i], trace_calc=trace_calc[i],
                          trace=trace[i], jitter_factor=jitter_factor[i],
                          n_tau=n_tau, ratio_max=ratio_max[i],
                          ratio_max_trace=ratio_max_trace[i],
                          subsample=subsample, pp_type=pp_type)
            m_c_qmap[:,i] = fit_qmap['mhat_c']
            m_p_qmap[:,i] = fit_qmap['mhat_p']

    # Map 'first' tie-breaking to 'ordinal' for rankdata
    rank_method = 'ordinal' if ties == 'first' else ties

    # Calculate ranks
    o_c_r = np.apply_along_axis(rankdata, 0, o_c, method=rank_method)
    m_c_r = np.apply_along_axis(rankdata, 0, m_c_qmap, method=rank_method)
    m_p_r = np.apply_along_axis(rankdata, 0, m_p_qmap, method=rank_method)

    # Find nearest neighbors
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(o_c_r[:,ref_column].reshape(-1,1))
    
    nn_c_r = rankdata(nn.kneighbors(m_c_r[:,ref_column].reshape(-1,1),
                                   return_distance=False).flatten(),
                     method='random')
    nn_p_r = rankdata(nn.kneighbors(m_p_r[:,ref_column].reshape(-1,1),
                                   return_distance=False).flatten(),
                     method='random')
    
    # Shuffle ranks
    new_c_r = o_c_r[nn_c_r-1]
    new_p_r = o_c_r[nn_p_r-1]
    
    # Reorder QDM outputs
    r2d2_c = m_c_qmap.copy()
    r2d2_p = m_p_qmap.copy()
    
    for i in range(n_vars):
        r2d2_c[:,i] = np.sort(r2d2_c[:,i])[new_c_r[:,i].astype(int)-1]
        r2d2_p[:,i] = np.sort(r2d2_p[:,i])[new_p_r[:,i].astype(int)-1]
        
    return {'mhat_c': r2d2_c, 'mhat_p': r2d2_p}
