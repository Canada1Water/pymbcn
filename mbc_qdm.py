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
from scipy.linalg import cholesky, qr #solve is also from scipy.linalg
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
from scipy.linalg import solve # Explicit import
from scipy.stats import rankdata
import pandas as pd # Import pandas for describe() in diagnostics

# --- Global flag for MRS diagnostics ---
MRS_DEBUG_PRINT_PY = False

# Helper function for nearPD
def _ensure_symmetric(A):
    return (A + A.T) / 2

def nearPD(A, epsilon_eig=1e-6, chol_jitter_factor=1e-9, max_jitter_iter=10):
    """
    Computes the nearest positive definite matrix to A.
    Primarily uses eigenvalue thresholding. If Cholesky still fails,
    applies iterative diagonal jitter.
    A: input matrix
    epsilon_eig: smallest allowed eigenvalue.
    chol_jitter_factor: initial factor for diagonal jitter if eigenvalue method isn't enough.
    max_jitter_iter: maximum iterations for jittering.
    """
    X = np.asarray(A)
    if X.ndim != 2 or X.shape[0] != X.shape[1]:
        raise ValueError("Input must be a square 2D matrix.")

    # Ensure symmetry
    Y = _ensure_symmetric(X)

    # Eigenvalue decomposition
    try:
        eigvals, eigvecs = np.linalg.eigh(Y)
    except np.linalg.LinAlgError:
        # If eigh fails, fall back to jittering the original symmetric matrix Y
        # print("Warning: nearPD eigendecomposition failed. Proceeding with jittering.")
        eigvals = None # Signal that eigenvalue adjustment was skipped

    if eigvals is not None:
        # Adjust eigenvalues: set any eigenvalue < epsilon_eig to epsilon_eig
        # R's Matrix::nearPD uses eig.tol * max_eigenvalue as threshold.
        # Using a fixed epsilon_eig is simpler and common.
        # max_eig = np.max(eigvals) if len(eigvals) > 0 else 1.0
        # current_epsilon_eig = epsilon_eig * max_eig if max_eig > 0 else epsilon_eig
        current_epsilon_eig = epsilon_eig # Using fixed epsilon

        if np.any(eigvals < current_epsilon_eig):
            eigvals[eigvals < current_epsilon_eig] = current_epsilon_eig
            Y = _ensure_symmetric(eigvecs @ np.diag(eigvals) @ eigvecs.T)
        # If all eigenvalues were already >= current_epsilon_eig, Y is unchanged from symmetric X
    
    # Try Cholesky on the eigenvalue-adjusted matrix
    try:
        np.linalg.cholesky(Y)
        return Y # Success
    except np.linalg.LinAlgError:
        # print("Warning: Cholesky failed after eigenvalue adjustment. Attempting jitter.")
        pass # Proceed to jittering

    # If Cholesky failed after eigenvalue adjustment (or eigh failed), try jittering
    # This Y is either from eigenvalue adjustment or the initial symmetric X if eigh failed.
    Y_jittered = Y.copy()
    for i in range(max_jitter_iter):
        try:
            # Add jitter: scaled identity matrix
            # Scale jitter by mean of diagonal elements to make it somewhat relative
            diag_mean_abs = np.mean(np.abs(np.diag(Y_jittered)))
            if diag_mean_abs < np.finfo(float).eps: diag_mean_abs = 1.0
            
            current_jitter = (chol_jitter_factor * (10**i)) * diag_mean_abs
            if current_jitter < np.finfo(float).eps: # Ensure jitter is not too small
                 current_jitter = chol_jitter_factor * (10**i)
            
            # Add to a fresh copy of Y to avoid accumulating jitter if Y itself was not PD enough
            # Or jitter Y_jittered iteratively
            Y_to_try = _ensure_symmetric(Y_jittered + np.eye(Y.shape[0]) * current_jitter)

            np.linalg.cholesky(Y_to_try)
            return Y_to_try # Success
        except np.linalg.LinAlgError:
            if i == max_jitter_iter - 1:
                # print(f"Warning: nearPD Cholesky failed after {max_jitter_iter} jitter attempts. Returning best effort before last jitter.")
                # Return the matrix before the last, potentially destabilizing, jitter
                # Or, could return Y_to_try (the last attempt)
                # R's nearPD would return its last best estimate.
                # Let's return the matrix that was last attempted with Cholesky.
                return Y_to_try 
    
    # Should ideally not be reached if max_jitter_iter is > 0 and jitter is applied.
    # However, if all attempts fail, return the eigenvalue-adjusted (or initial symmetric) Y.
    # print("Warning: nearPD exhausted options. Returning eigenvalue-adjusted/symmetrized matrix.")
    return Y


def QDM(o_c, m_c, m_p, ratio=False, trace=0.05, trace_calc=0.5*0.05,
        jitter_factor=0, n_tau=None, ratio_max=2, ratio_max_trace=10*0.05,
        ECBC=False, ties='first', subsample=None, pp_type='linear'):
    """Quantile Delta Mapping bias correction"""
    
    o_c_arr = np.asarray(o_c).copy() # Work on copies
    m_c_arr = np.asarray(m_c).copy()
    m_p_arr = np.asarray(m_p).copy()

    # Determine the actual jitter factor to use (R's 'factor' for jitter function)
    current_r_jitter_factor = jitter_factor
    if jitter_factor == 0 and (len(np.unique(o_c_arr)) == 1 or
                               len(np.unique(m_c_arr)) == 1 or
                               len(np.unique(m_p_arr)) == 1):
        current_r_jitter_factor = np.sqrt(np.finfo(float).eps) # R uses 1.0 for this case, but factor is then scaled.
                                                              # Let's use a small factor.
        
    if current_r_jitter_factor > 0:
        # R's jitter: amount = factor * diff(range(x))/50 or factor * z/50 if range is 0
        # where z = abs(mean(x)) or 1 if mean is too small.
        for i, arr_ref_tuple in enumerate([(o_c_arr, 'o_c'), (m_c_arr, 'm_c'), (m_p_arr, 'm_p')]):
            arr_ref, arr_name = arr_ref_tuple
            val_range = np.max(arr_ref) - np.min(arr_ref)
            if val_range <= np.finfo(float).eps * np.abs(np.mean(arr_ref)): # Check if range is effectively zero relative to mean
                z = np.abs(np.mean(arr_ref))
                if z < 1e-10 * (np.max(np.abs(arr_ref)) if len(arr_ref)>0 else 1.0) : z = 1.0 # R's logic for z
                amount = current_r_jitter_factor * z / 50.0
            else:
                amount = current_r_jitter_factor * val_range / 50.0
            
            # Ensure amount is not zero if jitter is intended and factor is non-zero
            if amount <= np.finfo(float).eps and current_r_jitter_factor > 0:
                amount = current_r_jitter_factor * np.sqrt(np.finfo(float).eps) # A very small amount
                if amount <= np.finfo(float).eps: # if current_r_jitter_factor is also tiny
                    amount = np.finfo(float).eps * 10 # Ensure it's slightly larger than eps

            noise = np.random.uniform(-amount, amount, len(arr_ref))

            if arr_name == 'o_c': o_c_arr += noise
            elif arr_name == 'm_c': m_c_arr += noise
            elif arr_name == 'm_p': m_p_arr += noise


    # Handle ratio data
    if ratio:
        epsilon = np.finfo(float).eps # A very small number
        # For o_c
        mask_o_c = o_c_arr < trace_calc
        count_o_c = np.sum(mask_o_c)
        if count_o_c > 0:
            o_c_arr[mask_o_c] = np.random.uniform(epsilon, trace_calc, count_o_c)
        # For m_c
        mask_m_c = m_c_arr < trace_calc
        count_m_c = np.sum(mask_m_c)
        if count_m_c > 0:
            m_c_arr[mask_m_c] = np.random.uniform(epsilon, trace_calc, count_m_c)
        # For m_p
        mask_m_p = m_p_arr < trace_calc
        count_m_p = np.sum(mask_m_p)
        if count_m_p > 0:
            m_p_arr[mask_m_p] = np.random.uniform(epsilon, trace_calc, count_m_p)


    # Calculate empirical quantiles
    n = len(m_p_arr)
    if n_tau is None:
        n_tau = n
    # R's seq(0,1,length=n.tau) includes 0 and 1. np.linspace does this by default.
    tau = np.linspace(0, 1, n_tau) 
    
    # Ensure tau has unique values for interpolation, especially if n_tau is small
    if len(tau) > 1 and np.allclose(np.diff(tau), 0): 
        tau = np.unique(tau) # Should not happen with linspace unless n_tau=1
    if len(tau) < 2 and n_tau > 1 : # If still not enough unique points
         tau = np.array([0.0, 1.0]) if n_tau > 1 else np.array([0.5])


    if subsample is not None:
        # Ensure sample size for np.random.choice is not larger than population
        # And that len(tau) is used for size if it's smaller than o_c_arr, etc.
        # R's sample(o.c, size=length(tau))
        # Python: np.random.choice(o_c_arr, size=len(o_c_arr) if len(tau) > len(o_c_arr) else len(tau), replace=True)
        # The R code uses sample(o.c, size=length(tau)). This implies length(tau) <= length(o.c)
        # If subsample is used, it implies o.c, m.c, m.p are large, so len(tau) is likely smaller.
        # We need to ensure the size for np.random.choice is valid.
        # Let's assume len(tau) is the intended sample size for quantile calculation.
        
        sample_size_for_quantile = len(tau)

        quant_o_c = np.mean([np.quantile(np.random.choice(o_c_arr, size=min(sample_size_for_quantile, len(o_c_arr)), replace=True), 
                                       tau, method=pp_type) 
                           for _ in range(subsample)], axis=0)
        quant_m_c = np.mean([np.quantile(np.random.choice(m_c_arr, size=min(sample_size_for_quantile, len(m_c_arr)), replace=True), 
                                       tau, method=pp_type) 
                           for _ in range(subsample)], axis=0)
        quant_m_p = np.mean([np.quantile(np.random.choice(m_p_arr, size=min(sample_size_for_quantile, len(m_p_arr)), replace=True), 
                                       tau, method=pp_type) 
                           for _ in range(subsample)], axis=0)
    else:
        quant_o_c = np.quantile(o_c_arr, tau, method=pp_type)
        quant_m_c = np.quantile(m_c_arr, tau, method=pp_type)
        quant_m_p = np.quantile(m_p_arr, tau, method=pp_type)

    # Ensure quantiles are sorted for interpolation (np.quantile should produce sorted)
    # but jitter or small n_tau might cause issues. Explicitly sort.
    quant_o_c = np.sort(quant_o_c)
    quant_m_c = np.sort(quant_m_c)
    quant_m_p = np.sort(quant_m_p)


    # Apply quantile delta mapping
    # np.interp needs xp to be increasing. Quantiles should be.
    # Handle rule=2: np.interp default behavior matches rule=2 (extrapolates with endpoint values)
    tau_m_p = np.interp(m_p_arr, quant_m_p, tau, left=tau[0], right=tau[-1]) 
    
    if ratio:
        approx_t_qmc_tmp = np.interp(tau_m_p, tau, quant_m_c, left=quant_m_c[0], right=quant_m_c[-1])
        # Avoid division by zero or very small numbers if not intended by trace logic
        # R's behavior: if approx_t_qmc_tmp is zero, delta_m becomes Inf.
        # Let's add a small epsilon to denominator to prevent strict zero division if it's problematic.
        # However, R's code doesn't explicitly do this, so Inf/NaN might be expected.
        # The trace logic should handle small values.
        # If approx_t_qmc_tmp is zero and m_p_arr is also zero, 0/0 -> NaN.
        # If approx_t_qmc_tmp is zero and m_p_arr is non-zero, non-zero/0 -> Inf.
        delta_m = m_p_arr / (approx_t_qmc_tmp + np.finfo(float).eps * (approx_t_qmc_tmp==0)) # Add small epsilon only if zero
        
        # Apply ratio_max constraint
        mask_ratio_max = (delta_m > ratio_max) & (approx_t_qmc_tmp < ratio_max_trace)
        delta_m[mask_ratio_max] = ratio_max
        
        # Handle Inf/NaN in delta_m before multiplication if necessary
        # If delta_m is Inf due to approx_t_qmc_tmp being near zero, and quant_o_c is also near zero,
        # the result might be NaN (0 * Inf). R handles this.
        # Python: np.inf * 0 is nan.
        delta_m = np.nan_to_num(delta_m, nan=1.0, posinf=ratio_max, neginf=1/ratio_max if ratio_max !=0 else 0) # Heuristic for Inf

        mhat_p = np.interp(tau_m_p, tau, quant_o_c, left=quant_o_c[0], right=quant_o_c[-1]) * delta_m
    else:
        delta_m = m_p_arr - np.interp(tau_m_p, tau, quant_m_c, left=quant_m_c[0], right=quant_m_c[-1])
        mhat_p = np.interp(tau_m_p, tau, quant_o_c, left=quant_o_c[0], right=quant_o_c[-1]) + delta_m

    mhat_c = np.interp(m_c_arr, quant_m_c, quant_o_c, left=quant_o_c[0], right=quant_o_c[-1])
    
    # Handle ratio data output
    if ratio:
        mhat_c[mhat_c < trace] = 0
        mhat_p[mhat_p < trace] = 0
        
    if ECBC:
        if len(mhat_p) == len(o_c_arr):
            # rankdata gives 1-based ranks, adjust for 0-based indexing
            # R's ties='first' is 'ordinal' in scipy.stats.rankdata
            rank_method_ecbc = 'ordinal' if ties == 'first' else ties
            mhat_p = np.sort(mhat_p)[rankdata(o_c_arr, method=rank_method_ecbc) - 1]
        else:
            raise ValueError('Schaake shuffle failed due to incompatible lengths')
            
    return {'mhat_c': mhat_c, 'mhat_p': mhat_p}

def escore(x, y, scale_x=False, n_cases=None, alpha=1): # method not used by Python version
    """Energy score for assessing equality of multivariate samples"""
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)

    if x_arr.ndim == 1: x_arr = x_arr.reshape(-1, 1)
    if y_arr.ndim == 1: y_arr = y_arr.reshape(-1, 1)

    # Initial NaN/Inf filtering based on both x and y
    x_finite_mask_initial = np.all(np.isfinite(x_arr), axis=1)
    y_finite_mask_initial = np.all(np.isfinite(y_arr), axis=1)
    common_finite_mask_initial = x_finite_mask_initial & y_finite_mask_initial
    
    x_clean = x_arr[common_finite_mask_initial]
    y_clean = y_arr[common_finite_mask_initial]
    
    if x_clean.shape[0] < 2 or y_clean.shape[0] < 2: 
        return np.nan

    x_proc = x_clean.copy() # Use copies for processing
    y_proc = y_clean.copy()

    if scale_x:
        mean_x = np.mean(x_proc, axis=0)
        std_x = np.std(x_proc, axis=0, ddof=1)
        
        # Create scaled versions, initially as float to allow NaNs
        x_scaled = np.zeros_like(x_proc, dtype=float)
        y_scaled = np.zeros_like(y_proc, dtype=float)

        for j in range(x_proc.shape[1]):
            if std_x[j] > 1e-12: # Effectively non-zero std for reference x column
                x_scaled[:,j] = (x_proc[:,j] - mean_x[j]) / std_x[j]
                y_scaled[:,j] = (y_proc[:,j] - mean_x[j]) / std_x[j]
            else: 
                # R's scale() would produce NaNs for x_col if sd is 0.
                # And y_col scaled by 0 or NA scale factor also becomes NaN/Inf.
                # These dimensions are effectively ignored by energy::edist.
                x_scaled[:,j] = np.nan 
                y_scaled[:,j] = np.nan
        
        x_proc = x_scaled
        y_proc = y_scaled

        # Re-filter NaNs that might have been introduced by scaling constant columns
        x_finite_mask_scaled = np.all(np.isfinite(x_proc), axis=1)
        y_finite_mask_scaled = np.all(np.isfinite(y_proc), axis=1)
        common_finite_mask_scaled = x_finite_mask_scaled & y_finite_mask_scaled
        
        x_proc = x_proc[common_finite_mask_scaled]
        y_proc = y_proc[common_finite_mask_scaled]

        if x_proc.shape[0] < 2 or y_proc.shape[0] < 2:
            return np.nan # Not enough data after scaling and NaN removal
    
    if n_cases is not None:
        n_current_proc = x_proc.shape[0] 
        actual_n_cases = min(n_current_proc, n_cases)
        if actual_n_cases >= 1: 
            idx_x = np.random.choice(x_proc.shape[0], actual_n_cases, replace=False)
            idx_y = np.random.choice(y_proc.shape[0], actual_n_cases, replace=False) # y_proc has same length as x_proc here
            x_proc = x_proc[idx_x]
            y_proc = y_proc[idx_y]
        else:
            return np.nan 
            
    if x_proc.shape[0] < 1 or y_proc.shape[0] < 1 or x_proc.shape[1] == 0: 
        # If all dimensions were removed due to constant columns in x, shape[1] could be 0
        return np.nan

    d_xx = cdist(x_proc, x_proc, 'euclidean')
    d_yy = cdist(y_proc, y_proc, 'euclidean')
    d_xy = cdist(x_proc, y_proc, 'euclidean')

    term1 = np.mean(d_xy) if d_xy.size > 0 else 0.0
    term2 = 0.5 * (np.mean(d_xx) if d_xx.size > 0 else 0.0)
    term3 = 0.5 * (np.mean(d_yy) if d_yy.size > 0 else 0.0)
    
    result = term1 - term2 - term3
    return result


def MRS(o_c, m_c, m_p, o_c_chol=None, o_p_chol=None, m_c_chol=None, m_p_chol=None):
    """Multivariate rescaling based on Cholesky decomposition"""
    global MRS_DEBUG_PRINT_PY # Allow modification of global
    o_c_arr = np.asarray(o_c)
    m_c_arr = np.asarray(m_c)
    m_p_arr = np.asarray(m_p)

    # Center based on multivariate means
    o_c_mean = np.mean(o_c_arr, axis=0)
    m_c_mean = np.mean(m_c_arr, axis=0)
    m_p_mean = np.mean(m_p_arr, axis=0)
    
    o_c_cent = o_c_arr - o_c_mean
    m_c_cent = m_c_arr - m_c_mean
    m_p_cent = m_p_arr - m_p_mean
    
    # Cholesky decomposition
    # R's chol is upper by default. Pass lower=False to np.linalg.cholesky
    if o_c_chol is None:
        o_c_chol = cholesky(nearPD(np.cov(o_c_cent, rowvar=False, ddof=1)), lower=False) 
    if o_p_chol is None: # R uses cov(o.c) for o.p.chol if not given
        o_p_chol = cholesky(nearPD(np.cov(o_c_cent, rowvar=False, ddof=1)), lower=False)
    if m_c_chol is None:
        m_c_chol = cholesky(nearPD(np.cov(m_c_cent, rowvar=False, ddof=1)), lower=False)
    if m_p_chol is None: # R uses cov(m.c) for m.p.chol if not given
        m_p_chol = cholesky(nearPD(np.cov(m_c_cent, rowvar=False, ddof=1)), lower=False)
        
    # Bias correction factors: R's solve(A) %*% B is A_inv @ B
    # Python's solve(A, B) solves Ax = B for x. So x = A_inv @ B.
    # This matches R's solve(m.c.chol) %*% o.c.chol
    # R: mbcfactor <- solve(m.c.chol) %*% o.c.chol  (U_mc^-1 @ U_oc)
    # Python: mbcfactor = solve(m_c_chol, o_c_chol) (U_mc^-1 @ U_oc)
    mbcfactor = solve(m_c_chol, o_c_chol) 
    mbpfactor = solve(m_p_chol, o_p_chol)

    if MRS_DEBUG_PRINT_PY:
        print("--- MRS DEBUG (Python) ---")
        print("mbcfactor (head):\n", mbcfactor[:min(3, mbcfactor.shape[0]), :min(3, mbcfactor.shape[1])])
        print("mbpfactor (head):\n", mbpfactor[:min(3, mbpfactor.shape[0]), :min(3, mbpfactor.shape[1])])
        print("--- END MRS DEBUG (Python) ---")
    
    # Multivariate bias correction
    mbc_c = m_c_cent @ mbcfactor
    mbc_p = m_p_cent @ mbpfactor
    
    # Recenter and account for change in means
    mbc_c = mbc_c + o_c_mean
    mbc_p = mbc_p + o_c_mean + (m_p_mean - m_c_mean)
    
    return {'mhat_c': mbc_c, 'mhat_p': mbc_p}

def rot_random(k):
    """Generate random orthogonal rotation matrix"""
    # R's `qr.Q(qr(matrix(rnorm(k*k), ncol=k)))`
    # Python's qr returns Q,R. R's qr.Q extracts Q.
    # R's qr.R ensures diag(R) >= 0 by multiplying Q columns by sign(diag(R)).
    # Python's np.linalg.qr does not guarantee this.
    # The R code `rot <- Q %*% diag(diagR/abs(diagR))` implements this sign correction.
    
    rand_mat = np.random.normal(size=(k, k))
    q_mat, r_mat = qr(rand_mat)
    
    diag_r = np.diag(r_mat)
    # Add epsilon to avoid division by zero if diag_r element is zero
    sign_correction_diag = diag_r / (np.abs(diag_r) + np.finfo(float).eps) 
    
    # If a diagonal element of R was zero, sign_correction_diag will be zero.
    # R's behavior: if diagR[i] is 0, diagR[i]/abs(diagR[i]) is NaN, then Q %*% diag(...) has NaNs.
    # This should be rare for a random matrix.
    # If it happens, R's rot.random would produce NaNs.
    # Let's ensure non-NaN for zero diag_r: if diag_r[i] == 0, use 1.0 for sign.
    sign_correction_diag[np.abs(diag_r) < np.finfo(float).eps] = 1.0

    rot = q_mat @ np.diag(sign_correction_diag)
    return rot

def MBCr(o_c, m_c, m_p, iter=20, cor_thresh=1e-4, ratio_seq=None, trace=0.05,
         trace_calc=0.5*0.05, jitter_factor=0, n_tau=None, ratio_max=2,
         ratio_max_trace=10*0.05, ties='first', qmap_precalc=False, # ties for initial rank and loop rank
         silent=False, subsample=None, pp_type='linear'):
    """Multivariate quantile mapping bias correction (Spearman correlation)"""
    global MRS_DEBUG_PRINT_PY # Allow modification of global
    MBCR_ITER_DEBUG_PY = True # Set to TRUE for Python debug prints in MBCr

    n_vars = o_c.shape[1]
    o_c_arr = np.asarray(o_c)
    m_c_arr = np.asarray(m_c)
    m_p_arr = np.asarray(m_p)

    if ratio_seq is None:
        ratio_seq_list = [False] * n_vars
    elif np.isscalar(ratio_seq):
        ratio_seq_list = [ratio_seq] * n_vars
    else:
        ratio_seq_list = list(ratio_seq)

    # Ensure trace params are lists/arrays of correct length
    def ensure_list_len(param, length, default_val_if_scalar):
        if np.isscalar(param):
            return [param] * length
        elif len(param) == length:
            return list(param)
        else: # Should not happen if called from testcode with correct params
            # print(f"Warning: param length mismatch. Using default for {param}.")
            return [default_val_if_scalar] * length 

    trace_calc_list = ensure_list_len(trace_calc, n_vars, 0.5*0.05)
    trace_list = ensure_list_len(trace, n_vars, 0.05)
    jitter_factor_list = ensure_list_len(jitter_factor, n_vars, 0)
    ratio_max_list = ensure_list_len(ratio_max, n_vars, 2)
    ratio_max_trace_list = ensure_list_len(ratio_max_trace, n_vars, 10*0.05)
        
    m_c_qmap = m_c_arr.copy()
    m_p_qmap = m_p_arr.copy()
    
    if not qmap_precalc:
        for i in range(n_vars):
            fit_qmap = QDM(o_c_arr[:,i], m_c_arr[:,i], m_p_arr[:,i],
                          ratio=ratio_seq_list[i], trace_calc=trace_calc_list[i],
                          trace=trace_list[i], jitter_factor=jitter_factor_list[i],
                          n_tau=n_tau, ratio_max=ratio_max_list[i],
                          ratio_max_trace=ratio_max_trace_list[i],
                          subsample=subsample, pp_type=pp_type, ties=ties) # Pass ties to QDM for ECBC if used
            m_c_qmap[:,i] = fit_qmap['mhat_c']
            m_p_qmap[:,i] = fit_qmap['mhat_p']

    # R's rank ties.method='first' is 'ordinal' in scipy
    rank_method_loop = 'ordinal' if ties == 'first' else ties

    o_c_r = np.apply_along_axis(rankdata, 0, o_c_arr, method=rank_method_loop)
    # R code ranks m.c, not m.c.qmap for the iteration start.
    # m.c.r <- apply(m.c, 2, rank, ties.method=ties)
    m_c_r = np.apply_along_axis(rankdata, 0, m_c_arr, method=rank_method_loop)
    m_p_r = np.apply_along_axis(rankdata, 0, m_p_arr, method=rank_method_loop)

    m_c_i_rank_check = m_c_r.copy() # For checking rank convergence: R's m.c.i <- m.c.r
    
    cor_i = np.corrcoef(m_c_r, rowvar=False, ddof=1) # Spearman uses ranks, then Pearson on ranks. ddof=1 for sample cov.
    cor_i[np.isnan(cor_i)] = 0 
    
    if MBCR_ITER_DEBUG_PY:
        print("--- MBCr DEBUG PY: Before Loop ---")
        print("Summary o_c_r (head):\n", pd.DataFrame(o_c_r[:,:min(3,o_c_r.shape[1])]).describe())
        print(o_c_r[:2,:min(3,o_c_r.shape[1])])
        print("Summary m_c_r (initial, head):\n", pd.DataFrame(m_c_r[:,:min(3,m_c_r.shape[1])]).describe())
        print(m_c_r[:2,:min(3,m_c_r.shape[1])])
        print("Summary m_p_r (initial, head):\n", pd.DataFrame(m_p_r[:,:min(3,m_p_r.shape[1])]).describe())
        print(m_p_r[:2,:min(3,m_p_r.shape[1])])
        
        cov_o_c_r_raw = np.cov(o_c_r, rowvar=False, ddof=1)
        print("cov(o_c_r) (raw) head:\n", cov_o_c_r_raw[:min(3,cov_o_c_r_raw.shape[0]), :min(3,cov_o_c_r_raw.shape[1])])
        npd_o_c_r_mat = nearPD(cov_o_c_r_raw)
        print("nearPD(cov(o_c_r)) head:\n", npd_o_c_r_mat[:min(3,npd_o_c_r_mat.shape[0]), :min(3,npd_o_c_r_mat.shape[1])])

    # Iterative MBC/reranking
    # R's chol is upper by default. Pass lower=False to np.linalg.cholesky
    # ddof=1 for sample covariance matrix, as R's cov() default.
    o_c_cov_r = nearPD(np.cov(o_c_r, rowvar=False, ddof=1))
    o_c_chol = cholesky(o_c_cov_r, lower=False) 
    o_p_chol = o_c_chol 
    
    if MBCR_ITER_DEBUG_PY:
        print("o_c_chol head:\n", o_c_chol[:min(3,o_c_chol.shape[0]), :min(3,o_c_chol.shape[1])])

    for k_iter_loop in range(iter): 
        if MBCR_ITER_DEBUG_PY and k_iter_loop == 0:
            print(f"--- MBCr DEBUG PY: Iteration {k_iter_loop+1} ---")
            print("Summary m_c_r (start of iter, head):\n", pd.DataFrame(m_c_r[:,:min(3,m_c_r.shape[1])]).describe())
            print(m_c_r[:2,:min(3,m_c_r.shape[1])])
            
            cov_m_c_r_raw = np.cov(m_c_r, rowvar=False, ddof=1)
            print("cov(m_c_r) (raw) head:\n", cov_m_c_r_raw[:min(3,cov_m_c_r_raw.shape[0]), :min(3,cov_m_c_r_raw.shape[1])])
            npd_m_c_r_mat = nearPD(cov_m_c_r_raw)
            print("nearPD(cov(m_c_r)) head:\n", npd_m_c_r_mat[:min(3,npd_m_c_r_mat.shape[0]), :min(3,npd_m_c_r_mat.shape[1])])

        m_c_cov_r = nearPD(np.cov(m_c_r, rowvar=False, ddof=1))
        m_c_chol = cholesky(m_c_cov_r, lower=False)
        m_p_chol = m_c_chol 

        if MBCR_ITER_DEBUG_PY and k_iter_loop == 0:
            print("m_c_chol head:\n", m_c_chol[:min(3,m_c_chol.shape[0]), :min(3,m_c_chol.shape[1])])
            MRS_DEBUG_PRINT_PY = True # Enable MRS debug prints

        fit_mbc = MRS(o_c_r, m_c_r, m_p_r, o_c_chol=o_c_chol,
                     o_p_chol=o_p_chol, m_c_chol=m_c_chol, m_p_chol=m_p_chol)
        
        if MBCR_ITER_DEBUG_PY and k_iter_loop == 0:
            MRS_DEBUG_PRINT_PY = False # Disable MRS debug prints
            print("Summary fit_mbc['mhat_c'] (after MRS, head):\n", pd.DataFrame(fit_mbc['mhat_c'][:,:min(3,fit_mbc['mhat_c'].shape[1])]).describe())
            print(fit_mbc['mhat_c'][:2,:min(3,fit_mbc['mhat_c'].shape[1])])


        m_c_r = np.apply_along_axis(rankdata, 0, fit_mbc['mhat_c'], method=rank_method_loop)
        m_p_r = np.apply_along_axis(rankdata, 0, fit_mbc['mhat_p'], method=rank_method_loop)

        if MBCR_ITER_DEBUG_PY and k_iter_loop == 0:
            print("Summary m_c_r (after re-ranking, head):\n", pd.DataFrame(m_c_r[:,:min(3,m_c_r.shape[1])]).describe())
            print(m_c_r[:2,:min(3,m_c_r.shape[1])])


        cor_j = np.corrcoef(m_c_r, rowvar=False, ddof=1)
        cor_j[np.isnan(cor_j)] = 0
        cor_diff = np.mean(np.abs(cor_j - cor_i))
        cor_i = cor_j
            
        if not silent:
            # R prints 1-based iteration, mean equality of ranks, cor_diff
            print(f"{k_iter_loop+1} {np.mean(m_c_r == m_c_i_rank_check):.6f} {cor_diff:.6g} ", end='')
        
        if MBCR_ITER_DEBUG_PY and k_iter_loop == 0:
            print(f"\ncor_diff for iter 1: {cor_diff}\n")
            
        if cor_diff < cor_thresh:
            break
        if np.array_equal(m_c_r, m_c_i_rank_check): 
            break
            
        m_c_i_rank_check = m_c_r.copy() 
        
    if not silent:
        print()
        
    m_c_r_final = np.empty_like(m_c_r, dtype=float)
    m_p_r_final = np.empty_like(m_p_r, dtype=float)

    # R: m.c.r[,i] <- sort(m.c.qmap[,i])[m.c.r[,i]]
    # m.c.r here is the final iterated ranks. These are 1-based.
    for i in range(n_vars):
        sorted_qmap_c = np.sort(m_c_qmap[:,i])
        sorted_qmap_p = np.sort(m_p_qmap[:,i])
        
        # Final m_c_r and m_p_r are 1-based ranks. Convert to 0-based indices.
        m_c_ranks_idx = (m_c_r[:,i].astype(int) - 1)
        m_p_ranks_idx = (m_p_r[:,i].astype(int) - 1)

        m_c_ranks_idx = np.clip(m_c_ranks_idx, 0, len(sorted_qmap_c)-1)
        m_p_ranks_idx = np.clip(m_p_ranks_idx, 0, len(sorted_qmap_p)-1)
        
        m_c_r_final[:,i] = sorted_qmap_c[m_c_ranks_idx]
        m_p_r_final[:,i] = sorted_qmap_p[m_p_ranks_idx]
        
    return {'mhat_c': m_c_r_final, 'mhat_p': m_p_r_final}

def MBCp(o_c, m_c, m_p, iter=20, cor_thresh=1e-4, ratio_seq=None, trace=0.05,
         trace_calc=0.5*0.05, jitter_factor=0, n_tau=None, ratio_max=2,
         ratio_max_trace=10*0.05, ties='first', qmap_precalc=False, 
         silent=False, subsample=None, pp_type='linear'):
    """Multivariate quantile mapping bias correction (Pearson correlation)"""
    global MRS_DEBUG_PRINT_PY # Allow modification of global
    MBCP_ITER_DEBUG_PY = True # Set to TRUE for Python debug prints in MBCp

    n_vars = o_c.shape[1]
    o_c_arr = np.asarray(o_c)
    m_c_arr = np.asarray(m_c)
    m_p_arr = np.asarray(m_p)

    if ratio_seq is None:
        ratio_seq_list = [False] * n_vars
    elif np.isscalar(ratio_seq):
        ratio_seq_list = [ratio_seq] * n_vars
    else:
        ratio_seq_list = list(ratio_seq)

    def ensure_list_len(param, length, default_val_if_scalar):
        if np.isscalar(param): return [param] * length
        return list(param)

    trace_calc_list = ensure_list_len(trace_calc, n_vars, 0.5*0.05)
    trace_list = ensure_list_len(trace, n_vars, 0.05)
    jitter_factor_list = ensure_list_len(jitter_factor, n_vars, 0) # For initial QDM
    ratio_max_list = ensure_list_len(ratio_max, n_vars, 2)
    ratio_max_trace_list = ensure_list_len(ratio_max_trace, n_vars, 10*0.05)
        
    m_c_qmap_initial_orig_mc = m_c_arr.copy() # Original m_c for initial QDM
    m_p_qmap_initial_orig_mp = m_p_arr.copy() # Original m_p for initial QDM
    
    # These will hold the results of the initial QDM pass.
    # These are m.c.qmap and m.p.qmap in R, used for the final shuffle.
    m_c_after_initial_qdm = np.empty_like(m_c_arr)
    m_p_after_initial_qdm = np.empty_like(m_p_arr)

    if not qmap_precalc:
        if MBCP_ITER_DEBUG_PY: print("--- MBCp DEBUG PY: Initial QDM ---")
        for i in range(n_vars):
            if MBCP_ITER_DEBUG_PY and i == 0:
                print("Initial QDM for var 0 - o_c_arr head:\n", o_c_arr[:2,i])
                print("Initial QDM for var 0 - m_c_qmap_initial_orig_mc head:\n", m_c_qmap_initial_orig_mc[:2,i])
                print("Initial QDM for var 0 - m_p_qmap_initial_orig_mp head:\n", m_p_qmap_initial_orig_mp[:2,i])
            
            fit_qmap = QDM(o_c_arr[:,i], m_c_qmap_initial_orig_mc[:,i], m_p_qmap_initial_orig_mp[:,i], 
                          ratio=ratio_seq_list[i], trace_calc=trace_calc_list[i],
                          trace=trace_list[i], jitter_factor=jitter_factor_list[i], # Use per-var jitter
                          n_tau=n_tau, ratio_max=ratio_max_list[i],
                          ratio_max_trace=ratio_max_trace_list[i],
                          subsample=subsample, pp_type=pp_type, ties=ties)
            m_c_after_initial_qdm[:,i] = fit_qmap['mhat_c']
            m_p_after_initial_qdm[:,i] = fit_qmap['mhat_p']
            if MBCP_ITER_DEBUG_PY and i == 0:
                print("Initial QDM for var 0 - m_c_after_initial_qdm head:\n", m_c_after_initial_qdm[:2,i])
                print("Initial QDM for var 0 - m_p_after_initial_qdm head:\n", m_p_after_initial_qdm[:2,i])
    else: # If qmap_precalc is True, assume m_c_arr and m_p_arr are already QDM'd
        m_c_after_initial_qdm = m_c_arr.copy()
        m_p_after_initial_qdm = m_p_arr.copy()

    # Iteration starts with QDM-corrected data
    m_c_iter = m_c_after_initial_qdm.copy()
    m_p_iter = m_p_after_initial_qdm.copy()
    
    if MBCP_ITER_DEBUG_PY:
        print("--- MBCp DEBUG PY: Before Loop ---")
        print("Summary m_c_iter (after initial QDM, start of loop) head:\n", pd.DataFrame(m_c_iter[:,:min(3,m_c_iter.shape[1])]).describe())
        print(m_c_iter[:2,:min(3,m_c_iter.shape[1])])

    cor_i = np.corrcoef(m_c_iter, rowvar=False, ddof=1) 
    cor_i[np.isnan(cor_i)] = 0
    
    o_c_cov_mat = np.cov(o_c_arr, rowvar=False, ddof=1) # Covariance of original observations
    o_c_chol = cholesky(nearPD(o_c_cov_mat), lower=False)
    o_p_chol = o_c_chol # Default in R
    
    if MBCP_ITER_DEBUG_PY:
        print("cov(o_c_arr) head:\n", o_c_cov_mat[:min(3,o_c_cov_mat.shape[0]), :min(3,o_c_cov_mat.shape[1])])
        print("o_c_chol head:\n", o_c_chol[:min(3,o_c_chol.shape[0]), :min(3,o_c_chol.shape[1])])
    
    for k_iter_loop in range(iter): 
        if MBCP_ITER_DEBUG_PY and k_iter_loop == 0:
            print(f"--- MBCp DEBUG PY: Iteration {k_iter_loop+1} ---")
            print("Summary m_c_iter (start of iter) head:\n", pd.DataFrame(m_c_iter[:,:min(3,m_c_iter.shape[1])]).describe())
            print(m_c_iter[:2,:min(3,m_c_iter.shape[1])])
            
            cov_m_c_iter_raw = np.cov(m_c_iter, rowvar=False, ddof=1)
            print("cov(m_c_iter) (raw) head:\n", cov_m_c_iter_raw[:min(3,cov_m_c_iter_raw.shape[0]), :min(3,cov_m_c_iter_raw.shape[1])])
            npd_m_c_iter_mat = nearPD(cov_m_c_iter_raw)
            print("nearPD(cov(m_c_iter)) head:\n", npd_m_c_iter_mat[:min(3,npd_m_c_iter_mat.shape[0]), :min(3,npd_m_c_iter_mat.shape[1])])

        m_c_cov = nearPD(np.cov(m_c_iter, rowvar=False, ddof=1))
        m_c_chol = cholesky(m_c_cov, lower=False)
        m_p_chol = m_c_chol # Default in R

        if MBCP_ITER_DEBUG_PY and k_iter_loop == 0:
            print("m_c_chol head:\n", m_c_chol[:min(3,m_c_chol.shape[0]), :min(3,m_c_chol.shape[1])])
            MRS_DEBUG_PRINT_PY = True

        # MRS uses o_c_arr (original obs), m_c_iter, m_p_iter
        fit_mbc = MRS(o_c_arr, m_c_iter, m_p_iter, o_c_chol=o_c_chol,
                     o_p_chol=o_p_chol, m_c_chol=m_c_chol, m_p_chol=m_p_chol)
        
        if MBCP_ITER_DEBUG_PY and k_iter_loop == 0:
            MRS_DEBUG_PRINT_PY = False
            print("Summary fit_mbc['mhat_c'] (after MRS) head:\n", pd.DataFrame(fit_mbc['mhat_c'][:,:min(3,fit_mbc['mhat_c'].shape[1])]).describe())
            print(fit_mbc['mhat_c'][:2,:min(3,fit_mbc['mhat_c'].shape[1])])
        
        m_c_iter_after_mrs = fit_mbc['mhat_c']
        m_p_iter_after_mrs = fit_mbc['mhat_p']
        
        # Inner QDM loop
        for j in range(n_vars):
            if MBCP_ITER_DEBUG_PY and k_iter_loop == 0 and j == 0:
                print(f"--- MBCp DEBUG PY: Iter {k_iter_loop+1}, Inner QDM var {j} ---")
                print("Inner QDM o_c_arr[:,j] head:\n", o_c_arr[:2,j])
                print("Inner QDM m_c_iter_after_mrs[:,j] head:\n", m_c_iter_after_mrs[:2,j])
                print("Inner QDM m_p_iter_after_mrs[:,j] head:\n", m_p_iter_after_mrs[:2,j])

            fit_qmap_inner = QDM(o_c_arr[:,j], m_c_iter_after_mrs[:,j], m_p_iter_after_mrs[:,j], 
                                ratio=False, # R uses ratio=FALSE
                                n_tau=n_tau, pp_type=pp_type, 
                                jitter_factor=0, # R's internal call doesn't seem to pass jitter
                                trace=trace_list[j], trace_calc=trace_calc_list[j], ties=ties) # Pass trace params
            m_c_iter[:,j] = fit_qmap_inner['mhat_c'] 
            m_p_iter[:,j] = fit_qmap_inner['mhat_p'] 
            if MBCP_ITER_DEBUG_PY and k_iter_loop == 0 and j == 0:
                print("Inner QDM m_c_iter[:,j] (updated) head:\n", m_c_iter[:2,j])
                print("Inner QDM m_p_iter[:,j] (updated) head:\n", m_p_iter[:2,j])
        
        if MBCP_ITER_DEBUG_PY and k_iter_loop == 0:
            print("Summary m_c_iter (after inner QDM loop) head:\n", pd.DataFrame(m_c_iter[:,:min(3,m_c_iter.shape[1])]).describe())
            print(m_c_iter[:2,:min(3,m_c_iter.shape[1])])
            
        cor_j = np.corrcoef(m_c_iter, rowvar=False, ddof=1)
        cor_j[np.isnan(cor_j)] = 0
        cor_diff = np.mean(np.abs(cor_j - cor_i))
        cor_i = cor_j
            
        if not silent:
            print(f"{k_iter_loop+1} {cor_diff:.6g} ", end='') 
        
        if MBCP_ITER_DEBUG_PY and k_iter_loop == 0:
            print(f"\ncor_diff for iter 1: {cor_diff}\n")
            
        if cor_diff < cor_thresh:
            break
            
    if not silent:
        print()
        
    # Final shuffle using the initially QDM'd outputs (m_c_after_initial_qdm, m_p_after_initial_qdm)
    # and ranks of the final iterated values (m_c_iter, m_p_iter)
    rank_method_final = 'ordinal' if ties == 'first' else ties

    m_c_final = np.empty_like(m_c_iter)
    m_p_final = np.empty_like(m_p_iter)

    for i in range(n_vars):
        ranks_c = rankdata(m_c_iter[:,i], method=rank_method_final) - 1 
        ranks_p = rankdata(m_p_iter[:,i], method=rank_method_final) - 1 

        sorted_initial_qdm_c = np.sort(m_c_after_initial_qdm[:,i])
        sorted_initial_qdm_p = np.sort(m_p_after_initial_qdm[:,i])
        
        ranks_c = np.clip(ranks_c, 0, len(sorted_initial_qdm_c)-1)
        ranks_p = np.clip(ranks_p, 0, len(sorted_initial_qdm_p)-1)

        m_c_final[:,i] = sorted_initial_qdm_c[ranks_c]
        m_p_final[:,i] = sorted_initial_qdm_p[ranks_p]

    return {'mhat_c': m_c_final, 'mhat_p': m_p_final}

def MBCn(o_c, m_c, m_p, iter=30, ratio_seq=None, trace=0.05,
         trace_calc=0.5*0.05, jitter_factor=0, n_tau=None, ratio_max=2,
         ratio_max_trace=10*0.05, ties='first', qmap_precalc=False, 
         rot_seq=None, silent=False, n_escore=0, return_all=False,
         subsample=None, pp_type='linear'):
    """Multivariate quantile mapping bias correction (N-dimensional pdf transfer)"""
    n_vars = o_c.shape[1]
    o_c_arr = np.asarray(o_c)
    m_c_arr = np.asarray(m_c)
    m_p_arr = np.asarray(m_p)

    if ratio_seq is None:
        ratio_seq_list = [False] * n_vars
    elif np.isscalar(ratio_seq):
        ratio_seq_list = [ratio_seq] * n_vars
    else:
        ratio_seq_list = list(ratio_seq)

    def ensure_list_len(param, length, default_val_if_scalar):
        if np.isscalar(param): return [param] * length
        return list(param)

    trace_calc_list = ensure_list_len(trace_calc, n_vars, 0.5*0.05)
    trace_list = ensure_list_len(trace, n_vars, 0.05)
    # jitter_factor for MBCn is for the QDM calls *inside the rotation loop*
    jitter_factor_list_loop = ensure_list_len(jitter_factor, n_vars, 0) 
    ratio_max_list = ensure_list_len(ratio_max, n_vars, 2) # For initial QDM
    ratio_max_trace_list = ensure_list_len(ratio_max_trace, n_vars, 10*0.05) # For initial QDM
        
    if rot_seq is not None and len(rot_seq) != iter:
        raise ValueError('length(rot_seq) != iter')
        
    escore_iter_values = np.full(iter+2, np.nan) 
    
    # For consistent escore calculation if n_escore is used
    escore_idx_o_c, escore_idx_m_c = None, None
    current_n_escore = 0
    if n_escore > 0:
        current_n_escore = min(o_c_arr.shape[0], m_c_arr.shape[0], n_escore)
        if current_n_escore > 0:
            escore_idx_o_c = np.random.choice(o_c_arr.shape[0], current_n_escore, replace=False)
            escore_idx_m_c = np.random.choice(m_c_arr.shape[0], current_n_escore, replace=False)
            escore_iter_values[0] = escore(o_c_arr[escore_idx_o_c], m_c_arr[escore_idx_m_c], scale_x=True)
            if not silent: print(f"RAW {escore_iter_values[0]:.6g} : ", end='')
        else: escore_iter_values[0] = np.nan

    # Initial QDM mapping (applied once to original m_c, m_p)
    # These are m_c_qmap and m_p_qmap in R
    m_c_after_initial_qdm = m_c_arr.copy()
    m_p_after_initial_qdm = m_p_arr.copy()

    if not qmap_precalc:
        # Jitter for initial QDM: R uses jitter.factor (scalar or vector) from MBCn's params.
        # This is distinct from the jitter_factor_list_loop for internal QDM.
        # The R code implies the main jitter_factor parameter of MBCn is for this first QDM pass.
        # Let's assume the `jitter_factor` parameter of MBCn is for this initial QDM.
        jitter_factor_initial_qdm = ensure_list_len(jitter_factor, n_vars, 0)

        for i in range(n_vars):
            fit_qmap = QDM(o_c_arr[:,i], m_c_arr[:,i], m_p_arr[:,i], 
                          ratio=ratio_seq_list[i], trace_calc=trace_calc_list[i],
                          trace=trace_list[i], jitter_factor=jitter_factor_initial_qdm[i], 
                          n_tau=n_tau, ratio_max=ratio_max_list[i],
                          ratio_max_trace=ratio_max_trace_list[i],
                          subsample=subsample, pp_type=pp_type, ties=ties)
            m_c_after_initial_qdm[:,i] = fit_qmap['mhat_c']
            m_p_after_initial_qdm[:,i] = fit_qmap['mhat_p']
    
    # Iteration starts with these QDM results
    m_c_iter = m_c_after_initial_qdm.copy()
    m_p_iter = m_p_after_initial_qdm.copy()
    
    if n_escore > 0: 
        if current_n_escore > 0:
            escore_iter_values[1] = escore(o_c_arr[escore_idx_o_c], m_c_iter[escore_idx_m_c], scale_x=True)
            if not silent: print(f"QDM {escore_iter_values[1]:.6g} : ", end='')
        else: escore_iter_values[1] = np.nan
    
    # Standardize observations (o_c_arr) using ddof=1 for std
    o_c_mean_stdize = np.mean(o_c_arr, axis=0)
    o_c_sdev_stdize = np.std(o_c_arr, axis=0, ddof=1) 
    o_c_sdev_stdize[o_c_sdev_stdize < np.finfo(float).eps] = 1.0 
    o_c_stdized = (o_c_arr - o_c_mean_stdize) / o_c_sdev_stdize
    
    m_iter_storage = [] 
    
    # Store the mean and sdev of the first m_c_p (which is m_c_iter, m_p_iter before loop)
    # for final rescaling. R: m.c.p <- rbind(m.c, m.p) then standardized.
    # The m.c, m.p here are the QDM'd versions.
    m_c_p_initial_iter = np.vstack((m_c_iter, m_p_iter))
    m_c_p_mean_for_final_rescale = np.mean(m_c_p_initial_iter, axis=0)
    m_c_p_sdev_for_final_rescale = np.std(m_c_p_initial_iter, axis=0, ddof=1)
    m_c_p_sdev_for_final_rescale[m_c_p_sdev_for_final_rescale < np.finfo(float).eps] = 1.0

    # Standardize m_c_iter and m_p_iter using their combined stats for the loop
    m_c_p_current_iter_combined = np.vstack((m_c_iter, m_p_iter))
    m_c_p_mean_stdize_loop = np.mean(m_c_p_current_iter_combined, axis=0)
    m_c_p_sdev_stdize_loop = np.std(m_c_p_current_iter_combined, axis=0, ddof=1)
    m_c_p_sdev_stdize_loop[m_c_p_sdev_stdize_loop < np.finfo(float).eps] = 1.0
    
    m_c_stdized_iter = (m_c_iter - m_c_p_mean_stdize_loop) / m_c_p_sdev_stdize_loop
    m_p_stdized_iter = (m_p_iter - m_c_p_mean_stdize_loop) / m_c_p_sdev_stdize_loop

    for k_iter_loop in range(iter): 
        if not silent: print(f"{k_iter_loop+1} ", end='') 
            
        # Concatenate o_c_stdized and the currently standardized m_c_stdized_iter, m_p_stdized_iter
        Xt = np.vstack((o_c_stdized, m_c_stdized_iter, m_p_stdized_iter))
            
        current_rot = rot_random(n_vars) if rot_seq is None else rot_seq[k_iter_loop]
            
        Z = Xt @ current_rot
        Z_o_c = Z[:o_c_arr.shape[0]]
        Z_m_c_iter_rot = Z[o_c_arr.shape[0] : o_c_arr.shape[0]+m_c_iter.shape[0]]
        Z_m_p_iter_rot = Z[o_c_arr.shape[0]+m_c_iter.shape[0]:]
        
        # Bias correct rotated variables using QDM (ratio=FALSE)
        # Jitter for these internal QDMs is jitter_factor_list_loop
        for j in range(n_vars):
            Z_qdm = QDM(Z_o_c[:,j], Z_m_c_iter_rot[:,j], Z_m_p_iter_rot[:,j], ratio=False,
                       jitter_factor=jitter_factor_list_loop[j], 
                       n_tau=n_tau, pp_type=pp_type,
                       trace=trace_list[j], trace_calc=trace_calc_list[j], ties=ties)
            Z_m_c_iter_rot[:,j] = Z_qdm['mhat_c']
            Z_m_p_iter_rot[:,j] = Z_qdm['mhat_p']
            
        # Rotate back the standardized model data
        m_c_stdized_iter = Z_m_c_iter_rot @ current_rot.T
        m_p_stdized_iter = Z_m_p_iter_rot @ current_rot.T
        # Xt for next iter is o_c_stdized, m_c_stdized_iter, m_p_stdized_iter (already updated)
        
        # For escore and return_all, need to unstandardize m_c_stdized_iter
        # using the m_c_p_sdev_stdize_loop and m_c_p_mean_stdize_loop
        m_c_temp_unstd = m_c_stdized_iter * m_c_p_sdev_stdize_loop + m_c_p_mean_stdize_loop
        
        if n_escore > 0:
            if current_n_escore > 0:
                # Escore uses original o_c_arr and the unstandardized m_c_temp_unstd
                escore_val = escore(o_c_arr[escore_idx_o_c], m_c_temp_unstd[escore_idx_m_c], scale_x=True)
                escore_iter_values[k_iter_loop+2] = escore_val
                if not silent: print(f"{escore_val:.6g} : ", end='')
            else: escore_iter_values[k_iter_loop+2] = np.nan
                
        if return_all: 
            m_p_temp_unstd = m_p_stdized_iter * m_c_p_sdev_stdize_loop + m_c_p_mean_stdize_loop
            m_iter_storage.append({'m_c': m_c_temp_unstd.copy(), 'm_p': m_p_temp_unstd.copy()})
            
    if not silent: print()
        
    # Rescale m_c_stdized_iter and m_p_stdized_iter back to original units
    # using the mean/sdev of the *initial* QDM'd m_c and m_p (before the loop).
    # R: m.c <- sweep(sweep(m.c, 2, attr(m.c.p, 'scaled:scale'), '*'), 2, attr(m.c.p, 'scaled:center'), '+')
    # attr(m.c.p, 'scaled:scale') are m_c_p_sdev from before loop.
    m_c_final_before_shuffle = m_c_stdized_iter * m_c_p_sdev_for_final_rescale + m_c_p_mean_for_final_rescale
    m_p_final_before_shuffle = m_p_stdized_iter * m_c_p_sdev_for_final_rescale + m_c_p_mean_for_final_rescale
    
    # Final shuffle using original QDM outputs (m_c_after_initial_qdm, m_p_after_initial_qdm)
    # and ranks of the final iterated (but rescaled) values.
    rank_method_final = 'ordinal' if ties == 'first' else ties

    m_c_output = np.empty_like(m_c_final_before_shuffle)
    m_p_output = np.empty_like(m_p_final_before_shuffle)

    for i in range(n_vars):
        ranks_c = rankdata(m_c_final_before_shuffle[:,i], method=rank_method_final) - 1
        ranks_p = rankdata(m_p_final_before_shuffle[:,i], method=rank_method_final) - 1

        sorted_initial_qdm_c = np.sort(m_c_after_initial_qdm[:,i])
        sorted_initial_qdm_p = np.sort(m_p_after_initial_qdm[:,i])
        
        ranks_c = np.clip(ranks_c, 0, len(sorted_initial_qdm_c)-1)
        ranks_p = np.clip(ranks_p, 0, len(sorted_initial_qdm_p)-1)

        m_c_output[:,i] = sorted_initial_qdm_c[ranks_c]
        m_p_output[:,i] = sorted_initial_qdm_p[ranks_p]
        
    escore_iter_dict = dict(zip(['RAW', 'QM'] + [k for k in range(iter)], escore_iter_values))
    
    return {'mhat_c': m_c_output, 'mhat_p': m_p_output, 'escore_iter': escore_iter_dict,
            'm_iter': m_iter_storage}

def R2D2(o_c, m_c, m_p, ref_column=0, ratio_seq=None, trace=0.05,
         trace_calc=0.5*0.05, jitter_factor=0, n_tau=None, ratio_max=2,
         ratio_max_trace=10*0.05, ties='first', qmap_precalc=False,
         subsample=None, pp_type='linear'):
    """Multivariate bias correction using nearest neighbor rank resampling"""
    o_c_arr = np.asarray(o_c)
    m_c_arr = np.asarray(m_c)
    m_p_arr = np.asarray(m_p)

    if o_c_arr.shape[0] != m_c_arr.shape[0] or o_c_arr.shape[0] != m_p_arr.shape[0]:
        raise ValueError("R2D2 requires data samples of equal length")
        
    n_vars = o_c_arr.shape[1]

    if ratio_seq is None:
        ratio_seq_list = [False] * n_vars
    elif np.isscalar(ratio_seq):
        ratio_seq_list = [ratio_seq] * n_vars
    else:
        ratio_seq_list = list(ratio_seq)
    
    def ensure_list_len(param, length, default_val_if_scalar):
        if np.isscalar(param): return [param] * length
        return list(param)

    trace_calc_list = ensure_list_len(trace_calc, n_vars, 0.5*0.05)
    trace_list = ensure_list_len(trace, n_vars, 0.05)
    jitter_factor_list = ensure_list_len(jitter_factor, n_vars, 0)
    ratio_max_list = ensure_list_len(ratio_max, n_vars, 2)
    ratio_max_trace_list = ensure_list_len(ratio_max_trace, n_vars, 10*0.05)
        
    m_c_qmap = m_c_arr.copy()
    m_p_qmap = m_p_arr.copy()
    
    if not qmap_precalc:
        for i in range(n_vars):
            fit_qmap = QDM(o_c_arr[:,i], m_c_arr[:,i], m_p_arr[:,i],
                          ratio=ratio_seq_list[i], trace_calc=trace_calc_list[i],
                          trace=trace_list[i], jitter_factor=jitter_factor_list[i],
                          n_tau=n_tau, ratio_max=ratio_max_list[i],
                          ratio_max_trace=ratio_max_trace_list[i],
                          subsample=subsample, pp_type=pp_type, ties=ties)
            m_c_qmap[:,i] = fit_qmap['mhat_c']
            m_p_qmap[:,i] = fit_qmap['mhat_p']

    rank_method_r2d2 = 'ordinal' if ties == 'first' else ties

    o_c_r = np.apply_along_axis(rankdata, 0, o_c_arr, method=rank_method_r2d2)
    # Ranks are calculated on QDM'd data in R: apply(m.c.qmap, 2, rank, ...)
    m_c_r = np.apply_along_axis(rankdata, 0, m_c_qmap, method=rank_method_r2d2)
    m_p_r = np.apply_along_axis(rankdata, 0, m_p_qmap, method=rank_method_r2d2)

    nn = NearestNeighbors(n_neighbors=1, algorithm='auto')
    o_c_r_refcol = o_c_r[:,ref_column].reshape(-1,1)
    m_c_r_refcol = m_c_r[:,ref_column].reshape(-1,1)
    m_p_r_refcol = m_p_r[:,ref_column].reshape(-1,1)

    nn.fit(o_c_r_refcol)
    
    # R: nn.c.r <- rank(knnx.index(o.c.r[,ref.column], query=m.c.r[,ref.column], k=1), ties.method='random')
    # knnx.index returns the indices of the nearest neighbors in o_c_r_refcol.
    # Then these indices are ranked with random tie-breaking.
    # This means if multiple query points map to the same neighbor index, their ranks are randomized.
    
    nn_c_indices_from_o_c_r = nn.kneighbors(m_c_r_refcol, return_distance=False).flatten()
    nn_p_indices_from_o_c_r = nn.kneighbors(m_p_r_refcol, return_distance=False).flatten()

    # These are 0-based indices into o_c_r.
    # R's `rank(..., ties.method='random')` applied to these indices.
    # This step is crucial and a bit unusual. It's not just taking o_c_r[nn_c_indices_from_o_c_r].
    # It's ranking the chosen indices themselves.
    # Let's call these "ranked_neighbor_selector_indices"
    ranked_neighbor_selector_indices_c = rankdata(nn_c_indices_from_o_c_r, method='random')
    ranked_neighbor_selector_indices_p = rankdata(nn_p_indices_from_o_c_r, method='random')

    # These ranked_neighbor_selector_indices are 1-based. Convert to 0-based for Python.
    # And these are used to select *rows* from o_c_r.
    # R: new.c.r <- o.c.r[nn.c.r,,drop=FALSE]
    # So, o_c_r is indexed by these "ranked_neighbor_selector_indices".
    new_c_r = o_c_r[(ranked_neighbor_selector_indices_c - 1).astype(int)]
    new_p_r = o_c_r[(ranked_neighbor_selector_indices_p - 1).astype(int)]
    
    r2d2_c = np.empty_like(m_c_qmap)
    r2d2_p = np.empty_like(m_p_qmap)
    
    for i in range(n_vars):
        sorted_m_c_qmap_i = np.sort(m_c_qmap[:,i]) # Sort the QDM'd data
        sorted_m_p_qmap_i = np.sort(m_p_qmap[:,i])
        
        # new_c_r[:,i] contains the ranks (1-based) from o_c_r that should be used.
        # Convert these ranks to 0-based indices for sorted_m_c_qmap_i.
        idx_c = new_c_r[:,i].astype(int) - 1
        idx_p = new_p_r[:,i].astype(int) - 1

        idx_c = np.clip(idx_c, 0, len(sorted_m_c_qmap_i)-1)
        idx_p = np.clip(idx_p, 0, len(sorted_m_p_qmap_i)-1)

        r2d2_c[:,i] = sorted_m_c_qmap_i[idx_c]
        r2d2_p[:,i] = sorted_m_p_qmap_i[idx_p]
        
    return {'mhat_c': r2d2_c, 'mhat_p': r2d2_p}
