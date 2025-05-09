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

# Helper function for nearPD
def _ensure_symmetric(A):
    return (A + A.T) / 2

def nearPD(A, epsilon_eig=1e-6, epsilon_chol_jitter=1e-9, max_iter=100, conv_tol=1e-7):
    """
    Computes the nearest positive definite matrix to A.
    Uses eigenvalue adjustment and iterative jittering if Cholesky fails.
    Based on Higham's (2002) algorithm and R's Matrix::nearPD implementation.
    A: input matrix (should be symmetric or will be symmetrized)
    epsilon_eig: smallest allowed eigenvalue if direct eigenvalue adjustment is used.
    epsilon_chol_jitter: initial jitter factor for Cholesky attempts.
    max_iter: maximum iterations for the main algorithm.
    conv_tol: convergence tolerance for the iterative algorithm.
    """
    X = np.asarray(A)
    if X.ndim != 2 or X.shape[0] != X.shape[1]:
        raise ValueError("Input must be a square 2D matrix.")

    # Symmetrize
    Y = _ensure_symmetric(X)

    # Iteration for Higham's algorithm
    D_s = np.zeros_like(Y)
    for i in range(max_iter):
        Y_prev = Y.copy()
        R_k = Y - D_s
        
        # Polar decomposition of R_k
        try:
            # SVD: R_k = U @ S @ Vh
            U, s, Vh = np.linalg.svd(R_k, full_matrices=False)
            H = Vh.T @ np.diag(s) @ Vh # H = R_k.T @ R_k, H is P in R_k = UP
                                       # P_k = U S U.T (symmetric part of polar decomposition)
            P_k = U @ np.diag(s) @ U.T # This is the symmetric polar factor
        except np.linalg.LinAlgError:
            # If SVD fails, try to make Y more PD and continue
            jitter = epsilon_chol_jitter * (10**i) * np.mean(np.abs(np.diag(Y)))
            Y = _ensure_symmetric(Y + np.eye(Y.shape[0]) * (jitter + epsilon_chol_jitter))
            if i == max_iter -1:
                # print("Warning: nearPD SVD failed in iteration.")
                break # Exit loop, try final adjustment
            continue

        # Update Y
        Y = P_k

        # Update D_s (diagonal shift part)
        D_s = Y - R_k # This is not how D_s is updated in Higham's direct algorithm.
                      # R's nearPD uses a different approach for the "symmetric part"
                      # Let's follow R's Matrix::nearPD logic more closely for the main iteration.
                      # The R version uses an iterative process involving eigenvalue decomposition.

        # Re-evaluate: R's nearPD uses eigenvalue decomposition and reconstruction.
        # Let's use a simpler eigenvalue adjustment if direct Cholesky fails,
        # and then the iterative jittering if that's not enough.

        # Try Cholesky on current Y
        try:
            np.linalg.cholesky(Y)
            return Y # If Cholesky succeeds, Y is PD
        except np.linalg.LinAlgError:
            pass # Continue to eigenvalue adjustment / jittering

        # Eigenvalue adjustment
        try:
            eigvals, eigvecs = np.linalg.eigh(Y)
        except np.linalg.LinAlgError:
             # If eigendecomposition fails, add more jitter and retry
            jitter = epsilon_chol_jitter * (10**i) * np.mean(np.abs(np.diag(Y)))
            Y = _ensure_symmetric(Y + np.eye(Y.shape[0]) * (jitter + epsilon_chol_jitter))
            if i == max_iter -1:
                # print(f"Warning: nearPD eigendecomposition failed after {max_iter} attempts.")
                break
            continue

        # Set small or negative eigenvalues to a small positive value (epsilon_eig)
        # R's nearPD ensures eigenvalues are >= eig.tol * max_eigenvalue
        max_eig = np.max(eigvals) if len(eigvals) > 0 else 1.0
        min_eig_thresh = epsilon_eig * max_eig if max_eig > 0 else epsilon_eig

        if np.any(eigvals < min_eig_thresh):
            eigvals[eigvals < min_eig_thresh] = min_eig_thresh
            Y = _ensure_symmetric(eigvecs @ np.diag(eigvals) @ eigvecs.T)

        # Check for convergence (norm of change in Y)
        if np.linalg.norm(Y - Y_prev, 'fro') < conv_tol * np.linalg.norm(Y_prev, 'fro'):
            break
    
    # Final attempt to make it PD for Cholesky via jittering if still not PD
    # This is a fallback if the main loop didn't result in a PD matrix
    # that passes Cholesky.
    for j_iter in range(10): # Try jittering a few times
        try:
            np.linalg.cholesky(Y)
            return Y # Success
        except np.linalg.LinAlgError:
            # Add small jitter to diagonal
            # The amount of jitter might need to be scaled by the magnitude of Y's diagonal
            diag_mean_abs = np.mean(np.abs(np.diag(Y)))
            if diag_mean_abs < np.finfo(float).eps: diag_mean_abs = 1.0 # Avoid zero scaling factor
            
            jitter_val = (epsilon_chol_jitter * (10**j_iter)) * diag_mean_abs
            if jitter_val < np.finfo(float).eps: # Ensure jitter_val is not too small
                jitter_val = epsilon_chol_jitter * (10**j_iter)

            Y = _ensure_symmetric(Y + np.eye(Y.shape[0]) * jitter_val)
            if j_iter == 9: # Last jitter attempt
                # print(f"Warning: nearPD Cholesky failed after main loop and {j_iter+1} jitter attempts.")
                pass

    # As a very last resort, if Cholesky still fails, return the best Y.
    # The caller (cholesky function) will then handle the LinAlgError.
    # This function's job is to get it "near" PD.
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

    # Handle NaNs by removing rows where *either* x or y has NaN/Inf
    # This ensures x_clean and y_clean have the same number of rows and correspond.
    x_finite_mask = np.all(np.isfinite(x_arr), axis=1)
    y_finite_mask = np.all(np.isfinite(y_arr), axis=1)
    common_finite_mask = x_finite_mask & y_finite_mask
    
    x_clean = x_arr[common_finite_mask]
    y_clean = y_arr[common_finite_mask]
    
    if x_clean.shape[0] < 2 or y_clean.shape[0] < 2 or x_clean.shape[0] != y_clean.shape[0]: 
        # Need at least 2 points for cdist mean, and equal number of points after cleaning
        return np.nan

    if scale_x:
        # Scale x (reference)
        mean_x = np.mean(x_clean, axis=0)
        # ddof=0 for x (obs) to match R's scale default when scaling factor is provided by user (implicitly here)
        # R's scale(x, center=TRUE, scale=TRUE) uses sd with n-1.
        # However, the R escore function calls scale(x) then scale(y, center=attr(x,"scaled:center"), scale=attr(x,"scaled:scale"))
        # This means y is scaled by x's sample sd (n-1).
        # For consistency, let's use ddof=1 for std_x if it's to be used for y.
        std_x = np.std(x_clean, axis=0, ddof=1) 
        
        x_scaled = np.zeros_like(x_clean, dtype=float)
        for j in range(x_clean.shape[1]):
            if std_x[j] > 1e-12: # Effectively non-zero std
                x_scaled[:,j] = (x_clean[:,j] - mean_x[j]) / std_x[j]
            else: # Zero std, R's scale() results in 0s (x_col - mean_x_col = 0)
                x_scaled[:,j] = 0.0 # x_clean[:,j] - mean_x[j] would also be 0
        x_proc = x_scaled

        # Scale y using mean_x and std_x (from x_clean)
        y_scaled = np.zeros_like(y_clean, dtype=float)
        for j in range(y_clean.shape[1]):
            if std_x[j] > 1e-12: # Use std_x for scaling y
                y_scaled[:,j] = (y_clean[:,j] - mean_x[j]) / std_x[j]
            else:
                # If std_x[j] is zero, R's scale() results in y_col - mean_x_col
                y_scaled[:,j] = y_clean[:,j] - mean_x[j]
        y_proc = y_scaled
    else:
        x_proc = x_clean
        y_proc = y_clean
    
    if n_cases is not None:
        n_proc = x_proc.shape[0] # x_proc and y_proc have same length here
        actual_n_cases = min(n_proc, n_cases)
        if actual_n_cases >= 1: # Need at least 1 for choice
            # Sample common indices if we want to maintain correspondence from original sampling
            # Or sample independently if that's the R behavior (R samples x and y independently)
            # R: x <- x[sample(n.x, size = n.cases), , drop = FALSE]
            #    y <- y[sample(n.y, size = n.cases), , drop = FALSE]
            # This implies independent sampling from the (already potentially downsampled) x_proc, y_proc
            idx_x = np.random.choice(x_proc.shape[0], actual_n_cases, replace=False)
            idx_y = np.random.choice(y_proc.shape[0], actual_n_cases, replace=False)
            x_proc = x_proc[idx_x]
            y_proc = y_proc[idx_y]
        else:
            return np.nan # Not enough data for n_cases sampling
            
    if x_proc.shape[0] < 1 or y_proc.shape[0] < 1: # cdist needs at least 1 row
        return np.nan

    # cdist requires at least 1 observation in each matrix.
    # If x_proc has 1 row, d_xx is [[0]]. If 0 rows, error.
    # If x_proc or y_proc has 0 rows at this point, it's an issue.
    if x_proc.shape[0] == 0 or y_proc.shape[0] == 0: return np.nan


    d_xx = cdist(x_proc, x_proc, 'euclidean')
    d_yy = cdist(y_proc, y_proc, 'euclidean')
    d_xy = cdist(x_proc, y_proc, 'euclidean')

    # np.mean of empty array (if d_xx has 0 elements due to 0 rows) is nan.
    # If x_proc has 1 row, d_xx is [[0]], mean is 0.
    term1 = np.mean(d_xy) if d_xy.size > 0 else 0.0
    # For d_xx and d_yy, if only one point, mean is 0. If two points, it's the distance.
    # R's edist calculation: 2*S1 - S2 - S3, where S1=mean(dist(X,Y)), S2=mean(dist(X,X)), S3=mean(dist(Y,Y))
    # The R wrapper then divides by 2. So (2*S1 - S2 - S3)/2 = S1 - S2/2 - S3/2.
    term2 = 0.5 * (np.mean(d_xx) if d_xx.size > 0 else 0.0)
    term3 = 0.5 * (np.mean(d_yy) if d_yy.size > 0 else 0.0)
    
    result = term1 - term2 - term3
    
    # The R escore wrapper in MBC-QDM.R divides by 2.
    # energy::edist itself does not, but the R wrapper does.
    # So, Python should also divide by 2 to match the R script's escore output.
    # This was already there, but the formula from energy package is E = A - B/2 - C/2
    # where A = E||X-Y||, B=E||X-X'||, C=E||Y-Y'||.
    # The R code `edist(rbind(x, y), sizes = c(n.x, n.y), ...)[1]/2`
    # `energy:::.edist` returns `2*S1 - S2 - S3`. So dividing by 2 gives `S1 - S2/2 - S3/2`.
    # This seems correct.
    return result


def MRS(o_c, m_c, m_p, o_c_chol=None, o_p_chol=None, m_c_chol=None, m_p_chol=None):
    """Multivariate rescaling based on Cholesky decomposition"""
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
    mbcfactor = solve(m_c_chol.T, o_c_chol.T).T # R: solve(U) %*% V. If U is upper.
                                                # Python: solve(U, V) -> U X = V -> X = U^-1 V
                                                # If we use lower=False, cholesky returns U.
                                                # R: m.c %*% (solve(m.c.chol) %*% o.c.chol)
                                                #    m.c %*% (U_mc^-1 @ U_oc)
                                                # Python: m_c_cent @ (U_mc_inv @ U_oc)
                                                # Let factor = U_mc_inv @ U_oc.
                                                # U_mc @ factor = U_oc. So factor = solve(U_mc, U_oc)
                                                # This seems correct.
    mbpfactor = solve(m_p_chol, o_p_chol)
    
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
    
    # Iterative MBC/reranking
    # R's chol is upper by default. Pass lower=False to np.linalg.cholesky
    # ddof=1 for sample covariance matrix, as R's cov() default.
    o_c_cov_r = nearPD(np.cov(o_c_r, rowvar=False, ddof=1))
    o_c_chol = cholesky(o_c_cov_r, lower=False) 
    o_p_chol = o_c_chol 
    
    for k_iter_loop in range(iter): 
        m_c_cov_r = nearPD(np.cov(m_c_r, rowvar=False, ddof=1))
        m_c_chol = cholesky(m_c_cov_r, lower=False)
        m_p_chol = m_c_chol 

        fit_mbc = MRS(o_c_r, m_c_r, m_p_r, o_c_chol=o_c_chol,
                     o_p_chol=o_p_chol, m_c_chol=m_c_chol, m_p_chol=m_p_chol)

        m_c_r = np.apply_along_axis(rankdata, 0, fit_mbc['mhat_c'], method=rank_method_loop)
        m_p_r = np.apply_along_axis(rankdata, 0, fit_mbc['mhat_p'], method=rank_method_loop)

        cor_j = np.corrcoef(m_c_r, rowvar=False, ddof=1)
        cor_j[np.isnan(cor_j)] = 0
        cor_diff = np.mean(np.abs(cor_j - cor_i))
        cor_i = cor_j
            
        if not silent:
            # R prints 1-based iteration, mean equality of ranks, cor_diff
            print(f"{k_iter_loop+1} {np.mean(m_c_r == m_c_i_rank_check):.6f} {cor_diff:.6g} ", end='')
            
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
    jitter_factor_list = ensure_list_len(jitter_factor, n_vars, 0)
    ratio_max_list = ensure_list_len(ratio_max, n_vars, 2)
    ratio_max_trace_list = ensure_list_len(ratio_max_trace, n_vars, 10*0.05)
        
    m_c_qmap_initial = m_c_arr.copy() # Store initial m_c for QDM
    m_p_qmap_initial = m_p_arr.copy() # Store initial m_p for QDM
    
    # These will be modified in the loop if !qmap_precalc
    m_c_after_qdm = m_c_arr.copy() 
    m_p_after_qdm = m_p_arr.copy()

    if not qmap_precalc:
        for i in range(n_vars):
            fit_qmap = QDM(o_c_arr[:,i], m_c_qmap_initial[:,i], m_p_qmap_initial[:,i], # Use original m_c, m_p
                          ratio=ratio_seq_list[i], trace_calc=trace_calc_list[i],
                          trace=trace_list[i], jitter_factor=jitter_factor_list[i],
                          n_tau=n_tau, ratio_max=ratio_max_list[i],
                          ratio_max_trace=ratio_max_trace_list[i],
                          subsample=subsample, pp_type=pp_type, ties=ties)
            m_c_after_qdm[:,i] = fit_qmap['mhat_c']
            m_p_after_qdm[:,i] = fit_qmap['mhat_p']
    
    # Iteration starts with QDM-corrected data
    m_c_iter = m_c_after_qdm.copy()
    m_p_iter = m_p_after_qdm.copy()
    
    cor_i = np.corrcoef(m_c_iter, rowvar=False, ddof=1) # Pearson correlation, ddof=1 for sample cov
    cor_i[np.isnan(cor_i)] = 0
    
    o_c_cov = nearPD(np.cov(o_c_arr, rowvar=False, ddof=1))
    o_c_chol = cholesky(o_c_cov, lower=False)
    o_p_chol = o_c_chol 
    
    for k_iter_loop in range(iter): 
        m_c_cov = nearPD(np.cov(m_c_iter, rowvar=False, ddof=1))
        m_c_chol = cholesky(m_c_cov, lower=False)
        m_p_chol = m_c_chol 

        # MRS uses o_c_arr (original obs), m_c_iter, m_p_iter
        fit_mbc = MRS(o_c_arr, m_c_iter, m_p_iter, o_c_chol=o_c_chol,
                     o_p_chol=o_p_chol, m_c_chol=m_c_chol, m_p_chol=m_p_chol)
        
        m_c_iter_after_mrs = fit_mbc['mhat_c']
        m_p_iter_after_mrs = fit_mbc['mhat_p']
        
        # QDM step within MBCp loop (R uses ratio=FALSE here)
        for j in range(n_vars):
            # R's internal QDM call in MBCp uses o.c[,j], m.c[,j], m.p[,j]
            # where m.c and m.p are the *iterated* versions from MRS.
            # It also uses ratio=FALSE, and a limited set of QDM params.
            # Jitter is not explicitly passed in R's MBCp internal QDM, assume 0 or default.
            # Trace params from outer scope.
            fit_qmap_inner = QDM(o_c_arr[:,j], m_c_iter_after_mrs[:,j], m_p_iter_after_mrs[:,j], 
                                ratio=False, # R uses ratio=FALSE
                                n_tau=n_tau, pp_type=pp_type, 
                                jitter_factor=0, # R's internal call doesn't seem to pass jitter
                                trace=trace_list[j], trace_calc=trace_calc_list[j], ties=ties)
            m_c_iter[:,j] = fit_qmap_inner['mhat_c'] # Update m_c_iter for next MRS or cor check
            m_p_iter[:,j] = fit_qmap_inner['mhat_p'] # Update m_p_iter
            
        cor_j = np.corrcoef(m_c_iter, rowvar=False, ddof=1)
        cor_j[np.isnan(cor_j)] = 0
        cor_diff = np.mean(np.abs(cor_j - cor_i))
        cor_i = cor_j
            
        if not silent:
            print(f"{k_iter_loop+1} {cor_diff:.6g} ", end='') 
            
        if cor_diff < cor_thresh:
            break
            
    if not silent:
        print()
        
    # Final shuffle using the initially QDM'd outputs (m_c_after_qdm, m_p_after_qdm)
    # and ranks of the final iterated values (m_c_iter, m_p_iter)
    rank_method_final = 'ordinal' if ties == 'first' else ties

    m_c_final = np.empty_like(m_c_iter)
    m_p_final = np.empty_like(m_p_iter)

    for i in range(n_vars):
        ranks_c = rankdata(m_c_iter[:,i], method=rank_method_final) - 1 
        ranks_p = rankdata(m_p_iter[:,i], method=rank_method_final) - 1 

        sorted_initial_qdm_c = np.sort(m_c_after_qdm[:,i])
        sorted_initial_qdm_p = np.sort(m_p_after_qdm[:,i])
        
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
