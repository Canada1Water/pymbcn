import jax
import jax.numpy as jnp
from jax.scipy.linalg import cholesky, qr
from jax.scipy.stats import rankdata
from jax.scipy.spatial.distance import cdist
from jax.scipy.stats import norm
from jax.random import uniform, normal
from jax import jit, vmap
import numpy as np
from sklearn.neighbors import NearestNeighbors

# Enable 64-bit precision for JAX
jax.config.update("jax_enable_x64", True)

def QDM(o_c, m_c, m_p, ratio=False, trace=0.05, trace_calc=0.5*0.05,
        jitter_factor=0, n_tau=None, ratio_max=2, ratio_max_trace=10*0.05,
        ECBC=False, ties='first', subsample=None, pp_type=7):
    """Quantile Delta Mapping bias correction"""
    # Handle jitter
    if jitter_factor == 0 and (len(jnp.unique(o_c)) == 1 or
                               len(jnp.unique(m_c)) == 1 or
                               len(jnp.unique(m_p)) == 1):
        jitter_factor = jnp.sqrt(jnp.finfo(float).eps)
        
    if jitter_factor > 0:
        key = jax.random.PRNGKey(0)
        o_c = o_c + uniform(key, (len(o_c),) * 2 * jitter_factor - jitter_factor
        m_c = m_c + uniform(key, (len(m_c),) * 2 * jitter_factor - jitter_factor
        m_p = m_p + uniform(key, (len(m_p),) * 2 * jitter_factor - jitter_factor

    # Handle ratio data
    if ratio:
        epsilon = jnp.finfo(float).eps
        o_c = jnp.where(o_c < trace_calc, uniform(key, (len(o_c),) * (trace_calc - epsilon) + epsilon, o_c)
        m_c = jnp.where(m_c < trace_calc, uniform(key, (len(m_c),) * (trace_calc - epsilon) + epsilon, m_c)
        m_p = jnp.where(m_p < trace_calc, uniform(key, (len(m_p),) * (trace_calc - epsilon) + epsilon, m_p)

    # Calculate empirical quantiles
    n = len(m_p)
    if n_tau is None:
        n_tau = n
    tau = jnp.linspace(0, 1, n_tau)
    
    if subsample is not None:
        quant_o_c = jnp.mean(vmap(lambda _: jnp.quantile(jax.random.choice(key, o_c, (len(tau),)), tau, method=pp_type))(jnp.arange(subsample)), axis=0)
        quant_m_c = jnp.mean(vmap(lambda _: jnp.quantile(jax.random.choice(key, m_c, (len(tau),)), tau, method=pp_type))(jnp.arange(subsample)), axis=0)
        quant_m_p = jnp.mean(vmap(lambda _: jnp.quantile(jax.random.choice(key, m_p, (len(tau),)), tau, method=pp_type))(jnp.arange(subsample)), axis=0)
    else:
        quant_o_c = jnp.quantile(o_c, tau, method=pp_type)
        quant_m_c = jnp.quantile(m_c, tau, method=pp_type)
        quant_m_p = jnp.quantile(m_p, tau, method=pp_type)

    # Apply quantile delta mapping
    tau_m_p = jnp.interp(m_p, quant_m_p, tau)
    
    if ratio:
        approx_t_qmc_tmp = jnp.interp(tau_m_p, tau, quant_m_c)
        delta_m = m_p / approx_t_qmc_tmp
        delta_m = jnp.where((delta_m > ratio_max) & (approx_t_qmc_tmp < ratio_max_trace), ratio_max, delta_m)
        mhat_p = jnp.interp(tau_m_p, tau, quant_o_c) * delta_m
    else:
        delta_m = m_p - jnp.interp(tau_m_p, tau, quant_m_c)
        mhat_p = jnp.interp(tau_m_p, tau, quant_o_c) + delta_m

    mhat_c = jnp.interp(m_c, quant_m_c, quant_o_c)
    
    # Handle ratio data
    if ratio:
        mhat_c = jnp.where(mhat_c < trace, 0, mhat_c)
        mhat_p = jnp.where(mhat_p < trace, 0, mhat_p)
        
    if ECBC:
        if len(mhat_p) == len(o_c):
            mhat_p = jnp.sort(mhat_p)[jnp.argsort(jnp.argsort(o_c))]
        else:
            raise ValueError('Schaake shuffle failed due to incompatible lengths')
            
    return {'mhat_c': mhat_c, 'mhat_p': mhat_p}

@jit
def escore(x, y, scale_x=False, n_cases=None, alpha=1, method='cluster'):
    """Energy score for assessing equality of multivariate samples"""
    if scale_x:
        x = (x - jnp.mean(x, axis=0)) / jnp.std(x, axis=0)
        y = (y - jnp.mean(y, axis=0)) / jnp.std(y, axis=0)
        
    if n_cases is not None:
        n_cases = min(len(x), len(y), n_cases)
        x = x[jax.random.choice(jax.random.PRNGKey(0), len(x), (n_cases,), replace=False)]
        y = y[jax.random.choice(jax.random.PRNGKey(0), len(y), (n_cases,), replace=False)]
        
    # Calculate energy distance
    xx = cdist(x, x, 'euclidean')
    yy = cdist(y, y, 'euclidean')
    xy = cdist(x, y, 'euclidean')
    
    return (jnp.mean(xy) - 0.5 * jnp.mean(xx) - 0.5 * jnp.mean(yy)) / 2

@jit
def MRS(o_c, m_c, m_p, o_c_chol=None, o_p_chol=None, m_c_chol=None, m_p_chol=None):
    """Multivariate rescaling based on Cholesky decomposition"""
    # Center based on multivariate means
    o_c_mean = jnp.mean(o_c, axis=0)
    m_c_mean = jnp.mean(m_c, axis=0)
    m_p_mean = jnp.mean(m_p, axis=0)
    
    o_c = o_c - o_c_mean
    m_c = m_c - m_c_mean
    m_p = m_p - m_p_mean
    
    # Cholesky decomposition
    if o_c_chol is None:
        o_c_chol = cholesky(jnp.cov(o_c, rowvar=False))
    if o_p_chol is None:
        o_p_chol = cholesky(jnp.cov(o_c, rowvar=False))
    if m_c_chol is None:
        m_c_chol = cholesky(jnp.cov(m_c, rowvar=False))
    if m_p_chol is None:
        m_p_chol = cholesky(jnp.cov(m_c, rowvar=False))
        
    # Bias correction factors
    mbcfactor = jnp.linalg.solve(m_c_chol, o_c_chol)
    mbpfactor = jnp.linalg.solve(m_p_chol, o_p_chol)
    
    # Multivariate bias correction
    mbc_c = m_c @ mbcfactor
    mbc_p = m_p @ mbpfactor
    
    # Recenter and account for change in means
    mbc_c = mbc_c + o_c_mean
    mbc_p = mbc_p + o_c_mean + (m_p_mean - m_c_mean)
    
    return {'mhat_c': mbc_c, 'mhat_p': mbc_p}

@jit
def rot_random(k):
    """Generate random orthogonal rotation matrix"""
    rand = normal(jax.random.PRNGKey(0), (k, k))
    Q, R = qr(rand)
    diagR = jnp.diag(R)
    rot = Q @ jnp.diag(diagR / jnp.abs(diagR))
    return rot

def MBCr(o_c, m_c, m_p, iter=20, cor_thresh=1e-4, ratio_seq=None, trace=0.05,
         trace_calc=0.5*0.05, jitter_factor=0, n_tau=None, ratio_max=2,
         ratio_max_trace=10*0.05, ties='first', qmap_precalc=False,
         silent=False, subsample=None, pp_type=7):
    """Multivariate quantile mapping bias correction (Spearman correlation)"""
    n_vars = o_c.shape[1]
    if ratio_seq is None:
        ratio_seq = [False] * n_vars
    if jnp.isscalar(trace_calc):
        trace_calc = [trace_calc] * n_vars
    if jnp.isscalar(trace):
        trace = [trace] * n_vars
    if jnp.isscalar(jitter_factor):
        jitter_factor = [jitter_factor] * n_vars
    if jnp.isscalar(ratio_max):
        ratio_max = [ratio_max] * n_vars
    if jnp.isscalar(ratio_max_trace):
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
    
    # Calculate ranks
    o_c_r = jnp.apply_along_axis(rankdata, 0, o_c, method=ties)
    m_c_r = jnp.apply_along_axis(rankdata, 0, m_c, method=ties)
    m_p_r = jnp.apply_along_axis(rankdata, 0, m_p, method=ties)
    
    m_c_i = m_c_r.copy()
    if cor_thresh > 0:
        cor_i = jnp.corrcoef(m_c_r, rowvar=False)
        cor_i = jnp.where(jnp.isnan(cor_i), 0, cor_i)
    
    # Iterative MBC/reranking
    o_c_chol = o_p_chol = cholesky(jnp.cov(o_c_r, rowvar=False))
    
    for i in range(iter):
        m_c_chol = m_p_chol = cholesky(jnp.cov(m_c_r, rowvar=False))
        fit_mbc = MRS(o_c_r, m_c_r, m_p_r, o_c_chol=o_c_chol,
                     o_p_chol=o_p_chol, m_c_chol=m_c_chol, m_p_chol=m_p_chol)
        
        m_c_r = jnp.apply_along_axis(rankdata, 0, fit_mbc['mhat_c'], method=ties)
        m_p_r = jnp.apply_along_axis(rankdata, 0, fit_mbc['mhat_p'], method=ties)
        
        if cor_thresh > 0:
            cor_j = jnp.corrcoef(m_c_r, rowvar=False)
            cor_j = jnp.where(jnp.isnan(cor_j), 0, cor_j)
            cor_diff = jnp.mean(jnp.abs(cor_j - cor_i))
            cor_i = cor_j
        else:
            cor_diff = 0
            
        if not silent:
            print(f"{i} {jnp.mean(m_c_r == m_c_i)} {cor_diff} ", end='')
            
        if cor_diff < cor_thresh:
            break
        if jnp.array_equal(m_c_r, m_c_i):
            break
            
        m_c_i = m_c_r.copy()
        
    if not silent:
        print()
        
    # Replace ordinal ranks with QDM outputs
    for i in range(n_vars):
        m_c_r[:,i] = jnp.sort(m_c_qmap[:,i])[m_c_r[:,i].astype(int)-1]
        m_p_r[:,i] = jnp.sort(m_p_qmap[:,i])[m_p_r[:,i].astype(int)-1]
        
    return {'mhat_c': m_c_r, 'mhat_p': m_p_r}

def MBCp(o_c, m_c, m_p, iter=20, cor_thresh=1e-4, ratio_seq=None, trace=0.05,
         trace_calc=0.5*0.05, jitter_factor=0, n_tau=None, ratio_max=2,
         ratio_max_trace=10*0.05, ties='first', qmap_precalc=False,
         silent=False, subsample=None, pp_type=7):
    """Multivariate quantile mapping bias correction (Pearson correlation)"""
    n_vars = o_c.shape[1]
    if ratio_seq is None:
        ratio_seq = [False] * n_vars
    if jnp.isscalar(trace_calc):
        trace_calc = [trace_calc] * n_vars
    if jnp.isscalar(trace):
        trace = [trace] * n_vars
    if jnp.isscalar(jitter_factor):
        jitter_factor = [jitter_factor] * n_vars
    if jnp.isscalar(ratio_max):
        ratio_max = [ratio_max] * n_vars
    if jnp.isscalar(ratio_max_trace):
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
        cor_i = jnp.corrcoef(m_c, rowvar=False)
        cor_i = jnp.where(jnp.isnan(cor_i), 0, cor_i)
    
    o_c_chol = o_p_chol = cholesky(jnp.cov(o_c, rowvar=False))
    
    for i in range(iter):
        m_c_chol = m_p_chol = cholesky(jnp.cov(m_c, rowvar=False))
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
            cor_j = jnp.corrcoef(m_c, rowvar=False)
            cor_j = jnp.where(jnp.isnan(cor_j), 0, cor_j)
            cor_diff = jnp.mean(jnp.abs(cor_j - cor_i))
            cor_i = cor_j
        else:
            cor_diff = 0
            
        if not silent:
            print(f"{i} {cor_diff} ", end='')
            
        if cor_diff < cor_thresh:
            break
            
    if not silent:
        print()
        
    # Replace with shuffled QDM elements
    for i in range(n_vars):
        m_c[:,i] = jnp.sort(m_c_qmap[:,i])[rankdata(m_c[:,i], method=ties)-1]
        m_p[:,i] = jnp.sort(m_p_qmap[:,i])[rankdata(m_p[:,i], method=ties)-1]
        
    return {'mhat_c': m_c, 'mhat_p': m_p}

def MBCn(o_c, m_c, m_p, iter=30, ratio_seq=None, trace=0.05,
         trace_calc=0.5*0.05, jitter_factor=0, n_tau=None, ratio_max=2,
         ratio_max_trace=10*0.05, ties='first', qmap_precalc=False,
         rot_seq=None, silent=False, n_escore=0, return_all=False,
         subsample=None, pp_type=7):
    """Multivariate quantile mapping bias correction (N-dimensional pdf transfer)"""
    n_vars = o_c.shape[1]
    if ratio_seq is None:
        ratio_seq = [False] * n_vars
    if jnp.isscalar(trace_calc):
        trace_calc = [trace_calc] * n_vars
    if jnp.isscalar(trace):
        trace = [trace] * n_vars
    if jnp.isscalar(jitter_factor):
        jitter_factor = [jitter_factor] * n_vars
    if jnp.isscalar(ratio_max):
        ratio_max = [ratio_max] * n_vars
    if jnp.isscalar(ratio_max_trace):
        ratio_max_trace = [ratio_max_trace] * n_vars
        
    if rot_seq is not None and len(rot_seq) != iter:
        raise ValueError('length(rot_seq) != iter')
        
    # Energy score tracking
    escore_iter = jnp.full(iter+2, jnp.nan)
    
    if n_escore > 0:
        n_escore = min(o_c.shape[0], m_c.shape[0], n_escore)
        escore_cases_o_c = jax.random.choice(jax.random.PRNGKey(0), o_c.shape[0], (n_escore,), replace=False)
        escore_cases_m_c = jax.random.choice(jax.random.PRNGKey(0), m_c.shape[0], (n_escore,), replace=False)
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
    o_c_mean = jnp.mean(o_c, axis=0)
    o_c_sdev = jnp.std(o_c, axis=0)
    o_c_sdev = jnp.where(o_c_sdev < jnp.finfo(float).eps, 1, o_c_sdev)
    o_c = (o_c - o_c_mean) / o_c_sdev
    
    m_c_p = jnp.vstack((m_c, m_p))
    m_c_p_mean = jnp.mean(m_c_p, axis=0)
    m_c_p_sdev = jnp.std(m_c_p, axis=0)
    m_c_p_sdev = jnp.where(m_c_p_sdev < jnp.finfo(float).eps, 1, m_c_p_sdev)
    m_c_p = (m_c_p - m_c_p_mean) / m_c_p_sdev
    
    Xt = jnp.vstack((o_c, m_c_p))
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
        Xt = jnp.vstack((o_c, m_c, m_p))
        
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
        m_c[:,i] = jnp.sort(m_c_qmap[:,i])[rankdata(m_c[:,i], method=ties)-1]
        m_p[:,i] = jnp.sort(m_p_qmap[:,i])[rankdata(m_p[:,i], method=ties)-1]
        
    escore_iter = dict(zip(['RAW', 'QM'] + list(range(iter)), escore_iter))
    
    return {'mhat_c': m_c, 'mhat_p': m_p, 'escore_iter': escore_iter,
            'm_iter': m_iter}

def R2D2(o_c, m_c, m_p, ref_column=0, ratio_seq=None, trace=0.05,
         trace_calc=0.5*0.05, jitter_factor=0, n_tau=None, ratio_max=2,
         ratio_max_trace=10*0.05, ties='first', qmap_precalc=False,
         subsample=None, pp_type=7):
    """Multivariate bias correction using nearest neighbor rank resampling"""
    if o_c.shape[0] != m_c.shape[0] or o_c.shape[0] != m_p.shape[0]:
        raise ValueError("R2D2 requires data samples of equal length")
        
    n_vars = o_c.shape[1]
    if ratio_seq is None:
        ratio_seq = [False] * n_vars
    if jnp.isscalar(trace_calc):
        trace_calc = [trace_calc] * n_vars
    if jnp.isscalar(trace):
        trace = [trace] * n_vars
    if jnp.isscalar(jitter_factor):
        jitter_factor = [jitter_factor] * n_vars
    if jnp.isscalar(ratio_max):
        ratio_max = [ratio_max] * n_vars
    if jnp.isscalar(ratio_max_trace):
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
    
    # Calculate ranks
    o_c_r = jnp.apply_along_axis(rankdata, 0, o_c, method=ties)
    m_c_r = jnp.apply_along_axis(rankdata, 0, m_c_qmap, method=ties)
    m_p_r = jnp.apply_along_axis(rankdata, 0, m_p_qmap, method=ties)
    
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
        r2d2_c[:,i] = jnp.sort(r2d2_c[:,i])[new_c_r[:,i].astype(int)-1]
        r2d2_p[:,i] = jnp.sort(r2d2_p[:,i])[new_p_r[:,i].astype(int)-1]
        
    return {'mhat_c': r2d2_c, 'mhat_p': r2d2_p}

