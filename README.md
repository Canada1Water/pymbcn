# pymbcn - Multivariate Bias Correction Methods for Climate Model Outputs

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Python implementation of multivariate bias correction (npdf) technique for climate model outputs. This is a machine translation of the R package by Alex J. Cannon: https://cran.r-project.org/web/packages/MBC/index.html to python.

- Quantile Delta Mapping (QDM)
- Multivariate Bias Correction (MBC) methods:
  - MBCp (Pearson correlation)
  - MBCr (Spearman correlation)
  - MBCn (N-dimensional PDF transfer)
- R2D2 (Rank resampling for dependence and distribution)
- Energy score calculations

## Installation

To install in development mode:

```bash
git clone https://github.com/yourusername/pymbcn.git
cd pymbcn
pip install -e .
```

For production use:

```bash
pip install git+https://github.com/yourusername/pymbcn.git
```

## Quick Start

```python
from pymbcn import QDM, MBCn
import numpy as np

# Generate sample data
obs = np.random.normal(size=(100, 3))
model_hist = np.random.normal(size=(100, 3))
model_fut = np.random.normal(size=(100, 3))

# Apply QDM
qdm_result = QDM(obs, model_hist, model_fut)

# Apply MBCn
# Generate rotation matrices for MBCn (same across all grid points)
n_iter_mbcn = 30  # Match MBCn's default iter parameter
rot_matrices_mbcn = [rot_random(n_vars) for _ in range(n_iter_mbcn)]
fit_mbcn = MBCn(o_c=obs, m_c=model_hist, m_p=model_fut,
               ratio_seq=py_ratio_seq, trace=py_trace_val,
               jitter_factor=0, 
               ties='first',    
               silent=False, n_escore=100,
               pp_type='linear',
               rot_seq=rot_matrices_mbcn,
               iter=n_iter_mbcn)
mbcn_c = fit_mbcn['mhat_c']
mbcn_p = fit_mbcn['mhat_p']

```

## Documentation

See MBC_R/MBC.pdf

## TODO

Implement xarray apply_ufunc example

Improve differences between R package and python package (e.g. Energy distance scores)
To see results run tests/compare_outputs.py

Improve performance with pytorch or jax

## References

Cannon, A.J., 2018. Multivariate quantile mapping bias correction: An N-dimensional probability density function transform for climate model simulations of multiple variables. Climate Dynamics, 50(1-2):31-49. doi:10.1007/s00382-017-3580-6

Cannon, A.J., 2016. Multivariate bias correction of climate model output: Matching marginal distributions and inter-variable dependence structure. Journal of Climate, 29:7045-7064. doi:10.1175/JCLID-15-0679.1

Cannon, A.J., S.R. Sobie, and T.Q. Murdock, 2015. Bias correction of simulated precipitation by quantile mapping: How well do methods preserve relative changes in quantiles and extremes? Journal of Climate, 28:6938-6959. doi:10.1175/JCLI-D-14-00754.1

Francois, B., M. Vrac, A.J. Cannon, Y. Robin, and D. Allard, 2020. Multivariate bias corrections of climate simulations: Which benefits for which losses? Earth System Dynamics, 11:537-562. doi:10.5194/esd-11-537-2020

Vrac, M., 2018. Multivariate bias adjustment of high-dimensional climate simulations: the Rank Resampling for Distributions and Dependences (R2D2) bias correction. Hydrology and Earth System Sciences, 22:3175-3196. doi:10.5194/hess-22-3175-2018

## License

MIT License - See [LICENSE](LICENSE) file for details.
