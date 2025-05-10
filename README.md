# pymbcn - Multivariate Bias Correction Methods for Climate Model Outputs

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Python implementation of multivariate bias correction techniques for climate model outputs, including:

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
mbcn_result = MBCn(obs, model_hist, model_fut)
```

## Documentation

Full documentation is available at [GitHub Wiki](https://github.com/yourusername/pymbcn/wiki).

## References

Based on methods described in:
- Cannon, A.J., 2018. Multivariate quantile mapping bias correction...
- Vrac, M., 2018. Multivariate bias adjustment of high-dimensional...

## License

MIT License - See [LICENSE](LICENSE) file for details.
