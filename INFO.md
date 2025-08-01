# pymbcn - Python Multivariate Bias Correction Methods

This document explains the `pymbcn` library, which implements statistical methods for correcting biases in climate model outputs.

## Overview
The library provides statistical methods to adjust climate model predictions to better match observational data while preserving important relationships between variables (like temperature and precipitation).

## Key Components

### 1. Helper Functions

- **`_ensure_symmetric(A)`**: Makes a matrix symmetric by averaging A and A.T
- **`nearPD(A)`**: Converts any matrix to the nearest positive definite matrix using eigenvalue adjustment and optional diagonal jittering (needed for statistical calculations)

### 2. Core Bias Correction Methods

#### **QDM (Quantile Delta Mapping)**
```python
QDM(o_c, m_c, m_p, ...)
```
- **Purpose**: Univariate bias correction (corrects one variable at a time)
- **Inputs**: 
  - `o_c`: Observed climate data (calibration period)
  - `m_c`: Model data (calibration period) 
  - `m_p`: Model data (projection period)
- **How it works**: Maps quantiles from model distributions to observed distributions, preserving the relative changes between periods

#### **MBCp (Multivariate Bias Correction - Pearson)**
```python
MBCp(o_c, m_c, m_p, ...)
```
- **Purpose**: Corrects multiple variables while preserving Pearson correlations
- **Process**: 
  1. Initial QDM correction for each variable
  2. Iterative adjustment using matrix rescaling (MRS)
  3. Inner QDM loops to maintain marginal distributions

#### **MBCr (Multivariate Bias Correction - Rank/Spearman)**
```python
MBCr(o_c, m_c, m_p, ...)
```
- **Purpose**: Corrects multiple variables while preserving rank correlations (Spearman)
- **Process**: Works with ranks instead of raw values, better for non-linear relationships

#### **MBCn (N-dimensional PDF Transfer)**
```python
MBCn(o_c, m_c, m_p, ...)
```
- **Purpose**: Most sophisticated method - preserves the full multivariate probability distribution
- **Process**:
  1. Initial QDM correction
  2. Standardization of data
  3. Iterative random rotations and bias corrections
  4. Final rescaling and shuffling

#### **R2D2 (Rank Resampling for Dependence and Distribution)**
```python
R2D2(o_c, m_c, m_p, ...)
```
- **Purpose**: Uses nearest neighbor resampling based on reference variable ranks
- **Process**: Finds nearest neighbors in rank space and resamples accordingly

### 3. Support Functions

#### **MRS (Multivariate Rescaling)**
- Uses Cholesky decomposition to adjust covariance structures
- Rescales data to match target correlation patterns

#### **Energy Score Calculation**
```python
escore(x, y, ...)
```
- Evaluates how well the corrected data matches observations
- Based on energy statistics (measures distributional differences)

#### **Rotation Matrix Generation**
- `rot_random(k)`: Creates random orthogonal matrices for MBCn
- `generate_rotation_sequence()`, `save_rotation_sequence()`, `load_rotation_sequence()`: Manage rotation sequences

## Typical Workflow

1. **Input**: You have three datasets:
   - Historical observations (`o_c`)
   - Historical model output (`m_c`) 
   - Future model projections (`m_p`)

2. **Choose Method**: 
   - QDM for single variables
   - MBCp/MBCr/MBCn for multiple correlated variables
   - R2D2 for spatial/temporal dependence preservation

3. **Output**: Bias-corrected model data that:
   - Matches observed statistical properties
   - Preserves model-projected changes
   - Maintains variable relationships

## Example Use Cases

- Correcting temperature and precipitation together (preserving their correlation)
- Adjusting multiple climate variables at once
- Downscaling climate model output to match local observations
- Preparing climate data for impact studies

## Technical Details

### Data Requirements
- All methods expect numpy arrays as input
- Variables should be in columns, time/space samples in rows
- Missing values should be handled before applying corrections

### Key Parameters
- `iter`: Number of iterations for iterative methods
- `ratio`: Whether to treat variables as ratios (e.g., precipitation)
- `trace`: Threshold for zero values in ratio variables
- `ties`: Method for handling tied ranks
- `n_tau`: Number of quantiles for mapping

### Dependencies
- numpy >= 1.20.0
- scipy >= 1.7.0
- scikit-learn >= 1.0.0
- pandas >= 1.3.0 (optional)

## References
Based on methods described in:
- Cannon, A.J., 2018. Multivariate quantile mapping bias correction...
- Vrac, M., 2018. Multivariate bias adjustment of high-dimensional...

The methods are particularly valuable because raw climate model output often has systematic biases that need correction before use in applications like agriculture, hydrology,