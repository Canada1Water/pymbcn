# testcode.py Documentation

## Overview

The `testcode.py` script is a comprehensive testing and validation suite for the pymbcn library's multivariate bias correction methods. It parallels the functionality of `testcode.R` from the R MBC package, providing side-by-side validation of the Python implementation against established R methods.

## Purpose

This script serves multiple critical functions:

1. **Validation Testing**: Ensures the Python implementation produces results equivalent to the R reference implementation
2. **Method Comparison**: Demonstrates differences between univariate (QDM) and multivariate (MBCp, MBCr, MBCn) bias correction methods
3. **Visualization**: Creates comprehensive plots and diagnostics for evaluating bias correction performance
4. **Quality Assurance**: Generates standardized outputs that can be compared across different implementations

## Input Data

The script reads climate data from a NetCDF file (`cccma_output.nc`) containing:

### Climate Variables (8 variables):
- **pr**: Precipitation (ratio variable with trace handling)
- **tas**: Near-surface air temperature 
- **dtr**: Diurnal temperature range
- **sfcWind**: Near-surface wind speed
- **ps**: Surface pressure
- **huss**: Near-surface specific humidity
- **rsds**: Surface downwelling shortwave radiation
- **rlds**: Surface downwelling longwave radiation

### Data Periods:
- **Control Period**: Historical/calibration data (`gcm_c`, `rcm_c`) used for training bias correction
- **Projection Period**: Future/validation data (`gcm_p`, `rcm_p`) used for applying and evaluating corrections

### Data Sources:
- **GCM Data**: Global Climate Model outputs (CanESM2) - the biased model data to be corrected
- **RCM Data**: Regional Climate Model outputs (CanRCM4) - the observational reference/target

## Processing Workflow

### 1. Data Loading and Preprocessing
```python
# Loads 8-variable climate dataset from NetCDF
# Handles missing files gracefully with synthetic data
# Extracts metadata (ratio_seq, trace values) for variable-specific processing
```

### 2. Univariate Bias Correction (QDM)
```python
# Applies Quantile Delta Mapping to each variable independently
# Processes both control and projection periods
# Includes adaptive trace adjustment for precipitation variables
# Generates before/after comparison plots
```

**Key Features:**
- Variable-specific ratio handling (precipitation uses multiplicative correction)
- Trace value adjustment for low-correlation cases
- Individual variable histograms and time series plots

### 3. Multivariate Bias Corrections

#### MBCp (Pearson Correlation Preservation)
- Preserves linear (Pearson) correlations between variables
- Uses iterative approach with quantile mapping and correlation adjustment
- Best for variables with linear relationships

#### MBCr (Spearman Correlation Preservation) 
- Preserves rank-based (Spearman) correlations
- Works with rank transformations before applying corrections
- Robust to non-linear relationships

#### MBCn (N-dimensional PDF Transfer)
- Most sophisticated method using rotation matrices
- Preserves full multivariate distribution structure
- Uses energy score optimization across multiple iterations

### 4. Comprehensive Visualization

#### Individual Variable Analysis:
- **Histograms**: Distribution comparison (GCM vs QDM-corrected)
- **Time Series**: Temporal pattern analysis
- Generated for all 8 climate variables

#### Correlation Analysis:
- **R-style Correlation Plots**: 4-panel comparison (Pearson/Spearman × Calibration/Evaluation)
- **Pairwise Scatter Plots**: Multi-variable relationship visualization
- Compares observational reference vs. each correction method

#### Calibration Pair Plots:
- Multi-variable scatter plot matrices
- Color-coded by method (CanESM2, CanRCM4, QDM, MBCp, MBCr, MBCn)
- Shows how well each method preserves multivariate relationships

### 5. Performance Evaluation

#### Energy Score Calculations:
```python
# Multivariate distance metric between corrected and reference data
# Lower scores indicate better performance
# Energy Skill Score (ESS) shows improvement over QDM baseline
```

#### Quantitative Metrics:
- Raw energy scores for each method
- Energy Skill Scores (ESS) showing relative improvement
- Iteration tracking for MBCn convergence

### 6. Output Generation

#### NetCDF File Creation:
```python
# Saves all corrected datasets to 'python_corrected_output.nc'
# Includes control and projection periods for all methods
# Structured for direct comparison with R outputs
```

## Generated Output Files

### Individual Variable Plots (16 files):
- `{variable}_gcm_p_vs_qdm_p_histograms.png` - Distribution comparisons
- `{variable}_gcm_p_vs_qdm_p_timeseries.png` - Temporal pattern analysis

### Method Comparison Plots (18 files):
- `{method}_r_style_correlations.png` - 4-panel correlation analysis (3 files)
- `{dataset}_calibration_pairs.png` - Pairwise scatter matrices (6 files)

### Data Output:
- `python_corrected_output.nc` - NetCDF file with all corrected datasets

## Usage in Validation Workflow

### Step 1: Run testcode.py
```bash
cd tests/
python testcode.py
```

### Step 2: Run R equivalent (if available)
```bash
Rscript testcode.R  # Generates r_corrected_output.nc
```

### Step 3: Compare Python vs R outputs
```bash
python compare_outputs.py
```
This generates:
- `comparison_py_vs_r_control.png` - Control period comparison
- `comparison_py_vs_r_projection.png` - Projection period comparison

## Key Features for Quality Assurance

### Reproducibility:
- Fixed random seed (`np.random.seed(1)`) ensures consistent results
- Identical parameters across Python and R implementations

### Error Handling:
- Graceful fallback to synthetic data if NetCDF file missing
- Robust correlation calculations with NaN handling
- Timeout protection for computationally expensive operations

### Adaptive Processing:
- Variable-specific parameter handling (ratio vs. absolute variables)
- Automatic trace adjustment for poor correlation cases
- Flexible plotting with memory and performance optimizations

## Performance Considerations

### Computational Optimization:
- Reduced variable count for pairwise plots (max 5 variables)
- Disabled smoothing in scatter plots for speed
- Progress indicators for long-running calculations
- Timeout protection for energy score calculations

### Memory Management:
- Efficient data structures for large climate datasets
- Proper cleanup of matplotlib figures
- Optimized plotting functions with reduced resolution options

## Scientific Validation

This test suite validates that the Python implementation:

1. **Preserves Statistical Properties**: Maintains appropriate correlations and distributions
2. **Matches R Implementation**: Produces numerically equivalent results to established methods
3. **Handles Edge Cases**: Correctly processes precipitation, temperature, and other climate variables
4. **Scales Appropriately**: Works with realistic climate dataset sizes and structures

The comprehensive output allows researchers to verify that bias correction methods are working correctly and producing scientifically valid results for climate model post-processing applications.

## Relationship to compare_outputs.py

The `compare_outputs.py` script provides the final validation step by creating scatter plots comparing Python vs R outputs for each correction method and variable. It:

- Reads both `python_corrected_output.nc` and `r_corrected_output.nc`
- Creates grid plots (4 methods × 8 variables) for both control and projection periods
- Calculates R² values to quantify agreement between implementations
- Generates publication-quality comparison figures

This two-step process (testcode.py → compare_outputs.py) ensures the Python implementation is both scientifically valid and numerically equivalent to the established R reference implementation.
