# Install if needed
# install.packages("ncdf4")
library(ncdf4)

# 1. Load the RData
load("cccma.RData") # Loads the 'cccma' list

# --- USER INPUT REQUIRED: Define Metadata ---

# A. Time Dimension Info
time_c_units <- "days since 1979-01-01 00:00:00"
time_p_units <- "days since 2039-01-01 00:00:00"
calendar <- "gregorian" # or "gregorian", "noleap", "360_day", etc.

# B. Variable Metadata (from MBC.pdf)
# Lookup standard names: http://cfconventions.org/standard-names.html
var_metadata <- list(
  pr = list(units = "mm day-1", longname = "precipitation", stdname = "precipitation_flux", missval = 1.e37),
  tas = list(units = "deg C", longname = "average surface emperature", stdname = "air_temperature", missval = 1.e37),
  dtr = list(units = "deg C", longname = "diurnal temperature range", stdname = "air_temperature_range", missval = 1.e37),
  sfcWind = list(units = "m s-1", longname = "surface wind speed", stdname = "wind_speed", missval = 1.e37),
  ps = list(units = "hPa", longname = "Surface Air Pressure", stdname = "surface_air_pressure", missval = 1.e37),
  huss = list(units = "kg kg-1", longname = "surface specific humidity", stdname = "specific_humidity", missval = 1.e37),
  rlds = list(units = "W m-2", longname = "surface downwelling longwave radiation", stdname = "surface_downwelling_longwave_flux_in_air", missval = 1.e37),
  rsds = list(units = "W m-2", longname = "surface downwelling shortwave radiation", stdname = "surface_downwelling_shortwave_flux_in_air", missval = 1.e37)
  # Add any other variables if necessary
)
# Define a single missing value to use in the NetCDF file FOR ALL variables defined above
# Ensure any actual missing values in your R matrices are converted to this *before* writing.
# Example: cccma$gcm.c[is.na(cccma$gcm.c)] <- -9999.0 (do for all 4 matrices if needed)
common_missval <- 1.e37

# C. Location / Region (CRITICAL - ADD DESCRIPTION)
location_description <- "Data for single point/region"

# D. Trace/Ratio Metadata (Optional, but recommended if saving)
trace_longname <- "numeric values indicating trace thresholds for each ratio quantity."
ratio_seq_longname <- "numeric values indicating trace thresholds for each ratio quantity."

# --- End USER INPUT ---

# 2. Extract data components and variable names
gcm_c_data <- cccma$gcm.c
gcm_p_data <- cccma$gcm.p
rcm_c_data <- cccma$rcm.c
rcm_p_data <- cccma$rcm.p
var_names <- colnames(gcm_c_data) # Get variable names ("pr", "tas", ...)
n_vars <- length(var_names)
ratio_seq_data <- cccma$ratio.seq
trace_data <- cccma$trace

# Handle potential Inf in trace - replace with the common missing value
trace_data[is.infinite(trace_data)] <- common_missval

# 3. Define NetCDF Dimensions
len_time_c <- nrow(gcm_c_data)
len_time_p <- nrow(gcm_p_data)

# Create time coordinate values (assuming sequence starts at 0)
time_c_vals <- seq(from = 0, by = 1, length.out = len_time_c)
time_p_vals <- seq(from = 0, by = 1, length.out = len_time_p)

dim_time_c <- ncdim_def(
  name = "time_c",
  units = time_c_units,
  vals = time_c_vals,
  unlim = TRUE, # Make time unlimited
  calendar = calendar
)

dim_time_p <- ncdim_def(
  name = "time_p",
  units = time_p_units,
  vals = time_p_vals,
  unlim = TRUE, # Make time unlimited
  calendar = calendar
)

# Optional dimension for the metadata vectors (ratio.seq, trace)
dim_var_index <- ncdim_def(
    name = "variable_index",
    units = "",
    vals = 1:n_vars # Simple index 1 to 8
)


# 4. Define NetCDF Variables (Loop through the 8 climate variables)
nc_vars_list <- list()

for (i in 1:n_vars) {
  var_short_name <- var_names[i]
  meta <- var_metadata[[var_short_name]] # Get metadata for this variable

  if (is.null(meta)) {
      warning(paste("Metadata not defined for variable:", var_short_name, "- Skipping definition for NetCDF."))
      next # Skip if no metadata provided in the list above
  }

  # Define the 4 variables for this climate type (gcm_c, gcm_p, rcm_c, rcm_p)
  nc_vars_list[[paste0("gcm_c_", var_short_name)]] <- ncvar_def(
    name = paste0("gcm_c_", var_short_name),
    units = meta$units,
    dim = list(dim_time_c),
    missval = meta$missval, # Use the specific missval from metadata list
    longname = paste("CanESM2 Control:", meta$longname),
    prec = "float" # Or "double" if needed
  )
  nc_vars_list[[paste0("gcm_p_", var_short_name)]] <- ncvar_def(
    name = paste0("gcm_p_", var_short_name),
    units = meta$units,
    dim = list(dim_time_p),
    missval = meta$missval,
    longname = paste("CanESM2 Projection:", meta$longname),
    prec = "float"
  )
  nc_vars_list[[paste0("rcm_c_", var_short_name)]] <- ncvar_def(
    name = paste0("rcm_c_", var_short_name),
    units = meta$units,
    dim = list(dim_time_c),
    missval = meta$missval,
    longname = paste("CanRCM4 Control:", meta$longname),
    prec = "float"
  )
  nc_vars_list[[paste0("rcm_p_", var_short_name)]] <- ncvar_def(
    name = paste0("rcm_p_", var_short_name),
    units = meta$units,
    dim = list(dim_time_p),
    missval = meta$missval,
    longname = paste("CanRCM4 Projection:", meta$longname),
    prec = "float"
  )
}

# Define variables for trace and ratio_seq (optional)
nc_vars_list[["trace"]] <- ncvar_def(
    name = "trace",
    units = "",
    dim = list(dim_var_index),
    missval = common_missval, # Use common missing value
    longname = trace_longname, # Define this above!
    prec = "float"
)
nc_vars_list[["ratio_seq"]] <- ncvar_def(
    name = "ratio_seq",
    units = "",
    dim = list(dim_var_index),
    missval = -1, # Use an integer missval for logical/integer data
    longname = ratio_seq_longname, # Define this above!
    prec = "integer" # Store logical as integer (0 or 1)
)


# 5. Create the NetCDF File
nc_filename <- "cccma_output.nc"
ncout <- nc_create(
  filename = nc_filename,
  vars = nc_vars_list,
  force_v4 = TRUE # Use netCDF4 format
)

# 6. Write Data to Variables
for (i in 1:n_vars) {
  var_short_name <- var_names[i]
  meta <- var_metadata[[var_short_name]]

  if (is.null(meta)) {
      warning(paste("Metadata not defined for variable:", var_short_name, "- Skipping writing data."))
      next
  }

  # Make sure data uses the specified missing value before writing
  gcm_c_col <- gcm_c_data[, i]
  gcm_p_col <- gcm_p_data[, i]
  rcm_c_col <- rcm_c_data[, i]
  rcm_p_col <- rcm_p_data[, i]

  # Example: Replace NA with the defined missval (do only if NAs exist)
  # gcm_c_col[is.na(gcm_c_col)] <- meta$missval
  # gcm_p_col[is.na(gcm_p_col)] <- meta$missval
  # ... etc

  print(paste("Writing variable:", var_short_name))
  ncvar_put(ncout, paste0("gcm_c_", var_short_name), gcm_c_col)
  ncvar_put(ncout, paste0("gcm_p_", var_short_name), gcm_p_col)
  ncvar_put(ncout, paste0("rcm_c_", var_short_name), rcm_c_col)
  ncvar_put(ncout, paste0("rcm_p_", var_short_name), rcm_p_col)
}

# Write trace and ratio_seq data (optional)
print("Writing trace and ratio_seq")
ncvar_put(ncout, "trace", trace_data)
ncvar_put(ncout, "ratio_seq", as.integer(ratio_seq_data)) # Convert logical to 0/1


# 7. Add Variable Attributes (like standard_name if defined)
for (i in 1:n_vars) {
  var_short_name <- var_names[i]
  meta <- var_metadata[[var_short_name]]
  if (!is.null(meta) && !is.null(meta$stdname) && nzchar(meta$stdname)) {
      ncatt_put(ncout, paste0("gcm_c_", var_short_name), "standard_name", meta$stdname)
      ncatt_put(ncout, paste0("gcm_p_", var_short_name), "standard_name", meta$stdname)
      ncatt_put(ncout, paste0("rcm_c_", var_short_name), "standard_name", meta$stdname)
      ncatt_put(ncout, paste0("rcm_p_", var_short_name), "standard_name", meta$stdname)
  }
}

# Add attribute explaining variable_index mapping (optional, but good practice)
ncatt_put(ncout, "trace", "variable_correspondence", paste("Index corresponds to variable names:", paste(var_names, collapse=", ")))
ncatt_put(ncout, "ratio_seq", "variable_correspondence", paste("Index corresponds to variable names:", paste(var_names, collapse=", ")))
# Alternative for ratio_seq: add flag_values and flag_meanings attributes
# ncatt_put(ncout, "ratio_seq", "flag_values", c(0, 1))
# ncatt_put(ncout, "ratio_seq", "flag_meanings", "false true")


# 8. Add Global Attributes (CRITICAL)
ncatt_put(ncout, 0, "title", "Climate Model Output Time Series from cccma.RData")
ncatt_put(ncout, 0, "institution", "ECCC")
ncatt_put(ncout, 0, "source", "cccma.RData; GCM: CanESM2; RCM: CanRCM4")
ncatt_put(ncout, 0, "history", paste("Created on", date(), "by G.V. using R ncdf4.", Sys.info()["user"])) # CHANGE THIS
ncatt_put(ncout, 0, "Conventions", "CF-1.8") # If you followed CF conventions
ncatt_put(ncout, 0, "references", "Cannon, A.J., 2018. Multivariate quantile mapping bias correction: An N-dimensional probability density function transform for climate model simulations of multiple variables. Climate Dynamics,
50(1-2):31-49. doi:10.1007/s00382-017-3580-6")
ncatt_put(ncout, 0, "comment", paste("Data contains GCM and RCM simulations for control (.c) and projection (.p) periods.", location_description))


# 9. Close the NetCDF File (writes to disk)
nc_close(ncout)

print(paste("NetCDF file created:", nc_filename))
