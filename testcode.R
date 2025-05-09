## Not run:
library("MBC")
# Source the local, modified version of MBC-QDM.R to use functions with diagnostics
source("MBC-QDM.R") 

# Load necessary data
#data("cccma.RData")
load("cccma.RData") 
set.seed(1)

# Univariate quantile mapping
qdm.c <- cccma$gcm.c * 0
qdm.p <- cccma$gcm.p * 0

for (i in seq(ncol(cccma$gcm.c))) {
  current_debug_name <- NULL
  if (cccma$ratio.seq[i] && i == which(cccma$ratio.seq)[1]) { # Assuming pr is the first ratio var
      current_debug_name <- "pr_initial_qdm_mp_debug"
  }
  if (colnames(cccma$gcm.c)[i] == "huss") { 
      current_debug_name <- "huss_qdm_debug"
  }
  current_trace_r <- cccma$trace[i]
  current_trace_calc_r <- 0.5 * current_trace_r # Default relationship

  fit.qdm <- QDM(o.c = cccma$rcm.c[, i], m.c = cccma$gcm.c[, i], m.p = cccma$gcm.p[, i], 
                 ratio = cccma$ratio.seq[i], 
                 trace = current_trace_r, 
                 trace.calc = current_trace_calc_r, # Pass trace_calc explicitly
                 debug_name = current_debug_name) # current_debug_name from earlier logic
  
  temp_qdm_c <- fit.qdm$mhat.c
  temp_qdm_p <- fit.qdm$mhat.p

  # Adaptive thresholding for ratio variables if correlation is low
  if (cccma$ratio.seq[i]) {
    original_gcm_p_series_r <- cccma$gcm.p[, i]
    corrected_qdm_p_series_r <- temp_qdm_p
    
    # Check for non-zero standard deviation
    if (sd(original_gcm_p_series_r, na.rm = TRUE) > 1e-9 && sd(corrected_qdm_p_series_r, na.rm = TRUE) > 1e-9) {
      correlation_r <- NA
      tryCatch({
        correlation_r <- cor(original_gcm_p_series_r, corrected_qdm_p_series_r, use = "pairwise.complete.obs")
      }, error = function(e) {
        cat(paste0("R UNIQDM LOOP - Var: ", colnames(cccma$gcm.c)[i], ", Error calculating correlation: ", e$message, "\n"))
      })
      
      if (!is.na(correlation_r) && correlation_r < 0.8) {
        cat(paste0("R UNIQDM LOOP - Var: ", colnames(cccma$gcm.c)[i], ", Low correlation (", sprintf("%.2f", correlation_r), ") with original trace ", sprintf("%.4f", current_trace_r), ". Adjusting trace.\n"))
        adjusted_trace_r <- current_trace_r * 2.0
        adjusted_trace_calc_r <- 0.5 * adjusted_trace_r
        cat(paste0("R UNIQDM LOOP - Var: ", colnames(cccma$gcm.c)[i], ", New trace: ", sprintf("%.4f", adjusted_trace_r), ", New trace_calc: ", sprintf("%.4f", adjusted_trace_calc_r), "\n"))
        
        fit.qdm.adjusted <- QDM(o.c = cccma$rcm.c[, i], m.c = cccma$gcm.c[, i], m.p = cccma$gcm.p[, i], 
                               ratio = cccma$ratio.seq[i], 
                               trace = adjusted_trace_r, 
                               trace.calc = adjusted_trace_calc_r,
                               debug_name = current_debug_name)
        temp_qdm_c <- fit.qdm.adjusted$mhat.c # Update both if QDM is re-run
        temp_qdm_p <- fit.qdm.adjusted$mhat.p
      }
    } else {
      cat(paste0("R UNIQDM LOOP - Var: ", colnames(cccma$gcm.c)[i], ", Skipping correlation check due to zero variance in time series.\n"))
    }
  }
  qdm.c[, i] <- temp_qdm_c
  qdm.p[, i] <- temp_qdm_p
}

# --- GCM_P vs QDM_P Histograms and Time Series Plots for all variables ---
cat("\nGenerating GCM_P vs QDM_P R histograms and time series plots for all variables...\n")
for (plot_var_idx in 1:ncol(cccma$gcm.p)) {
  plot_var_name <- colnames(cccma$gcm.p)[plot_var_idx]
  gcm_p_var_data_r <- cccma$gcm.p[, plot_var_idx]
  qdm_p_var_data_r <- qdm.p[, plot_var_idx]

  # Determine common range and bins for histograms
  combined_data_for_hist_r <- c(gcm_p_var_data_r, qdm_p_var_data_r)
  min_val_r <- min(combined_data_for_hist_r, na.rm = TRUE)
  max_val_r <- max(combined_data_for_hist_r, na.rm = TRUE)
  # Define breaks for 30 bins covering the common range
  hist_breaks_r <- seq(min_val_r, max_val_r, length.out = 31) 
  # Ensure breaks are unique in case min_val_r == max_val_r
  if (length(unique(hist_breaks_r)) < 2) {
    hist_breaks_r <- seq(min_val_r - 0.5, max_val_r + 0.5, length.out = 31)
    if (length(unique(hist_breaks_r)) < 2) hist_breaks_r <- 30 # Fallback
  }


  # --- Histograms ---
  png_hist_filename <- paste0(plot_var_name, "_gcm_p_vs_qdm_p_histograms_r.png")
  png(png_hist_filename, width=1000, height=500)
  par(mfrow = c(1, 2))
  
  hist(gcm_p_var_data_r, breaks = hist_breaks_r, xlim = c(min_val_r, max_val_r), col = "blue", 
       main = paste("Histogram of Original GCM Projection Data for", toupper(plot_var_name)), 
       xlab = paste(toupper(plot_var_name), "Value"))
  
  hist(qdm_p_var_data_r, breaks = hist_breaks_r, xlim = c(min_val_r, max_val_r), col = "green", 
       main = paste("Histogram of QDM Processed Data for", toupper(plot_var_name), "(Projection)"), 
       xlab = paste(toupper(plot_var_name), "Value"))
  
  dev.off()
  cat(toupper(plot_var_name), "GCM_P vs QDM_P R histograms saved to", png_hist_filename, "\n")

  # --- Time Series Plots ---
  png_ts_filename <- paste0(plot_var_name, "_gcm_p_vs_qdm_p_timeseries_r.png")
  png(png_ts_filename, width=1000, height=800)
  par(mfrow = c(2, 1), mar = c(4, 4, 2, 1)) 

  time_axis_p_r <- seq_len(length(gcm_p_var_data_r))

  plot(time_axis_p_r, gcm_p_var_data_r, type = 'l', col = "blue",
       main = paste("Time Series of Original GCM Projection Data for", toupper(plot_var_name)), 
       xlab = "Time Index", ylab = paste(toupper(plot_var_name), "Value"))
  grid()
  
  plot(time_axis_p_r, qdm_p_var_data_r, type = 'l', col = "green", 
       main = paste("Time Series of QDM Processed Data for", toupper(plot_var_name), "(Projection)"), 
       xlab = "Time Index", ylab = paste(toupper(plot_var_name), "Value"))
  grid()
  
  dev.off()
  cat(toupper(plot_var_name), "GCM_P vs QDM_P R time series plots saved to", png_ts_filename, "\n")
}

# Multivariate MBCp bias correction
fit.mbcp <- MBCp(o.c = cccma$rcm.c, m.c = cccma$gcm.c, m.p = cccma$gcm.p, 
                  ratio.seq = cccma$ratio.seq, trace = cccma$trace)
mbcp.c <- fit.mbcp$mhat.c
mbcp.p <- fit.mbcp$mhat.p

# Multivariate MBCr bias correction
fit.mbcr <- MBCr(o.c = cccma$rcm.c, m.c = cccma$gcm.c, m.p = cccma$gcm.p, 
                  ratio.seq = cccma$ratio.seq, trace = cccma$trace)
mbcr.c <- fit.mbcr$mhat.c
mbcr.p <- fit.mbcr$mhat.p

# Multivariate MBCn bias correction
fit.mbcn <- MBCn(o.c = cccma$rcm.c, m.c = cccma$gcm.c, m.p = cccma$gcm.p, 
                  ratio.seq = cccma$ratio.seq, trace = cccma$trace,
                  n.escore = 100, silent = FALSE) # Added n.escore and silent
mbcn.c <- fit.mbcn$mhat.c
mbcn.p <- fit.mbcn$mhat.p
colnames(mbcn.c) <- colnames(mbcn.p) <- colnames(cccma$rcm.c)

# Correlation matrices (Pearson and Spearman)
# MBCp
dev.new()
par(mfrow = c(2, 2))

# Pearson correlation plots for MBCp calibration
plot(c(cor(cccma$rcm.c)), c(cor(qdm.c)), col = 'black', pch = 19, 
     xlim = c(-1, 1), ylim = c(-1, 1), 
     xlab = 'CanRCM4', ylab = 'CanESM2 MBCp', 
     main = 'Pearson correlation\nMBCp calibration')
abline(0, 1)
grid()
points(c(cor(cccma$rcm.c)), c(cor(mbcp.c)), col = 'red')

# Pearson correlation plots for MBCp evaluation
plot(c(cor(cccma$rcm.p)), c(cor(qdm.p)), col = 'black', pch = 19, 
     xlim = c(-1, 1), ylim = c(-1, 1), 
     xlab = 'CanRCM4', ylab = 'CanESM2 MBCp', 
     main = 'Pearson correlation\nMBCp evaluation')
abline(0, 1)
grid()
points(c(cor(cccma$rcm.p)), c(cor(mbcp.p)), col = 'red')

# Spearman correlation plots for MBCp calibration
plot(c(cor(cccma$rcm.c, method = 'spearman')), c(cor(qdm.c, method = 'spearman')), 
     col = 'black', pch = 19, xlim = c(-1, 1), ylim = c(-1, 1), 
     xlab = 'CanRCM4', ylab = 'CanESM2 MBCp', 
     main = 'Spearman correlation\nMBCp calibration')
abline(0, 1)
grid()
points(c(cor(cccma$rcm.c, method = 'spearman')), c(cor(mbcp.c, method = 'spearman')), col = 'red')

# Spearman correlation plots for MBCp evaluation
plot(c(cor(cccma$rcm.p, method = 'spearman')), c(cor(qdm.p, method = 'spearman')), 
     col = 'black', pch = 19, xlim = c(-1, 1), ylim = c(-1, 1), 
     xlab = 'CanRCM4', ylab = 'CanESM2 MBCp', 
     main = 'Spearman correlation\nMBCp evaluation')
abline(0, 1)
grid()
points(c(cor(cccma$rcm.p, method = 'spearman')), c(cor(mbcp.p, method = 'spearman')), col = 'red')

# Repeat for MBCr and MBCn using similar structure
# MBCr plots...
# MBCn plots...

# Helper function for histogram on pairs plot diagonal
panel.hist <- function(x, ...) {
    usr <- par("usr"); on.exit(par(usr))
    par(usr = c(usr[1:2], 0, 1.5) )
    h <- hist(x, plot = FALSE, breaks=20) # Use 20 bins, similar to Python
    breaks <- h$breaks; nB <- length(breaks)
    y <- h$counts; y <- y/max(y)
    rect(breaks[-nB], 0, breaks[-1], y, ...) # Removed col="cyan"
}

# Pairwise scatterplots
dev.new()
pairs(cccma$gcm.c, main = 'CanESM2 calibration', col = '#0000001A', diag.panel = panel.hist, upper.panel=NULL, lower.panel=panel.smooth)
dev.new()
pairs(cccma$rcm.c, main = 'CanRCM4 calibration', col = '#0000001A', diag.panel = panel.hist, upper.panel=NULL, lower.panel=panel.smooth)
dev.new()
pairs(qdm.c, main = 'QDM calibration', col = '#0000001A', diag.panel = panel.hist, upper.panel=NULL, lower.panel=panel.smooth)
dev.new()
pairs(mbcp.c, main = 'MBCp calibration', col = '#FF00001A', diag.panel = panel.hist, upper.panel=NULL, lower.panel=panel.smooth)
dev.new()
pairs(mbcr.c, main = 'MBCr calibration', col = '#0000FF1A', diag.panel = panel.hist, upper.panel=NULL, lower.panel=panel.smooth)
dev.new()
pairs(mbcn.c, main = 'MBCn calibration', col = '#FFA5001A', diag.panel = panel.hist, upper.panel=NULL, lower.panel=panel.smooth)

# Energy distance skill score relative to univariate QDM
escore.qdm <- escore(cccma$rcm.p, qdm.p, scale.x = TRUE)
escore.mbcp <- escore(cccma$rcm.p, mbcp.p, scale.x = TRUE)
escore.mbcr <- escore(cccma$rcm.p, mbcr.p, scale.x = TRUE)
escore.mbcn <- escore(cccma$rcm.p, mbcn.p, scale.x = TRUE)

cat('ESS (MBCp):', 1 - escore.mbcp / escore.qdm, '\n')
cat('ESS (MBCr):', 1 - escore.mbcr / escore.qdm, '\n')
cat('ESS (MBCn):', 1 - escore.mbcn / escore.qdm, '\n')

## End(Not run)
