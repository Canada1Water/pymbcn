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
  fit.qdm <- QDM(o.c = cccma$rcm.c[, i], m.c = cccma$gcm.c[, i], m.p = cccma$gcm.p[, i], 
                 ratio = cccma$ratio.seq[i], trace = cccma$trace[i], debug_name = current_debug_name)
  qdm.c[, i] <- fit.qdm$mhat.c
  qdm.p[, i] <- fit.qdm$mhat.p
}

# Huss Histograms (Original GCM_P vs QDM_P)
if ("huss" %in% colnames(cccma$gcm.p)) {
  huss_idx_r <- which(colnames(cccma$gcm.p) == "huss")
  
  png("huss_gcm_p_vs_qdm_p_histograms_r.png", width=1000, height=500)
  par(mfrow = c(1, 2))
  
  hist(cccma$gcm.p[, huss_idx_r], breaks = 30, col = "blue", 
       main = "Histogram of Original GCM Projection Data for Huss", xlab = "Huss Value")
  
  hist(qdm.p[, huss_idx_r], breaks = 30, col = "green", 
       main = "Histogram of QDM Processed Data for Huss (Projection)", xlab = "Huss Value")
  
  dev.off()
  cat("Huss GCM_P vs QDM_P R histograms saved to huss_gcm_p_vs_qdm_p_histograms_r.png\n")

  # Huss Time Series Plots (Original GCM_P vs QDM_P)
  png("huss_gcm_p_vs_qdm_p_timeseries_r.png", width=1000, height=800)
  par(mfrow = c(2, 1), mar = c(4, 4, 2, 1)) # Adjust margins for titles

  time_axis_p_r <- seq_len(nrow(cccma$gcm.p))

  plot(time_axis_p_r, cccma$gcm.p[, huss_idx_r], type = 'l', col = "blue",
       main = "Time Series of Original GCM Projection Data for Huss", 
       xlab = "Time Index", ylab = "Huss Value")
  grid()
  
  plot(time_axis_p_r, qdm.p[, huss_idx_r], type = 'l', col = "green", 
       main = "Time Series of QDM Processed Data for Huss (Projection)", 
       xlab = "Time Index", ylab = "Huss Value")
  grid()
  
  dev.off()
  cat("Huss GCM_P vs QDM_P R time series plots saved to huss_gcm_p_vs_qdm_p_timeseries_r.png\n")
  
} else {
  cat("Huss variable not found for R histogram and time series plotting.\n")
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

# Additional debug for huss in R
if ("huss" %in% colnames(cccma$gcm.c)) {
  huss_idx <- which(colnames(cccma$gcm.c) == "huss")
  cat("\n--- R HUSS QDM.C COMPARISON ---\n")
  cat("Summary of cccma$rcm.c[,huss]:\n"); print(summary(cccma$rcm.c[,huss_idx])); cat("Head:\n"); print(head(cccma$rcm.c[,huss_idx]))
  cat("Summary of cccma$gcm.c[,huss]:\n"); print(summary(cccma$gcm.c[,huss_idx])); cat("Head:\n"); print(head(cccma$gcm.c[,huss_idx]))
  cat("Summary of qdm.c[,huss] (from mhat.c):\n"); print(summary(qdm.c[,huss_idx])); cat("Head:\n"); print(head(qdm.c[,huss_idx]))
  cat("Are cccma$gcm.c[,huss] and qdm.c[,huss] all.equal? ", all.equal(cccma$gcm.c[,huss_idx], qdm.c[,huss_idx]), "\n")
  cat("Are cccma$rcm.c[,huss] and cccma$gcm.c[,huss] all.equal? ", all.equal(cccma$rcm.c[,huss_idx], cccma$gcm.c[,huss_idx]), "\n")
}

## End(Not run)
