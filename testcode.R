## Not run:
library("MBC")
# Load necessary data
#data("cccma.RData")
load("cccma.RData") 
set.seed(1)

# Univariate quantile mapping
qdm.c <- cccma$gcm.c * 0
qdm.p <- cccma$gcm.p * 0

for (i in seq(ncol(cccma$gcm.c))) {
  fit.qdm <- QDM(o.c = cccma$rcm.c[, i], m.c = cccma$gcm.c[, i], m.p = cccma$gcm.p[, i], 
                 ratio = cccma$ratio.seq[i], trace = cccma$trace[i])
  qdm.c[, i] <- fit.qdm$mhat.c
  qdm.p[, i] <- fit.qdm$mhat.p
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
                  ratio.seq = cccma$ratio.seq, trace = cccma$trace)
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

# Pairwise scatterplots
dev.new()
pairs(cccma$gcm.c, main = 'CanESM2 calibration', col = '#0000001A')
dev.new()
pairs(cccma$rcm.c, main = 'CanRCM4 calibration', col = '#0000001A')
dev.new()
pairs(qdm.c, main = 'QDM calibration', col = '#0000001A')
dev.new()
pairs(mbcp.c, main = 'MBCp calibration', col = '#FF00001A')
dev.new()
pairs(mbcr.c, main = 'MBCr calibration', col = '#0000FF1A')
dev.new()
pairs(mbcn.c, main = 'MBCn calibration', col = '#FFA5001A')

# Energy distance skill score relative to univariate QDM
escore.qdm <- escore(cccma$rcm.p, qdm.p, scale.x = TRUE)
escore.mbcp <- escore(cccma$rcm.p, mbcp.p, scale.x = TRUE)
escore.mbcr <- escore(cccma$rcm.p, mbcr.p, scale.x = TRUE)
escore.mbcn <- escore(cccma$rcm.p, mbcn.p, scale.x = TRUE)

cat('ESS (MBCp):', 1 - escore.mbcp / escore.qdm, '\n')
cat('ESS (MBCr):', 1 - escore.mbcr / escore.qdm, '\n')
cat('ESS (MBCn):', 1 - escore.mbcn / escore.qdm, '\n')
## End(Not run)

