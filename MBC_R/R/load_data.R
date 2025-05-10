# Assuming 'cccma' is a data frame loaded in R
load("cccma.RData")
write.csv(cccma$gcm.c, file = "cccma_gcm_c.csv")
write.csv(cccma$gcm.p, file = "cccma_gcm_p.csv")
# Continue exporting other necessary parts of the data similarly
