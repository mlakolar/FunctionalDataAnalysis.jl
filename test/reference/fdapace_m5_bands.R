# Run from the repository root: Rscript test/reference/fdapace_m5_bands.R
# Reference: R 4.5.1, fdapace 0.6.0, stats::qchisq / stats::qnorm.
# CI consumes the committed CSVs and does not run R.
library(fdapace)
stopifnot(packageVersion("fdapace") == "0.6.0")
options(digits = 17)
read_fixture <- function(name) unname(as.matrix(read.csv(paste0("test/data/", name),
    header = FALSE, comment.char = "#")))
input <- read_fixture("m3_sparse_input.csv")
mean_input <- read_fixture("m3_sparse_mean.csv")
grid <- mean_input[, 1]
Ly <- lapply(seq_len(max(input[, 1])), function(i) input[input[, 1] == i, 3])
Lt <- lapply(seq_along(Ly), function(i) input[input[, 1] == i, 2])
optns <- list(dataType = "Sparse", error = TRUE, methodXi = "CE",
    methodMuCovEst = "smooth", nRegGrid = length(grid), maxK = 5L,
    FVEthreshold = 1, useBinnedData = "OFF", methodRho = "vanilla",
    userMu = list(t = grid, mu = mean_input[, 2]),
    userCov = list(t = grid, cov = read_fixture("m3_sparse_surface.csv")),
    userSigma2 = as.numeric(read_fixture("m3_sparse_noise.csv")), verbose = FALSE)
header <- paste("#", R.version.string, "; fdapace", packageVersion("fdapace"))
write_reference <- function(x, path, description) {
    writeLines(c(header, paste("#", description)), path)
    write.table(x, path, append = TRUE, sep = ",", row.names = FALSE,
                col.names = FALSE, quote = FALSE)
}
reference <- list()
for (K in c(1L, 3L, 5L)) {
    object <- FPCA(Ly, Lt, c(optns, list(methodSelectK = K)))
    # userMu leaves bwMu NULL. fitted.FPCA computes and discards muObs/phiObs
    # using it even for p=0; provide a valid bandwidth for those dead calculations.
    object$bwMu <- .3
    # SubsetFPCA leaves xiVar at maxK, making native fitted() nonconformable
    # for selected K < maxK. Verify this source-audited behavior explicitly.
    if (K < 5L) {
        native <- try(fitted(object, ciOptns = list(alpha = .05,
                      cvgMethod = "interval")), silent = TRUE)
        stopifnot(inherits(native, "try-error"))
        object$xiVar <- lapply(object$xiVar, function(v) v[seq_len(K), seq_len(K), drop = FALSE])
    }
    for (coverage in c("interval", "band")) {
        bands <- fitted(object, ciOptns = list(alpha = .05, cvgMethod = coverage))
        reference[[length(reference) + 1L]] <- do.call(rbind, lapply(seq_along(Ly), function(i) {
            data.frame(coverage, K, curve = i, position = seq_along(grid),
                fitted = bands$fitted[i, ], lower = bands$cvgLower[i, ], upper = bands$cvgUpper[i, ])
        }))
    }
}
write_reference(do.call(rbind, reference), "test/data/m5_fdapace_bands.csv",
    "columns: coverage,K,curve,grid_position,fitted,lower,upper; alpha=.05; xiVar selected block for K<5; bwMu=.3 for unused intermediates; full covariance in CE solve")
quantiles <- expand.grid(K = c(1L, 2L, 3L, 4L, 5L, 20L, 51L, 200L),
                         alpha = c(1e-10, .001, .05, .5, .9, 1-1e-10))
quantiles$value <- qchisq(quantiles$alpha, df = quantiles$K, lower.tail = FALSE)
write_reference(quantiles, "test/data/m5_chisq.csv",
    "columns: K,alpha,upper_tail_chisq_quantile; stats::qchisq(alpha,K,lower.tail=FALSE)")
cat(header, "\nGenerated fitted bands for K=1,3,5, both coverage types; and chi-square quantiles.\n")
