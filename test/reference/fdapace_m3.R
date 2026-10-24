# Run from the repository root: Rscript test/reference/fdapace_m3.R
# Reference implementation audited against CRAN fdapace 0.6.0.
# This script requires R and fdapace only; CI consumes the committed CSVs.
library(fdapace)
options(digits = 17)
dir.create("test/data", showWarnings = FALSE)
header <- paste("#", R.version.string, "; fdapace", packageVersion("fdapace"))
write_reference <- function(x, path, description) {
    writeLines(c(header, paste("#", description)), path)
    write.table(x, path, append = TRUE, sep = ",", row.names = FALSE,
                col.names = FALSE, quote = FALSE)
}

# Deterministic rank-five regular curves. Neither R's nor Julia's RNG is used.
# Nonzero endpoints expose fdapace's rectangular eigenvalue scaling versus
# the package's trapezoidal operator discretization.
n <- 18L
grid <- seq(0, 1, length.out = 21L)
population_lambda <- c(4, 1, .3, .08, .02)
phi <- outer(grid, 0:4, function(t, k) cos(pi * t * k))
phi[, 2:5] <- sqrt(2) * phi[, 2:5]
mu <- as.vector(phi %*% c(2, .4, .3, .2, .1))
xi <- outer(seq_len(n), seq_len(5L), function(i, k) cos(2 * pi * i * k / n))
xi <- sweep(xi, 2, sqrt(2 * population_lambda), "*")
curves <- sweep(xi %*% t(phi), 2, mu, "+")
input <- do.call(rbind, lapply(seq_len(n), function(i) {
    data.frame(curve = i, time = grid, value = curves[i, ])
}))
write_reference(input, "test/data/m3_dense_input.csv", "columns: curve,time,value")

Ly <- lapply(seq_len(n), function(i) as.vector(curves[i, ]))
Lt <- rep(list(grid), n)
common <- list(dataType = "Dense", methodMuCovEst = "cross-sectional",
               error = FALSE, methodXi = "IN", nRegGrid = length(grid),
               maxK = 5L, useBinnedData = "OFF", verbose = FALSE)
fixed <- FPCA(Ly, Lt, c(common, list(methodSelectK = 3L)))
fve_fit <- FPCA(Ly, Lt, c(common, list(methodSelectK = "FVE", FVEthreshold = .99)))
full <- FPCA(Ly, Lt, c(common, list(methodSelectK = 5L)))
stopifnot(length(fixed$workGrid) == length(grid),
          max(abs(fixed$workGrid - grid)) < 1e-14,
          fixed$selectK == 3L, length(full$cumFVE) == 5L)

# Long format: quantity,index,component,value (without a column-name row).
# mean/eigenfunction: index is the work-grid position.
# eigenvalue/all_eigenvalue/fve: index is the component number, component is zero.
# all_eigenvalue/fve come from the full-K run because SubsetFPCA truncates
# the reported cumFVE to the selected K.
# score: index is the subject number; component is the component number.
# selected_k: index=1 is the fixed-K run; index=2 is the FVE(.99) run.
# covariance: index is the row position, component is the column position.
rows <- function(quantity, index, component, value) {
    data.frame(quantity, index, component, value)
}
reference <- rbind(
    rows("mean", seq_along(grid), 0L, fixed$mu),
    rows("eigenvalue", seq_along(fixed$lambda), 0L, fixed$lambda),
    rows("all_eigenvalue", seq_along(full$lambda), 0L, full$lambda),
    do.call(rbind, lapply(seq_len(ncol(fixed$phi)), function(k) {
        rows("eigenfunction", seq_along(grid), k, fixed$phi[, k])
    })),
    rows("fve", seq_along(full$cumFVE), 0L, full$cumFVE),
    do.call(rbind, lapply(seq_len(ncol(fixed$xiEst)), function(k) {
        rows("score", seq_len(n), k, fixed$xiEst[, k])
    })),
    rows("selected_k", 1:2, 0L, c(fixed$selectK, fve_fit$selectK)),
    do.call(rbind, lapply(seq_along(grid), function(j) {
        rows("covariance", seq_along(grid), j, fixed$smoothedCov[, j])
    }))
)
write_reference(reference, "test/data/m3_fdapace_dense.csv",
                "columns: quantity,index,component,value; selected_k: 1=fixed K=3, 2=FVE(.99)")
cat(header, "\n")
cat("Dense fixture: n=", n, ", m=", length(grid), ", maxK=5; fixed K=",
    fixed$selectK, ", FVE(.99) K=", fve_fit$selectK, "\n", sep = "")
cat("Eigenvalues:", format(fixed$lambda, digits = 17), "\n")
