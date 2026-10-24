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

# Sparse conditional-expectation fixture. All five sine eigenfunctions vanish
# at the endpoints, so fdapace's rectangular operator and our trapezoidal
# operator agree here. The affine supplied mean also agrees under fdapace's
# cubic-spline userMu interpolation and our documented linear interpolation.
sparse_n <- 40L
sparse_grid <- seq(0, 1, length.out = 21L)
observation_grid <- seq(0, 1, length.out = 41L)
sparse_lambda <- c(2, .7, .25, .06, .015)
sparse_phi <- sqrt(2) * outer(sparse_grid, seq_len(5L), function(t, k) sin(pi * t * k))
sparse_mu <- 2 + sparse_grid
sparse_cov <- sparse_phi %*% diag(sparse_lambda) %*% t(sparse_phi)
sparse_cov <- (sparse_cov + t(sparse_cov)) / 2
sparse_sigma2 <- .12
sparse_times <- lapply(seq_len(sparse_n), function(i) {
    ni <- 2L + (i - 1L) %% 7L
    positions <- sort(((i - 1L) * 3L + seq.int(0L, ni - 1L) * 7L) %% 41L + 1L)
    observation_grid[positions]
})
stopifnot(identical(sort(unique(unlist(sparse_times))), observation_grid))
sparse_values <- lapply(seq_len(sparse_n), function(i) {
    ti <- sparse_times[[i]]
    xi <- sqrt(2 * sparse_lambda) * cos(2 * pi * i * seq_len(5L) / sparse_n)
    interpolated_phi <- sapply(seq_len(5L), function(k) approx(sparse_grid, sparse_phi[, k], ti)$y)
    as.vector(2 + ti + interpolated_phi %*% xi +
              sqrt(2 * sparse_sigma2) * sin(13 * i + 7 * seq_along(ti)))
})
sparse_input <- do.call(rbind, lapply(seq_len(sparse_n), function(i) {
    data.frame(curve = i, time = sparse_times[[i]], value = sparse_values[[i]])
}))
write_reference(sparse_input, "test/data/m3_sparse_input.csv", "columns: curve,time,value; deterministic 40 curves with 2 to 8 observations")
write_reference(data.frame(time = sparse_grid, value = sparse_mu),
                "test/data/m3_sparse_mean.csv", "columns: time,value; affine supplied mean and PACE work grid")
write_reference(sparse_cov, "test/data/m3_sparse_surface.csv", "supplied symmetric covariance matrix; rows and columns follow m3_sparse_mean.csv")
write_reference(sparse_sigma2, "test/data/m3_sparse_noise.csv", "supplied measurement-error variance sigma2")

sparse_common <- list(dataType = "Sparse", error = TRUE, methodXi = "CE",
    methodMuCovEst = "smooth", nRegGrid = length(sparse_grid), maxK = 5L,
    FVEthreshold = 1, useBinnedData = "OFF", methodRho = "vanilla",
    userMu = list(t = sparse_grid, mu = sparse_mu),
    userCov = list(t = sparse_grid, cov = sparse_cov),
    userSigma2 = sparse_sigma2, verbose = FALSE)
sparse_fixed <- FPCA(sparse_values, sparse_times, c(sparse_common, list(methodSelectK = 3L)))
sparse_full <- FPCA(sparse_values, sparse_times, c(sparse_common, list(methodSelectK = 5L)))
sparse_aic <- FPCA(sparse_values, sparse_times, c(sparse_common, list(methodSelectK = "AIC")))
sparse_bic <- FPCA(sparse_values, sparse_times, c(sparse_common, list(methodSelectK = "BIC")))
sparse_fve_options <- modifyList(sparse_common, list(methodSelectK = "FVE", FVEthreshold = .99))
sparse_fve <- FPCA(sparse_values, sparse_times, sparse_fve_options)
stopifnot(max(abs(sparse_fixed$workGrid - sparse_grid)) < 1e-14,
          sparse_fixed$selectK == 3L, length(sparse_full$lambda) == 5L,
          sparse_fixed$rho == 0, sparse_fixed$sigma2 == sparse_sigma2)

# Include off-work-grid times that occur on the training union obsGrid.
# predict.FPCA interpolates Sigma AFTER adding noise on obsGrid; restricting
# held-out times to that union avoids its separate off-union noise-interpolation
# behavior, and compares the documented G(t_i,t_i) + sigma2*I quantity.
heldout_times <- list(observation_grid[c(2, 8, 14, 20, 26, 32, 38)],
                     observation_grid[c(1, 5, 11, 17, 23, 29, 35, 41)],
                     observation_grid[c(4, 10, 16, 22, 28, 34, 40)])
heldout_values <- lapply(seq_along(heldout_times), function(i) {
    ti <- heldout_times[[i]]
    xi <- sqrt(sparse_lambda) * sin(i * seq_len(5L))
    interpolated_phi <- sapply(seq_len(5L), function(k) approx(sparse_grid, sparse_phi[, k], ti)$y)
    as.vector(2 + ti + interpolated_phi %*% xi + .1 * cos(3 * i + seq_along(ti)))
})
heldout_input <- do.call(rbind, lapply(seq_along(heldout_times), function(i) {
    data.frame(curve = i, time = heldout_times[[i]], value = heldout_values[[i]])
}))
write_reference(heldout_input, "test/data/m3_sparse_heldout.csv", "columns: curve,time,value; held-out curves, including times between work-grid nodes")
heldout_prediction <- predict(sparse_fixed, heldout_values, heldout_times, xiMethod = "CE")

# GetLogLik omits the Gaussian constant sum_i N_i*log(2*pi). Keep the raw
# fdapace criteria and full Gaussian criteria so this harmless offset is
# independently visible; both have the same stopping rule and global minimum.
sparse_deviance <- vapply(seq_len(5L), function(k) as.numeric(fdapace:::GetLogLik(sparse_full, k)), 0.0)
gaussian_constant <- sum(lengths(sparse_times)) * log(2 * pi)
raw_aic <- sparse_deviance + 2 * seq_len(5L)
raw_bic <- sparse_deviance + log(sparse_n) * seq_len(5L)
stopifnot(abs(sparse_aic$criterionValue - raw_aic[sparse_aic$selectK]) < 1e-10,
          abs(sparse_bic$criterionValue - raw_bic[sparse_bic$selectK]) < 1e-10)

sparse_reference <- rbind(
    rows("mean", seq_along(sparse_grid), 0L, sparse_fixed$mu),
    rows("eigenvalue", seq_along(sparse_fixed$lambda), 0L, sparse_fixed$lambda),
    rows("all_eigenvalue", seq_along(sparse_full$lambda), 0L, sparse_full$lambda),
    do.call(rbind, lapply(seq_len(3L), function(k) {
        rows("eigenfunction", seq_along(sparse_grid), k, sparse_fixed$phi[, k])
    })),
    rows("fve", seq_along(sparse_full$cumFVE), 0L, sparse_full$cumFVE),
    do.call(rbind, lapply(seq_len(3L), function(k) {
        rows("score", seq_len(sparse_n), k, sparse_fixed$xiEst[, k])
    })),
    do.call(rbind, lapply(seq_len(sparse_n), function(i) {
        # SubsetFPCA truncates xiEst/phi/lambda but leaves xiVar at full rank.
        rows("score_covariance", i, seq_len(9L), as.vector(sparse_fixed$xiVar[[i]][1:3, 1:3]))
    })),
    rows("selected_k", 1:4, 0L,
         c(sparse_fixed$selectK, sparse_aic$selectK, sparse_bic$selectK, sparse_fve$selectK)),
    rows("global_minimizer", 1:2, 0L, c(which.min(raw_aic), which.min(raw_bic))),
    rows("aic", seq_len(5L), 0L, raw_aic + gaussian_constant),
    rows("bic", seq_len(5L), 0L, raw_bic + gaussian_constant),
    rows("fdapace_aic", seq_len(5L), 0L, raw_aic),
    rows("fdapace_bic", seq_len(5L), 0L, raw_bic),
    rows("gaussian_constant", 1L, 0L, gaussian_constant),
    do.call(rbind, lapply(seq_along(sparse_grid), function(j) {
        rows("covariance", seq_along(sparse_grid), j, sparse_fixed$smoothedCov[, j])
    })),
    do.call(rbind, lapply(seq_along(sparse_grid), function(j) {
        rows("fitted_covariance", seq_along(sparse_grid), j, sparse_fixed$fittedCov[, j])
    })),
    do.call(rbind, lapply(seq_len(3L), function(k) {
        rows("prediction_score", seq_along(heldout_times), k, heldout_prediction$scores[, k])
    }))
)
write_reference(sparse_reference, "test/data/m3_fdapace_pace.csv",
    paste("columns: quantity,index,component,value; score_covariance component is vec(3x3) index;",
          "selected_k: 1=fixed K=3, 2=AIC, 3=BIC, 4=FVE(.99); global_minimizer: 1=AIC, 2=BIC;",
          "aic/bic include Gaussian constant; fdapace_aic/fdapace_bic omit it"))
cat("Sparse fixture: n=", sparse_n, ", m=", length(sparse_grid),
    ", maxK=5; fixed/AIC/BIC/FVE(.99) K=", paste(c(sparse_fixed$selectK,
    sparse_aic$selectK, sparse_bic$selectK, sparse_fve$selectK), collapse = "/"), "\n", sep = "")
cat("Full Gaussian AIC:", format(raw_aic + gaussian_constant, digits = 17), "\n")
cat("Full Gaussian BIC:", format(raw_bic + gaussian_constant, digits = 17), "\n")
