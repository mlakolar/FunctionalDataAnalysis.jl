# Run from the repository root: Rscript test/reference/regression_m7.R
# References: R 4.5.1; fdapace 0.6.0; fda.usc 2.2.0 (current CRAN).
# Sources audited: fdapace R/FLM.R, R/FLMCI.R, R/SetOptions.R;
# fda.usc R/fregre.pc.R, R/fdata2pc.R, R/predict.fregre.fd.r,
# R/inprod.fdata.R, R/ops.fda.usc.R (CRAN source archives).
# https://cran.r-project.org/src/contrib/fdapace_0.6.0.tar.gz
# https://cran.r-project.org/src/contrib/fda.usc_2.2.0.tar.gz
# CI reads these small CSVs and does not need R.
library(fdapace)
library(fda.usc)
stopifnot(packageVersion("fdapace") == "0.6.0",
          packageVersion("fda.usc") == "2.2.0")
options(digits = 17)
header <- paste("#", R.version.string, "; fdapace", packageVersion("fdapace"),
                "; fda.usc", packageVersion("fda.usc"), "; regression_m7.R")
write_ref <- function(x, name, description) {
    path <- paste0("test/data/m7_", name, ".csv")
    writeLines(c(header, paste("#", description)), path)
    # write.table's numeric conversion uses 15 significant digits even when
    # options(digits=17); preserve inputs such as 1-eps rather than writing 1.
    x <- as.data.frame(x)
    x[] <- lapply(x,function(column) if (is.numeric(column)) sprintf("%.17g",column) else column)
    write.table(x, path, append = TRUE, sep = ",", row.names = FALSE,
                col.names = FALSE, quote = FALSE)
}
rows <- function(quantity, value, component = 0L) {
    data.frame(quantity, index = seq_along(value), component, value = as.vector(value))
}
matrix_rows <- function(quantity, value) {
    do.call(rbind, lapply(seq_len(ncol(value)), function(j) rows(quantity, value[, j], j)))
}

# Deterministic, rank-three/rank-two predictors with vanishing endpoints.
# The rectangular fdapace eigenstep, fda.usc's Euclidean SVD normalization,
# and the package's trapezoidal operator agree on these particular subspaces.
# Means also vanish at endpoints, so the intercept conversion is exact here.
# Neither R nor Julia RNG streams are used.
n <- 24L
grid <- seq(0, 1, length.out = 21L)
phi <- sqrt(2) * outer(grid, 1:3, function(t, k) sin(pi * t * k))
theta <- 2 * pi * seq_len(n) / n
a <- cbind(1.4 * cos(theta), .8 * cos(2 * theta), .4 * cos(3 * theta))
b <- cbind(1.1 * sin(theta), .6 * sin(2 * theta))
mu1 <- as.vector(phi %*% c(.8, .2, -.1))
mu2 <- as.vector(phi[, 1:2] %*% c(.2, -.5))
x1 <- sweep(a %*% t(phi), 2, mu1, "+")
x2 <- sweep(b %*% t(phi[, 1:2]), 2, mu2, "+")
z <- .2 + sin(4 * theta) + .3 * cos(theta)
err <- .18 * sin(5 * theta) + .04 * cos(7 * theta)
y <- 1.3 + as.vector(a %*% c(1.2, -.7, .3)) +
    as.vector(b %*% c(.8, .4)) + .5 * z + err
y_single <- -.6 + as.vector(a %*% c(1.1, -.4, .7)) + err
atest <- rbind(c(.3, -.4, .1), c(-.2, .15, .25), c(.6, .2, -.15))
btest <- rbind(c(.2, .4), c(-.1, .3), c(.5, -.2))
ztest <- c(-.4, .7, .1)
xtest1 <- sweep(atest %*% t(phi), 2, mu1, "+")
xtest2 <- sweep(btest %*% t(phi[, 1:2]), 2, mu2, "+")
write_ref(grid, "dense_grid", "common grid for dense fixtures")
write_ref(x1, "dense_input1", "24 subjects by 21 grid points; functional predictor 1")
write_ref(x2, "dense_input2", "24 subjects by 21 grid points; functional predictor 2")
write_ref(cbind(y, z, y_single), "dense_response", "columns: response_multi,scalar_covariate,response_single")
write_ref(xtest1, "dense_heldout1", "three new curves of predictor 1; same grid")
write_ref(xtest2, "dense_heldout2", "three new curves of predictor 2; same grid")
write_ref(ztest, "dense_heldout_scalar", "three new scalar covariates")
as_fdapace <- function(x) list(Ly = lapply(seq_len(nrow(x)), function(i) x[i, ]),
                             Lt = rep(list(grid), nrow(x)))
dense_options <- function(k) list(dataType = "Dense", methodMuCovEst = "cross-sectional",
    error = FALSE, methodXi = "IN", shrink = FALSE, nRegGrid = length(grid),
    maxK = k, methodSelectK = k, useBinnedData = "OFF", verbose = FALSE)
multi <- FLM(y, list(as_fdapace(x1), as_fdapace(x2), z),
    XTest = list(as_fdapace(xtest1), as_fdapace(xtest2), ztest),
    optnsListX = list(dense_options(3L), dense_options(2L)))
# Leave the FPCA component rule at fdapace's FVE default.
default_options <- lapply(c(3L,2L),function(k) {
    opt <- dense_options(k)
    opt$methodSelectK <- NULL
    opt$maxK <- NULL
    opt
})
default_multi <- FLM(y,list(as_fdapace(x1),as_fdapace(x2),z),optnsListX=default_options)
stopifnot(identical(vapply(default_multi$estXi[1:2],ncol,1L),c(3L,2L)))

# FLM does not expose covariance/SE. The conditional lm calculation below
# validates the same score regression, without claiming FLMCI bootstrap parity.
# Coefficient order in the lm covariance is intercept, all PC scores, scalar.
conditional_output <- function(flm, response, phis, new_scores, scalar = NULL,
                               new_scalar = NULL) {
    design <- do.call(cbind, flm$estXi)
    object <- lm(response ~ design)
    covariance <- vcov(object)
    sigma2 <- summary(object)$sigma^2
    new_design <- cbind(1, do.call(cbind, new_scores))
    pred <- as.vector(new_design %*% coef(object))
    pred_se <- sqrt(rowSums((new_design %*% covariance) * new_design))
    crit <- qt(.975, object$df.residual)
    out <- rbind(rows("centered_intercept", coef(object)[1]),
        rows("native_intercept", flm$alpha), rows("fitted", flm$yHat),
        rows("residual", response - flm$yHat), rows("sigma2", sigma2),
        rows("r2", flm$R2), rows("df_residual", object$df.residual),
        rows("prediction", flm$yPred), rows("prediction_se", pred_se),
        rows("confidence_lower", pred - crit * pred_se),
        rows("confidence_upper", pred + crit * pred_se),
        rows("prediction_lower", pred - crit * sqrt(pred_se^2 + sigma2)),
        rows("prediction_upper", pred + crit * sqrt(pred_se^2 + sigma2)),
        rows("critical_value", crit), rows("score_coefficient", coef(object)),
        matrix_rows("coefficient_covariance", covariance))
    start <- 2L
    for (j in seq_along(phis)) {
        k <- ncol(phis[[j]])
        block <- covariance[start:(start + k - 1L), start:(start + k - 1L), drop = FALSE]
        se <- sqrt(pmax(0, rowSums((phis[[j]] %*% block) * phis[[j]])))
        out <- rbind(out, rows(paste0("beta", j), flm$betaList[[j]]),
                     rows(paste0("beta", j, "_se"), se),
                     rows(paste0("beta", j, "_lower"), flm$betaList[[j]] - crit * se),
                     rows(paste0("beta", j, "_upper"), flm$betaList[[j]] + crit * se),
                     matrix_rows(paste0("score", j), flm$estXi[[j]]))
        start <- start + k
    }
    if (!is.null(scalar)) out <- rbind(out, rows("scalar_coefficient", tail(coef(object), length(scalar))))
    stopifnot(max(abs(pred - as.vector(flm$yPred))) < 1e-12)
    out
}
write_ref(conditional_output(multi, y, multi$phiX, multi$testXi, scalar = 1),
    "fdapace_dense", "quantity,index,component,value; real FLM fit plus explicitly conditional OLS uncertainty")

# fregre.pc's unpenalized default uses l=1:3. Its default inprod integration
# is TRAPZ (ops.fda.usc), although the helper is named int.simpson2.
single <- fregre.pc(fdata(x1, grid), y_single)
new_fd <- fdata(xtest1, grid)
confidence <- predict(single, new_fd, se.fit = TRUE, interval = "confidence")
prediction <- predict(single, new_fd, interval = "prediction")
beta_cov <- single$Vp[-1, -1, drop = FALSE]
single_phi <- t(single$fdata.comp$basis$data)
beta_se <- sqrt(pmax(0, rowSums((single_phi %*% beta_cov) * single_phi)))
crit <- qt(.975, single$df.residual)
single_rows <- rbind(rows("centered_intercept", single$coefficients[1]),
    rows("beta1", single$beta.est$data), rows("beta1_se", beta_se),
    rows("beta1_lower", single$beta.est$data - crit * beta_se),
    rows("beta1_upper", single$beta.est$data + crit * beta_se),
    rows("fitted", single$fitted.values), rows("residual", single$residuals),
    rows("sigma2", single$sr2), rows("r2", single$r2),
    rows("df_residual", single$df.residual), rows("selected_components", single$l),
    rows("prediction", confidence$fit[, 1]), rows("prediction_se", confidence$se.fit),
    rows("confidence_lower", confidence$fit[, 2]), rows("confidence_upper", confidence$fit[, 3]),
    rows("prediction_lower", prediction[, 2]), rows("prediction_upper", prediction[, 3]),
    rows("critical_value", crit), rows("score_coefficient", single$coefficients),
    matrix_rows("coefficient_covariance", single$Vp),
    matrix_rows("score1", single$fdata.comp$coefs[, single$l]))
write_ref(single_rows, "fdausc_dense", "quantity,index,component,value; real fregre.pc default l=1:3 and predict.fregre.fd intervals")

# Reuse the small milestone-3 sparse inputs, marginal covariance and noise.
# The five sine eigenfunctions vanish at endpoints; affine supplied mean and
# linear eigenfunction interpolation have the same convention in both packages.
input <- read.csv("test/data/m3_sparse_input.csv", comment.char = "#", header = FALSE)
mean_input <- read.csv("test/data/m3_sparse_mean.csv", comment.char = "#", header = FALSE)
covariance <- unname(as.matrix(read.csv("test/data/m3_sparse_surface.csv", comment.char = "#", header = FALSE)))
sigma2 <- scan("test/data/m3_sparse_noise.csv", comment.char = "#", quiet = TRUE)
heldout <- read.csv("test/data/m3_sparse_heldout.csv", comment.char = "#", header = FALSE)
get_curves <- function(input) list(
    Ly = lapply(seq_len(max(input[, 1])), function(i) input[input[, 1] == i, 3]),
    Lt = lapply(seq_len(max(input[, 1])), function(i) input[input[, 1] == i, 2]))
sparse_x <- get_curves(input)
sparse_new <- get_curves(heldout)
sparse_n <- length(sparse_x$Ly)
latent <- outer(seq_len(sparse_n), seq_len(5L), function(i, k) cos(2 * pi * i * k / sparse_n))
latent <- sweep(latent, 2, sqrt(2 * c(2, .7, .25, .06, .015)), "*")
sparse_y <- .8 + as.vector(latent[, 1:3] %*% c(1.1, -.5, .3)) + .15 * sin(11 * seq_len(sparse_n))
sparse_options <- list(dataType = "Sparse", error = TRUE, methodXi = "CE",
    methodMuCovEst = "smooth", nRegGrid = nrow(mean_input), maxK = 5L,
    methodSelectK = 3L, FVEthreshold = 1, useBinnedData = "OFF", methodRho = "vanilla",
    userMu = list(t = mean_input[, 1], mu = mean_input[, 2]),
    userCov = list(t = mean_input[, 1], cov = covariance),
    userSigma2 = sigma2, verbose = FALSE)
sparse <- FLM(sparse_y, list(sparse_x), XTest = list(sparse_new), optnsListX = sparse_options)
write_ref(sparse_y, "sparse_response", "response for existing m3_sparse_input.csv")
write_ref(conditional_output(sparse, sparse_y, sparse$phiX, sparse$testXi),
    "fdapace_sparse", "quantity,index,component,value; real FLM on PACE CE scores plus explicitly conditional OLS uncertainty")
cat(header, "\nGenerated dense multiple-predictor FLM, dense fregre.pc, and sparse PACE-score FLM references.\n")

# Leading-component criteria. fda.usc fregre.pc.cv normally performs stepwise
# subset selection. Its CV code additionally divides each held-out residual
# by n-ck (ck=k+2 for a fold). The package deliberately uses ordinary LOOCV
# MSE instead. Compute that documented quantity with real fregre.pc refits.
selection_rows <- NULL
for (k in 1:3) {
    object <- fregre.pc(fdata(x1, grid), y_single, l=seq_len(k))
    rss <- sum(object$residuals^2)
    errors <- sapply(seq_len(n), function(i) {
        fold <- fregre.pc(fdata(x1[-i, , drop=FALSE], grid), y_single[-i], l=seq_len(k))
        (y_single[i] - as.numeric(predict(fold, fdata(x1[i, , drop=FALSE], grid))))^2
    })
    selection_rows <- rbind(selection_rows,
        data.frame(quantity=c("AIC","BIC","loocv","fdausc_scaled_cv"), index=k,
                   component=0, value=c(n*log(rss/n)+2*(k+1),
                       n*log(rss/n)+log(n)*(k+1), mean(errors),mean(errors)/(n-k-2)^2)))
}
write_ref(selection_rows,"selection","leading-count AIC/BIC; full-refit ordinary LOOCV and source-audited fda.usc scaled quantity")

quantiles <- expand.grid(df=c(1L,2L,5L,11L,20L,100L,1000L),alpha=c(.05,.9,1e-6,1-1e-10,1-.Machine$double.eps))
quantiles$value <- qt(quantiles$alpha/2,quantiles$df,lower.tail=FALSE)
write_ref(quantiles,"t_quantiles","columns: residual_dof,alpha,Student_t_two_sided_critical_value")
