# Run from the repository root: Rscript test/reference/fdausc_m7g.R
# References: R 4.5.1; fda.usc 2.2.0 (current CRAN) fregre.glm with a
# create.pc.basis(l=1:3) basis, predict.fregre.glm and summary.glm.
# Sources audited: fda.usc R/fregre.glm.R, R/predict.fregre.glm.R,
# R/create.pc.basis.R, R/fdata2basis.R; stats glm.fit and summary.glm.
# https://cran.r-project.org/src/contrib/fda.usc_2.2.0.tar.gz
# fregre.glm calls glm() with the default glm.control(epsilon=1e-8), which
# leaves coefficients about 1e-7 from the maximum likelihood estimate. The
# goldens are therefore the converged estimate: glm.fit on fregre.glm's own
# design (z$x) with epsilon=1e-15, checked against fregre.glm to 1e-6. New
# curves use fda.usc's own projection (fdata2basis and vs.list), as in
# predict.fregre.glm. Inputs are the milestone-7 dense fixtures; the RNG seed
# is consumed only by R. CI reads the committed CSV and does not need R.
suppressMessages(library(fda.usc))
stopifnot(packageVersion("fda.usc") == "2.2.0")
options(digits = 17)
header <- paste("#", R.version.string, "; fda.usc", packageVersion("fda.usc"), "; fdausc_m7g.R")
rd <- function(name) unname(as.matrix(read.csv(paste0("test/data/m7_", name, ".csv"),
                                               comment.char = "#", header = FALSE)))
grid <- drop(rd("dense_grid")); x1 <- rd("dense_input1"); xt <- rd("dense_heldout1")
response <- rd("dense_response"); zscalar <- response[, 2]
ztest <- drop(rd("dense_heldout_scalar"))
n <- nrow(x1)
signal <- (response[, 3] - mean(response[, 3])) / sd(response[, 3])

set.seed(7209)
cases <- list(
    binomial = list(family = binomial(), y = rbinom(n, 1, plogis(0.3 + 1.2 * signal)), scalar = FALSE),
    poisson = list(family = poisson(), y = rpois(n, exp(1 + 0.5 * signal + 0.3 * zscalar)), scalar = TRUE),
    gaussian_log = list(family = gaussian(link = "log"), y = exp(1 + 0.4 * signal) + 0.2 * rnorm(n),
                        scalar = FALSE))
stopifnot(all(cases$gaussian_log$y > 0))

fd <- fdata(x1, grid)
basis <- create.pc.basis(fd, l = 1:3)
newx <- fdata(xt, grid)
new_scores <- fdata2basis(newx, basis)$coefs
tight_control <- glm.control(epsilon = 1e-15, maxit = 100)
rows <- function(case, quantity, value) {
    data.frame(case = case, quantity = quantity, index = seq_along(value), value = as.vector(value))
}
out <- NULL
designs <- list()
for (case in names(cases)) {
    spec <- cases[[case]]
    y <- spec$y
    if (spec$scalar) {
        dat <- list(df = data.frame(y = y, z = zscalar), x = fd)
        object <- fregre.glm(y ~ x + z, family = spec$family, data = dat, basis.x = list(x = basis))
    } else {
        dat <- list(df = data.frame(y = y), x = fd)
        object <- fregre.glm(y ~ x, family = spec$family, data = dat, basis.x = list(x = basis))
    }
    X <- object$x
    designs[[case]] <- X
    tight <- glm.fit(X, y, family = spec$family, control = tight_control)
    stopifnot(tight$converged,
              max(abs(tight$coefficients - coef(object))) < 1e-6 * max(abs(coef(object))))
    b <- tight$coefficients
    # summary.glm: (X'WX)^{-1} times the dispersion (Pearson / df for gaussian).
    dispersion <- if (spec$family$family == "gaussian")
        sum((y - tight$fitted.values)^2) / tight$df.residual else 1
    # glm.fit's weights are those of its last iteration's start; evaluate the
    # Fisher weights at the converged mean instead.
    eta_hat <- tight$linear.predictors
    fisher <- spec$family$mu.eta(eta_hat)^2 / spec$family$variance(tight$fitted.values)
    covariance <- chol2inv(chol(crossprod(X, fisher * X))) * dispersion
    pc <- grep("^x\\.PC", colnames(X))
    beta <- drop(b[pc] %*% basis$basis$data)
    harmonics <- t(basis$basis$data)
    beta_se <- sqrt(rowSums((harmonics %*% covariance[pc, pc]) * harmonics))
    # fda.usc orders the design as intercept, scalar covariates, PC scores.
    Xnew <- cbind(1, if (spec$scalar) ztest, new_scores %*% object$vs.list$x)
    stopifnot(identical(colnames(X)[pc], paste0("x.PC", 1:3)), max(pc) == ncol(X))
    stopifnot(max(abs(drop(Xnew %*% coef(object)) -
        predict(object, list(x = newx, df = data.frame(z = ztest)), type = "link"))) < 1e-12)
    eta_new <- drop(Xnew %*% b)
    se_new <- sqrt(rowSums((Xnew %*% covariance) * Xnew))
    # glm.fit's aic already adds 2 * rank (and the Gaussian family counts its
    # dispersion): R's AIC() and BIC() for glm.
    k <- ncol(X) + (spec$family$family == "gaussian")
    aic <- tight$aic
    cat(case, "dispersion", dispersion, "deviance", tight$deviance, "AIC", aic, "\n")
    out <- rbind(out, rows(case, "y", y), rows(case, "intercept", b[1]),
        rows(case, "fitted", tight$fitted.values), rows(case, "eta", tight$linear.predictors),
        rows(case, "deviance", tight$deviance), rows(case, "null_deviance", tight$null.deviance),
        rows(case, "dispersion", dispersion), rows(case, "beta", beta), rows(case, "beta_se", beta_se),
        rows(case, "predict_link", eta_new), rows(case, "predict_se", se_new),
        rows(case, "predict_response", spec$family$linkinv(eta_new)),
        rows(case, "AIC", aic), rows(case, "BIC", aic - 2 * k + log(n) * k))
    if (spec$scalar) out <- rbind(out, rows(case, "scalar_coefficient", b["z"]))
}
# Leading-count GLM AIC and BIC (R's AIC()/BIC() for glm) for the binomial case.
y <- cases$binomial$y
stopifnot(max(abs(fdata2basis(fd, basis)$coefs - designs$binomial[, 2:4])) < 1e-12)
for (l in 1:3) {
    X <- cbind(1, fdata2basis(fd, basis)$coefs[, 1:l, drop = FALSE])
    tight <- glm.fit(X, y, family = binomial(), control = tight_control)
    out <- rbind(out, transform(rows("selection", "AIC", tight$aic), index = l),
                 transform(rows("selection", "BIC", tight$aic + (log(n) - 2) * ncol(X)), index = l))
}
path <- "test/data/m7g_fdausc.csv"
writeLines(c(header, "# case,quantity,index,value"), path)
out$value <- sprintf("%.17g", out$value)
write.table(out, path, append = TRUE, sep = ",", row.names = FALSE, col.names = FALSE, quote = FALSE)
