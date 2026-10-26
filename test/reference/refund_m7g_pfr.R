# Run from the repository root (after test/reference/refund_m7_pfr.R, whose
# committed curves this script reuses):
#     Rscript test/reference/refund_m7g_pfr.R
# Audited reference: refund 0.1-40 pfr(family=) with lf() terms, which passes
# `family` and `control` through to mgcv 1.9-3 gam (penalized IRLS with
# Laplace-approximate REML, or UBRE for method="GCV.Cp" with known scale).
# coef.pfr's bands use mgcv's Vc (useVc=TRUE). predict.pfr declares
# type="response", but forwards only the arguments in match.call() to
# predict.gam, whose default is type="link": without an explicit type it
# returns the linear predictor. Both scales are requested explicitly here. R 4.5.1. Tight IRLS and outer tolerances make R's own fits
# accurate well below the 1e-8 comparison tolerance. The RNG seed is consumed
# only by R; Julia reads the realized responses. CI consumes the CSV only.
suppressMessages(library(refund))
suppressMessages(library(mgcv))
stopifnot(packageVersion("refund") == package_version("0.1-40"))
stopifnot(packageVersion("mgcv") == package_version("1.9-3"))
header <- paste("#", R.version.string, "; refund", packageVersion("refund"),
                "; mgcv", packageVersion("mgcv"))

stored <- read.csv("test/data/m7_refund_pfr.csv", comment.char = "#", header = FALSE,
                   col.names = c("quantity", "row", "column", "value"))
matrix_of <- function(q) {
    rows <- stored[stored$quantity == q, ]
    out <- matrix(0, max(rows$row), max(rows$column))
    out[cbind(rows$row, rows$column)] <- rows$value
    out
}
g1 <- drop(matrix_of("grid1")); X1 <- matrix_of("X1"); z <- drop(matrix_of("z"))
n <- 50; train <- 1:n; test <- (n + 1):nrow(X1)
trap <- function(g) { d <- diff(g); 0.5 * c(d[1], d[-1] + d[-length(d)], d[length(d)]) }
signal <- drop(X1 %*% (trap(g1) * sin(2 * pi * g1)))

set.seed(7208)
eta <- 0.3 + 0.5 * z + 3 * signal
ys <- list(
    binomial = rbinom(length(eta), 1, plogis(eta)),
    poisson = rpois(length(eta), exp(0.5 + 0.3 * z + 5 * signal)),
    gaussian_log = exp(1 + 0.4 * eta) + 0.25 * rnorm(length(eta)))
stopifnot(all(ys$gaussian_log > 0))
families <- list(binomial = binomial(), poisson = poisson(), gaussian_log = gaussian(link = "log"))
control <- gam.control(epsilon = 1e-10, newton = list(conv.tol = 1e-10))

out <- data.frame(quantity = character(), row = integer(), column = integer(), value = numeric())
add <- function(q, x) {
    if (is.null(x)) return(invisible())
    x <- as.matrix(x)
    out <<- rbind(out, data.frame(quantity = q, row = as.integer(row(x)),
                                  column = as.integer(col(x)), value = as.vector(x)))
}
for (case in names(ys)) add(paste0("y_", case), ys[[case]])

record <- function(case, fit, newdata) {
    # "step failed" means no further decrease at rounding level; the gradient
    # printed here is then far below mgcv's default conv.tol.
    stopifnot(fit$converged, fit$outer.info$conv %in% c("full convergence", "step failed"),
              max(abs(fit$outer.info$grad)) < 1e-6)
    cat(case, "sp", fit$sp, "score", fit$gcv.ubre, "edf", sum(fit$edf), "scale", fit$sig2,
        "conv", fit$outer.info$conv, "grad", fit$outer.info$grad, "\n")
    add(paste0(case, "_sp"), fit$sp)
    add(paste0(case, "_S_scale"), sapply(fit$smooth, function(s) s$S.scale))
    add(paste0(case, "_coefficients"), fit$coefficients)
    add(paste0(case, "_fitted"), fit$fitted.values)
    add(paste0(case, "_eta"), fit$linear.predictors)
    add(paste0(case, "_sigma2"), fit$sig2)
    add(paste0(case, "_score"), as.numeric(fit$gcv.ubre))
    add(paste0(case, "_edf_total"), sum(fit$edf))
    add(paste0(case, "_deviance"), fit$deviance)
    add(paste0(case, "_null_deviance"), fit$null.deviance)
    if (!is.null(fit$reml.scale) && fit$method == "REML") add(paste0(case, "_reml_scale"), fit$reml.scale)
    cv <- coef(fit, select = 1)
    cp <- coef(fit, select = 1, useVc = FALSE)
    add(paste0(case, "_beta_grid"), cv[[1]])
    add(paste0(case, "_beta"), cv$value)
    add(paste0(case, "_se"), cv$se)
    add(paste0(case, "_se_vp"), cp$se)
    p <- predict(fit, newdata = newdata, type = "link", se.fit = TRUE)
    add(paste0(case, "_predict_link"), as.vector(p$fit))
    add(paste0(case, "_predict_se"), as.vector(p$se.fit))
    add(paste0(case, "_predict_response"), as.vector(predict(fit, newdata = newdata, type = "response")))
    stopifnot(isTRUE(all.equal(as.vector(predict(fit, newdata = newdata)), as.vector(p$fit))))
}

Xa <- X1[train, ]; zz <- z[train]
newdata <- list(Xa = X1[test, ], zz = z[test])
for (case in names(ys)) {
    y <- ys[[case]][train]
    fit <- pfr(y ~ zz + lf(Xa, argvals = g1, integration = "trapezoidal"),
               family = families[[case]], method = "REML", control = control)
    record(case, fit, newdata)
}
y <- ys$binomial[train]
fit <- pfr(y ~ zz + lf(Xa, argvals = g1, integration = "trapezoidal"),
           family = binomial(), method = "GCV.Cp", control = control)
record("binomial_ubre", fit, newdata)

path <- "test/data/m7g_refund_pfr.csv"
writeLines(c(header, "# quantity,row,column,value (column-major matrices)"), path)
out$value <- sprintf("%.17g", out$value)
write.table(out, path, sep = ",", row.names = FALSE, col.names = FALSE, quote = FALSE, append = TRUE)
