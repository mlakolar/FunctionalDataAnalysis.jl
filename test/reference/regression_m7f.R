# Run from the repository root: Rscript test/reference/regression_m7f.R
# References: R 4.5.1; refund 0.1.40; mgcv 1.9-3.
# Sources audited (printed from the installed namespaces):
#   refund pffr, pffr_prepare, pffr_process_ff_terms, pffr_configure_algorithm,
#   ff, compute_integration_weights, trace_lv, coef.pffr (arguments);
#   mgcv smoothCon (matrix arguments, by-variable constraints, S.scale),
#   smooth.construct.tensor.smooth.spec (np reparameterization, penalty
#   normalization), smooth.construct.ps.smooth.spec.
# Penalized function-on-function regression on dense grids: one functional
# covariate, then a functional covariate plus a scalar covariate, then the
# first model with an irregular response (pffr's ydata). CI reads these small
# CSVs and does not need R.
suppressMessages({
    library(refund)
    library(mgcv)
})
stopifnot(packageVersion("refund") == "0.1.40", packageVersion("mgcv") == "1.9.3")
options(digits = 17)
header <- paste("#", R.version.string, "; refund", packageVersion("refund"),
                "; mgcv", packageVersion("mgcv"), "; regression_m7f.R")
write_ref <- function(x, name, description) {
    path <- paste0("test/data/m7f_", name, ".csv")
    writeLines(c(header, paste("#", description)), path)
    x <- as.data.frame(x)
    x[] <- lapply(x, function(column) if (is.numeric(column)) sprintf("%.17g", column) else column)
    write.table(x, path, append = TRUE, sep = ",", row.names = FALSE,
                col.names = FALSE, quote = FALSE)
}
rows <- function(quantity, value, component = 0L) {
    data.frame(quantity, index = seq_along(value), component, value = as.vector(value))
}
matrix_rows <- function(quantity, value) {
    value <- as.matrix(value)
    do.call(rbind, lapply(seq_len(ncol(value)), function(j) rows(quantity, value[, j], j)))
}

## Inputs: 40 curves; covariate on 25 points of [0, 1], response on 21 points
## of [0, 1]. Covariate curves have four smooth modes plus rough noise;
## residual curves are correlated within subjects.
n <- 40L; Jx <- 25L; Jy <- 21L
t <- seq(0, 1, length = Jx)
s <- seq(0, 1, length = Jy)
i <- seq_len(n)
X <- outer(cos(1.3 * i), sin(pi * t)) + outer(sin(2.1 * i), cos(pi * t)) * .8 +
    outer(cos(0.7 * i + 1), sin(2 * pi * t)) * .6 + outer(sin(3.7 * i), t^2) * .5 +
    .1 * sin(outer(i, seq_len(Jx), function(a, b) a * 1.9 + b * 2.3 + a * b * .29))
z <- cos(2.9 * i)
beta_true <- outer(t, s, function(tt, ss) cos(pi * (tt - ss)) * (1 + ss))
noise <- outer(cos(1.1 * i), sin(pi * s)) * .3 + outer(sin(2.7 * i), cos(2 * pi * s)) * .2 +
    .1 * sin(outer(i, seq_len(Jy), function(a, b) a * 1.7 + b * 3.1 + a * b * .41))
quad <- c(diff(t), 0) / 2 + c(0, diff(t)) / 2
signal <- (X * matrix(quad, n, Jx, byrow = TRUE)) %*% beta_true
Y1 <- outer(rep(1, n), 1 + sin(2 * pi * s)) + signal + noise
Y2 <- Y1 + outer(z, 1 - 2 * s^2)
write_ref(cbind(z, X), "input_covariates",
          "columns: scalar covariate z, then the 25 covariate values on seq(0, 1, length = 25)")
write_ref(cbind(Y1, Y2), "input_responses",
          "columns: 21 values of response 1 (functional term only), then 21 values of response 2 (plus z)")

# Evaluation grids for surfaces and coefficient functions (subgrids of t and s).
tt <- t[seq(1, Jx, by = 3)]   # 9 points
ss <- s[seq(1, Jy, by = 2)]   # 11 points

# Prediction inputs: three new covariate curves and scalars.
inew <- c(41.5, 43.25, 47)
Xnew <- outer(cos(1.3 * inew), sin(pi * t)) + outer(sin(2.1 * inew), cos(pi * t)) * .8 +
    outer(cos(0.7 * inew + 1), sin(2 * pi * t)) * .6
znew <- c(.4, -.9, 1.3)

# Coefficient maps: rows of the model matrix that give beta(t, s) on (tt, ss)
# (covariate grid varying slowest) and the intercept/scalar functions on ss.
surface_map <- function(m, label) {
    sm <- m$smooth[[label]]
    nd <- list()
    grid <- expand.grid(s = ss, t = tt)
    nd[[sm$margin[[1]]$term]] <- grid$t
    nd[[sm$margin[[2]]$term]] <- grid$s
    nd[[sm$by]] <- rep(1, nrow(grid))
    out <- matrix(0, nrow(grid), length(m$coefficients))
    out[, sm$first.para:sm$last.para] <- PredictMat(sm, as.data.frame(nd))
    out
}
function_map <- function(m, label, intercept = FALSE) {
    sm <- m$smooth[[label]]
    nd <- data.frame(ss, 1)
    names(nd) <- c(sm$term, if (sm$by == "NA") "dummy" else sm$by)
    out <- matrix(0, length(ss), length(m$coefficients))
    out[, sm$first.para:sm$last.para] <- PredictMat(sm, nd)
    if (intercept) out[, 1] <- 1
    out
}
se_of <- function(L, V) sqrt(rowSums((L %*% V) * L))
term_edf <- function(m, label, intercept = FALSE) {
    sm <- m$smooth[[label]]
    sum(m$edf[sm$first.para:sm$last.para]) + intercept * m$edf[1]
}
lambda_raw <- function(m) {
    # Smoothing parameters in the units of the unscaled penalties (te penalties
    # after mgcv's marginal reparameterization and normalization).
    unlist(lapply(m$smooth, function(sm) rep(1, length(sm$S)) / sm$S.scale)) * m$sp
}
labels_of <- function(m) sapply(m$smooth, function(sm) sm$label)

## Model 1: Y1 ~ ff(X) with trapezoidal integration (the package default).
d1 <- list(Y = Y1, X = X)
fit1 <- function(sandwich) pffr(Y ~ ff(X, xind = t, integration = "trapezoidal"), yind = s,
                                data = d1, sandwich = sandwich)
m1 <- fit1("none")
m1c <- fit1("cluster")
stopifnot(isTRUE(all.equal(m1$coefficients, m1c$coefficients)))
lab1 <- labels_of(m1)
print(lab1)
stopifnot(sapply(m1$smooth, function(sm) sm$last.para - sm$first.para + 1) == c(19, 25))  # centered intercept smooth
Ls1 <- surface_map(m1, lab1[2])
Li1 <- function_map(m1, lab1[1], intercept = TRUE)
pred1 <- predict(m1, newdata = list(X = Xnew))
# The REML score at a second, fixed sp: the package's REML differs from mgcv's
# by a lambda-free constant (other coordinates for the penalties' null space),
# so differences between two smoothing parameters must agree.
m1d <- pffr(Y ~ ff(X, xind = t, integration = "trapezoidal"), yind = s, data = d1,
            sandwich = "none", sp = 2 * m1$sp)
write_ref(rbind(
    rows("lambda", lambda_raw(m1)),
    rows("sp", m1$sp),
    rows("reml", m1$gcv.ubre),
    rows("reml_double", m1d$gcv.ubre),
    rows("sigma2", m1$sig2),
    rows("edf", c(term_edf(m1, lab1[1], TRUE), term_edf(m1, lab1[2]))),
    rows("edf_total", sum(m1$edf)),
    rows("surface_grid", tt),
    rows("response_grid", ss),
    rows("beta", Ls1 %*% m1$coefficients),
    rows("se_bayesian", se_of(Ls1, m1$Vp)),
    rows("se_frequentist", se_of(Ls1, m1$Ve)),
    rows("se_cluster", se_of(Ls1, m1c$Vp)),
    rows("intercept", Li1 %*% m1$coefficients),
    rows("intercept_se_bayesian", se_of(Li1, m1$Vp)),
    rows("intercept_se_cluster", se_of(Li1, m1c$Vp)),
    matrix_rows("fitted", t(fitted(m1))),
    matrix_rows("new_covariates", t(Xnew)),
    matrix_rows("prediction", t(pred1))),
    "pffr_ff", "pffr(Y1 ~ ff(X, integration='trapezoidal')), REML, default bases; sandwich none and cluster")

## Model 2: Y2 ~ z + ff(X) with refund's defaults (integration = "simpson").
d2 <- list(Y = Y2, X = X, z = z)
fit2 <- function(sandwich) pffr(Y ~ z + ff(X, xind = t), yind = s, data = d2, sandwich = sandwich)
m2 <- fit2("none")
m2c <- fit2("cluster")
stopifnot(isTRUE(all.equal(m2$coefficients, m2c$coefficients)))
lab2 <- labels_of(m2)
print(lab2)
stopifnot(sapply(m2$smooth, function(sm) sm$last.para - sm$first.para + 1) == c(19, 5, 25))
Ls2 <- surface_map(m2, lab2[3])
Li2 <- function_map(m2, lab2[1], intercept = TRUE)
Lz2 <- function_map(m2, lab2[2])
pred2 <- predict(m2, newdata = list(X = Xnew, z = znew))
m2d <- pffr(Y ~ z + ff(X, xind = t), yind = s, data = d2, sandwich = "none", sp = 2 * m2$sp)
write_ref(rbind(
    rows("lambda", lambda_raw(m2)),
    rows("sp", m2$sp),
    rows("reml", m2$gcv.ubre),
    rows("reml_double", m2d$gcv.ubre),
    rows("sigma2", m2$sig2),
    rows("edf", c(term_edf(m2, lab2[1], TRUE), term_edf(m2, lab2[2]), term_edf(m2, lab2[3]))),
    rows("edf_total", sum(m2$edf)),
    rows("beta", Ls2 %*% m2$coefficients),
    rows("se_bayesian", se_of(Ls2, m2$Vp)),
    rows("se_cluster", se_of(Ls2, m2c$Vp)),
    rows("intercept", Li2 %*% m2$coefficients),
    rows("scalar", Lz2 %*% m2$coefficients),
    rows("scalar_se_bayesian", se_of(Lz2, m2$Vp)),
    rows("scalar_se_cluster", se_of(Lz2, m2c$Vp)),
    matrix_rows("fitted", t(fitted(m2))),
    rows("new_scalar", znew),
    matrix_rows("prediction", t(pred2))),
    "pffr_ff_scalar", "pffr(Y2 ~ z + ff(X)), REML, default bases and Simpson weights; sandwich none and cluster")

## Model 3: model 1 with an irregular response (ydata). Curve i keeps the
## points with (i + j) %% 3 != 0; both ends of the grid stay observed for some
## curve, so the yindex range is [0, 1].
keep <- outer(i, seq_len(Jy), function(a, b) (a + b) %% 3 != 0)
ydata <- data.frame(.obs = rep(i, Jy)[keep], .index = rep(s, each = n)[keep], .value = Y1[keep])
ydata <- ydata[order(ydata$.obs, ydata$.index), ]
d3 <- data.frame(id = i); d3$X <- X   # pffr matches ydata$.obs to rownames(data)
m3 <- pffr(Y ~ ff(X, xind = t, integration = "trapezoidal"), ydata = ydata, data = d3,
           sandwich = "none")
lab3 <- labels_of(m3)
stopifnot(max(abs(as.vector(m3$y) - ydata$.value)) == 0)   # rows in ydata order
Ls3 <- surface_map(m3, lab3[2])
write_ref(rbind(
    rows("lambda", lambda_raw(m3)),
    rows("reml", m3$gcv.ubre),
    rows("sigma2", m3$sig2),
    rows("edf_total", sum(m3$edf)),
    rows("beta", Ls3 %*% m3$coefficients),
    rows("se_bayesian", se_of(Ls3, m3$Vp)),
    rows("fitted", as.vector(m3$fitted.values))),
    "pffr_ff_sparse", "pffr(Y ~ ff(X, integration='trapezoidal')) with ydata (points with (i + j) %% 3 != 0), REML, sandwich none")
