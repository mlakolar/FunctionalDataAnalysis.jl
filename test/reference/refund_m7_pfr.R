# Run from the repository root:
#     Rscript test/reference/refund_m7_pfr.R
# Audited reference: refund 0.1-40 pfr() with lf() terms, fitted by mgcv 1.9-3
# gam(method="REML") (and "GCV.Cp" for one case). R 4.5.1. CI consumes the
# committed CSV only; the RNG seed below is consumed only by R.
suppressMessages(library(refund))
suppressMessages(library(mgcv))
stopifnot(packageVersion("refund") == package_version("0.1-40"))
stopifnot(packageVersion("mgcv") == package_version("1.9-3"))
header <- paste("#", R.version.string, "; refund", packageVersion("refund"),
                "; mgcv", packageVersion("mgcv"))

set.seed(7204)
n <- 50; nnew <- 6; N <- n + nnew
# Grids that are not symmetric about their mean make mgcv's thin plate basis
# deterministic (see the Regression documentation); g3 is refund's uniform case.
g1 <- seq(0, 1, length = 25)^1.5
g2 <- 2 * seq(0, 1, length = 31)^0.7
g3 <- seq(0, 1, length = 21)
curves <- function(g) {
    u <- g / max(g)
    basis <- rbind(1 + 0 * u, u, sin(pi * u), cos(pi * u), sin(2 * pi * u),
                   cos(2 * pi * u), sin(3 * pi * u), cos(3 * pi * u))
    matrix(rnorm(N * 8), N, 8) %*% diag(1 / sqrt(1:8)) %*% basis +
        matrix(rnorm(N * length(g), sd = 0.1), N)
}
X1 <- curves(g1); X2 <- curves(g2); X3 <- curves(g3)
X4 <- X1 + matrix(rnorm(N * length(g1), sd = 0.5), N)
z <- rnorm(N)
trap <- function(g) { d <- diff(g); 0.5 * c(d[1], d[-1] + d[-length(d)], d[length(d)]) }
b1 <- sin(2 * pi * g1); b2 <- cos(pi * g2 / 2) * g2; b3 <- sin(2 * pi * g3) + g3
e <- rnorm(N)
ys <- list(
    one = 1 + 0.5 * z + drop(X1 %*% (trap(g1) * b1)) + 0.3 * e,
    two = 1 + 0.5 * z + drop(X1 %*% (trap(g1) * b1)) + drop(X2 %*% (trap(g2) * b2)) + 0.3 * e,
    simpson = 1 + 0.5 * z + drop(X3 %*% (trap(g3) * b3)) + 0.3 * e,
    presmooth = 1 + 0.5 * z + drop(X1 %*% (trap(g1) * b1)) + 0.3 * e)
train <- 1:n; test <- (n + 1):N

out <- data.frame(quantity = character(), row = integer(), column = integer(), value = numeric())
add <- function(q, x) {
    if (is.null(x)) return(invisible())
    x <- as.matrix(x)
    out <<- rbind(out, data.frame(quantity = q, row = as.integer(row(x)),
                                  column = as.integer(col(x)), value = as.vector(x)))
}
add("grid1", g1); add("grid2", g2); add("grid3", g3)
add("X1", X1); add("X2", X2); add("X3", X3); add("X4", X4); add("z", z)
for (case in names(ys)) add(paste0("y_", case), ys[[case]])

record <- function(case, fit, newdata, nterms) {
    add(paste0(case, "_sp"), fit$sp)
    add(paste0(case, "_S_scale"), sapply(fit$smooth, function(s) s$S.scale))
    add(paste0(case, "_parametric"), fit$coefficients[1:(fit$smooth[[1]]$first.para - 1)])
    add(paste0(case, "_fitted"), fit$fitted.values)
    add(paste0(case, "_sigma2"), fit$sig2)
    add(paste0(case, "_score"), as.numeric(fit$gcv.ubre))
    add(paste0(case, "_edf"), sapply(fit$smooth, function(s) sum(fit$edf[s$first.para:s$last.para])))
    add(paste0(case, "_edf_total"), sum(fit$edf))
    if (!is.null(fit$reml.scale)) add(paste0(case, "_reml_scale"), fit$reml.scale)
    add(paste0(case, "_hessian"), fit$outer.info$hess)
    for (j in 1:nterms) {
        cv <- coef(fit, select = j)
        cp <- coef(fit, select = j, useVc = FALSE)
        add(paste0(case, "_beta_grid", j), cv[[1]])
        add(paste0(case, "_beta", j), cv$value)
        add(paste0(case, "_se", j), cv$se)
        add(paste0(case, "_se_vp", j), cp$se)
    }
    p <- predict(fit, newdata = newdata, se.fit = TRUE)
    add(paste0(case, "_predict"), as.vector(p$fit))
    add(paste0(case, "_predict_se"), as.vector(p$se.fit))
}

y <- ys$one[train]; Xa <- X1[train, ]; zz <- z[train]
fit <- pfr(y ~ zz + lf(Xa, argvals = g1, integration = "trapezoidal"), method = "REML")
record("one", fit, list(Xa = X1[test, ], zz = z[test]), 1)
fit <- pfr(y ~ zz + lf(Xa, argvals = g1, integration = "trapezoidal"), method = "GCV.Cp")
record("gcv", fit, list(Xa = X1[test, ], zz = z[test]), 1)

y <- ys$two[train]; Xb <- X2[train, ]
fit <- pfr(y ~ zz + lf(Xa, argvals = g1, integration = "trapezoidal") +
               lf(Xb, argvals = g2, integration = "trapezoidal"), method = "REML")
record("two", fit, list(Xa = X1[test, ], Xb = X2[test, ], zz = z[test]), 2)

# refund defaults: integration = "simpson", uniform grid.
y <- ys$simpson[train]; Xc <- X3[train, ]
fit <- pfr(y ~ zz + lf(Xc, argvals = g3), method = "REML")
record("simpson", fit, list(Xc = X3[test, ], zz = z[test]), 1)

y <- ys$presmooth[train]; Xd <- X4[train, ]
fit <- pfr(y ~ zz + lf(Xd, argvals = g1, integration = "trapezoidal", presmooth = "bspline"),
           method = "REML")
record("presmooth", fit, list(Xd = X4[test, ], zz = z[test]), 1)

# mgcv's thin plate regression basis itself (no constraint), on g1 (k = 10),
# on 8 points of g2 (k = 5) and on 12 points with k = 12 (no eigen truncation).
basis_case <- function(name, x, k) {
    sm <- smoothCon(s(x, k = k), data.frame(x = x))[[1]]
    add(paste0("tp_", name, "_points"), x)
    add(paste0("tp_", name, "_design"), PredictMat(sm, data.frame(x = x)))
    add(paste0("tp_", name, "_penalty"), sm$S[[1]] * sm$S.scale)
    add(paste0("tp_", name, "_scale"), sm$S.scale)
}
basis_case("g1", g1, 10)
basis_case("small", g2[c(1, 3, 6, 10, 15, 21, 26, 31)], 5)
basis_case("knots", g1[seq(1, 23, by = 2)], 12)

path <- "test/data/m7_refund_pfr.csv"
writeLines(c(header, "# quantity,row,column,value (column-major matrices)"), path)
out$value <- sprintf("%.17g", out$value)
write.table(out, path, sep = ",", row.names = FALSE, col.names = FALSE, quote = FALSE, append = TRUE)
