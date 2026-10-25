# Run from the repository root: Rscript test/reference/regression_m7b.R
# References: R 4.5.1; fdapace 0.6.0; fda 6.3.0.
# Sources audited (printed from the installed namespaces):
#   fdapace FLM, GetBR2New, FLMCI, FCReg, demean, MvCov, uniCov,
#   GetCrCovYZ, GetRawCrCovFuncFunc, GetSmoothedMeanCurve, Lwls2D;
#   fda fRegress.fd, fRegress.double, fRegress.formula, fRegress.stderr,
#   fRegressArgCheck, fdPar, inprod, times.fd, sum.fd.
# fdapace 0.6.0 has no ConcurReg; its concurrent estimator is FCReg.
# CI reads these small CSVs and does not need R.
library(fdapace)
library(fda)
stopifnot(packageVersion("fdapace") == "0.6.0", packageVersion("fda") == "6.3.0")
options(digits = 17)
header <- paste("#", R.version.string, "; fdapace", packageVersion("fdapace"),
                "; fda", packageVersion("fda"), "; regression_m7b.R")
write_ref <- function(x, name, description) {
    path <- paste0("test/data/m7b_", name, ".csv")
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
read_input <- function(name) unname(as.matrix(read.csv(paste0("test/data/", name, ".csv"),
    comment.char = "#", header = FALSE)))

## Function-on-function FLM, dense. Reuse the milestone 7 dense covariates.
grid <- as.vector(read_input("m7_dense_grid"))
x1 <- read_input("m7_dense_input1")
x2 <- read_input("m7_dense_input2")
xtest1 <- read_input("m7_dense_heldout1")
xtest2 <- read_input("m7_dense_heldout2")
z <- read_input("m7_dense_response")[, 2]
ztest <- as.vector(read_input("m7_dense_heldout_scalar"))
n <- nrow(x1)
theta <- 2 * pi * seq_len(n) / n
a <- cbind(1.4 * cos(theta), .8 * cos(2 * theta), .4 * cos(3 * theta))
b <- cbind(1.1 * sin(theta), .6 * sin(2 * theta))
# Response scores on two sine response eigenfunctions vanishing at endpoints:
# fdapace's rectangular eigenstep and the package's trapezoidal one agree.
psi <- sqrt(2) * outer(grid, 1:2, function(s, k) sin(pi * s * k))
eta <- cbind(.9 * a[, 1] - .5 * a[, 2] + .3 * b[, 1] + .4 * z,
             -.3 * a[, 1] + .6 * a[, 3] + .5 * b[, 2] - .2 * z)
eta <- eta + cbind(.12 * sin(5 * theta), .07 * cos(7 * theta))
muY <- as.vector(psi %*% c(.5, -.25))
ymat <- sweep(eta %*% t(psi), 2, muY, "+")
write_ref(ymat, "fof_dense_response", "24 response curves on the 21-point dense grid")
as_fdapace <- function(x, t = grid) list(Ly = lapply(seq_len(nrow(x)), function(i) x[i, ]),
                                         Lt = rep(list(t), nrow(x)))
dense_options <- function(k) list(dataType = "Dense", methodMuCovEst = "cross-sectional",
    error = FALSE, methodXi = "IN", shrink = FALSE, nRegGrid = length(grid),
    maxK = k, methodSelectK = k, useBinnedData = "OFF", verbose = FALSE)
fof <- FLM(as_fdapace(ymat), list(as_fdapace(x1), as_fdapace(x2), z),
    XTest = list(as_fdapace(xtest1), as_fdapace(xtest2), ztest),
    optnsListY = dense_options(2L), optnsListX = list(dense_options(3L), dense_options(2L)))
default_options <- function(opt) { opt$methodSelectK <- NULL; opt$maxK <- NULL; opt }
fof_default <- FLM(as_fdapace(ymat), list(as_fdapace(x1), as_fdapace(x2), z),
    optnsListY = default_options(dense_options(2L)),
    optnsListX = lapply(list(dense_options(3L), dense_options(2L)), default_options))
stopifnot(identical(c(vapply(fof_default$estXi[1:2], ncol, 1L), length(fof_default$lambdaY)),
                    c(3L, 2L, 2L)))

# FLM returns no standard errors. The explicitly conditional multivariate OLS
# quantities below use the same score design: Cov(vec B) = SigmaE %x% (D'D)^-1.
conditional_fof <- function(flm, yscores, phis, new_scores, prefix) {
    design <- cbind(1, do.call(cbind, flm$estXi))
    p <- ncol(design)
    fit <- lm.fit(design, yscores)
    B <- fit$coefficients
    E <- fit$residuals
    df <- nrow(design) - p
    sigmaE <- crossprod(E) / df
    C <- solve(crossprod(design))
    crit <- qt(.975, df)
    phiY <- flm$phiY
    sY <- rowSums((phiY %*% sigmaE) * phiY)
    new_design <- cbind(1, do.call(cbind, new_scores))
    dnew <- rowSums((new_design %*% C) * new_design)
    pred_se <- sqrt(outer(dnew, sY))
    pred <- flm$yPred
    fitted_scores <- design %*% B
    centered <- sweep(yscores, 2, colMeans(yscores)) %*% t(phiY)
    resid <- (yscores - fitted_scores) %*% t(phiY)
    pointwise <- 1 - colSums(resid^2) / colSums(centered^2)
    stopifnot(max(abs(flm$yHat - sweep(fitted_scores %*% t(phiY), 2, flm$muY, "+"))) < 1e-10)
    # Interval endpoints are estimate -/+ critical_value*se; residuals are the
    # reconstructed response minus fitted. Only the inputs to those are stored.
    out <- rbind(rows("df_residual", df), rows("r2", flm$R2), rows("critical_value", crit),
        matrix_rows("fitted", flm$yHat), matrix_rows("prediction", pred),
        matrix_rows("prediction_se", pred_se), rows("response_score_variance", sY),
        rows("intercept_function", as.vector(B[1, ] %*% t(phiY)) + flm$muY),
        rows("pointwise_r2", pointwise), matrix_rows("score_coefficient", B),
        matrix_rows("residual_covariance", sigmaE), matrix_rows("response_scores", yscores))
    start <- 2L
    for (j in seq_along(phis)) {
        k <- ncol(phis[[j]])
        block <- C[start:(start + k - 1L), start:(start + k - 1L), drop = FALSE]
        sX <- rowSums((phis[[j]] %*% block) * phis[[j]])
        se <- sqrt(outer(sX, sY))
        out <- rbind(out, matrix_rows(paste0("beta", j), flm$betaList[[j]]),
            matrix_rows(paste0("beta", j, "_se"), se))
        start <- start + k
    }
    if (p >= start) out <- rbind(out, rows("scalar_coefficient", B[start:p, ]))
    out
}
yfpca <- FPCA(as_fdapace(ymat)$Ly, as_fdapace(ymat)$Lt, dense_options(2L))
write_ref(conditional_fof(fof, yfpca$xiEst, fof$phiX, fof$testXi),
    "fof_dense", "quantity,index,component,value; real FLM functional-response fit plus explicitly conditional OLS uncertainty")

## Function-on-function FLM, sparse covariate and sparse response.
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
ygrid <- mean_input[, 1]
etas <- cbind(.8 * latent[, 1] - .4 * latent[, 2], .5 * latent[, 2] + .6 * latent[, 3])
# The response is observed at the covariate's sparse times, with small noise.
sparse_y <- list(Lt = sparse_x$Lt, Ly = lapply(seq_len(sparse_n), function(i) {
    t <- sparse_x$Lt[[i]]
    1 - .5 * t + sqrt(2) * (etas[i, 1] * sin(pi * t) + etas[i, 2] * sin(2 * pi * t)) +
        .05 * sin(13 * i + 7 * t)
}))
ymean <- 1 - .5 * ygrid
ypsi <- sqrt(2) * outer(ygrid, 1:2, function(s, k) sin(pi * s * k))
ycov <- ypsi %*% diag(c(1.1, .45)) %*% t(ypsi)
ysigma2 <- .01
write_ref(do.call(rbind, lapply(seq_len(sparse_n), function(i)
    cbind(i, sparse_y$Lt[[i]], sparse_y$Ly[[i]]))), "fof_sparse_response",
    "columns: curve,time,value; sparse response at the m3 covariate times")
write_ref(cbind(ygrid, ymean), "fof_sparse_response_mean", "columns: time,value; supplied response mean")
write_ref(ycov, "fof_sparse_response_surface", "supplied response covariance on the mean grid")
sparse_options <- function(mu, cov, s2, k) list(dataType = "Sparse", error = TRUE, methodXi = "CE",
    methodMuCovEst = "smooth", nRegGrid = nrow(mean_input), maxK = 5L,
    methodSelectK = k, FVEthreshold = 1, useBinnedData = "OFF", methodRho = "vanilla",
    userMu = list(t = mean_input[, 1], mu = mu),
    userCov = list(t = mean_input[, 1], cov = cov),
    userSigma2 = s2, verbose = FALSE)
sparse_fof <- FLM(sparse_y, list(sparse_x), XTest = list(sparse_new),
    optnsListY = sparse_options(ymean, ycov, ysigma2, 2L),
    optnsListX = sparse_options(mean_input[, 2], covariance, sigma2, 3L))
sparse_yfpca <- FPCA(sparse_y$Ly, sparse_y$Lt, sparse_options(ymean, ycov, ysigma2, 2L))
write_ref(rbind(rows("response_noise", ysigma2),
                conditional_fof(sparse_fof, sparse_yfpca$xiEst, sparse_fof$phiX, sparse_fof$testXi)),
    "fof_sparse", "quantity,index,component,value; real sparse FLM functional-response fit plus conditional OLS uncertainty")

## Concurrent regression: fdapace 0.6.0 FCReg (successor of ConcurReg).
# Dense: 20 subjects on 11 points, two functional covariates and one scalar.
cgrid <- seq(0, 1, length.out = 11L)
cn <- 20L
u <- 2 * pi * seq_len(cn) / cn
cx1 <- outer(cos(u), sin(pi * cgrid)) + outer(.6 * sin(2 * u), cgrid) +
    matrix(1 + cgrid, cn, length(cgrid), byrow = TRUE)
cx2 <- outer(sin(u), cos(pi * cgrid)) + outer(.4 * cos(3 * u), cgrid^2) +
    matrix(.5 - .3 * cgrid, cn, length(cgrid), byrow = TRUE)
cz <- .3 * cos(2 * u) + .2 * sin(5 * u)
cbeta0 <- 1 + .5 * cgrid
cbeta1 <- 1.2 - .8 * cgrid
cbeta2 <- sin(pi * cgrid)
cgamma <- .5 + cgrid
cnoise <- .1 * outer(sin(7 * u), cos(3 * pi * cgrid)) + .05 * outer(cos(11 * u), sin(5 * cgrid))
cy <- matrix(cbeta0, cn, length(cgrid), byrow = TRUE) + sweep(cx1, 2, cbeta1, "*") +
    sweep(cx2, 2, cbeta2, "*") + outer(cz, cgamma) + cnoise
write_ref(cbind(cx1, cx2, cy, cz), "concurrent_dense_input",
    "columns: 11 x1 values, 11 x2 values, 11 response values, scalar z; grid seq(0,1,length=11)")
dense_vars <- list(X1 = as_fdapace(cx1, cgrid), X2 = as_fdapace(cx2, cgrid), Z = cz, Y = as_fdapace(cy, cgrid))
dense_cr <- FCReg(dense_vars, userBwMu = .15, userBwCov = .2, outGrid = cgrid, kern = "gauss",
    measurementError = TRUE)
concurrent_rows <- function(fit) rbind(matrix_rows("beta", t(fit$beta)),
    rows("beta0", fit$beta0), rows("r2", fit$R2), rows("grid", fit$outGrid))
write_ref(concurrent_rows(dense_cr), "concurrent_dense",
    "quantity,index,component,value; FCReg userBwMu=0.15 userBwCov=0.2 gauss measurementError=TRUE")
dense_cr_noerror <- FCReg(dense_vars[c("X1", "Y")], userBwMu = .15, userBwCov = .2, outGrid = cgrid,
    kern = "gauss", measurementError = FALSE)
write_ref(concurrent_rows(dense_cr_noerror), "concurrent_dense_noerror",
    "quantity,index,component,value; FCReg one covariate, measurementError=FALSE")

# Sparse: 30 subjects with 4 to 7 times; x1 and the response share times,
# x2 is observed at different times. Output grid inside the observed range.
sn <- 30L
pool <- seq(0, 1, by = .05)
times_of <- function(i, shift) {
    m <- 4L + (i %% 4L)
    sort(pool[unique(((i * c(3L, 7L, 11L, 13L, 17L, 19L, 5L)[seq_len(m)] + shift) %% length(pool)) + 1L)])
}
st <- lapply(seq_len(sn), times_of, shift = 0L)
st2 <- lapply(seq_len(sn), times_of, shift = 4L)
v <- 2 * pi * seq_len(sn) / sn
sx1 <- function(i, t) 1 + t + cos(v[i]) * sin(pi * t) + .5 * sin(2 * v[i]) * t
sx2 <- function(i, t) .5 - .3 * t + sin(v[i]) * cos(pi * t)
sy <- lapply(seq_len(sn), function(i) {
    t <- st[[i]]
    (1 + .5 * t) + (1.2 - .8 * t) * sx1(i, t) + sin(pi * t) * sx2(i, t) +
        .1 * sin(7 * i + 3 * t)
})
sparse_vars <- list(X1 = list(Ly = lapply(seq_len(sn), function(i) sx1(i, st[[i]])), Lt = st),
    X2 = list(Ly = lapply(seq_len(sn), function(i) sx2(i, st2[[i]])), Lt = st2),
    Y = list(Ly = sy, Lt = st))
long <- function(var, name) do.call(rbind, lapply(seq_len(sn), function(i)
    data.frame(variable = name, curve = i, time = var$Lt[[i]], value = var$Ly[[i]])))
sparse_long <- rbind(long(sparse_vars$X1, 1), long(sparse_vars$X2, 2), long(sparse_vars$Y, 3))
write_ref(sparse_long, "concurrent_sparse_input", "columns: variable (1=x1, 2=x2, 3=response),curve,time,value")
sgrid <- seq(.1, .9, by = .1)
sparse_cr <- FCReg(sparse_vars, userBwMu = .2, userBwCov = .25, outGrid = sgrid, kern = "gauss",
    measurementError = TRUE)
write_ref(concurrent_rows(sparse_cr), "concurrent_sparse",
    "quantity,index,component,value; FCReg userBwMu=0.2 userBwCov=0.25 gauss measurementError=TRUE")

## Function-on-scalar regression: fda 6.3.0 fRegress with fixed fdPar lambda.
fn <- 20L
w <- 2 * pi * seq_len(fn) / fn
z1 <- cos(w) + .3 * sin(3 * w)
z2 <- sin(2 * w) - .2 * cos(w)
# Golub-Welsch Gauss-Legendre rule on [0, 1]; exact for polynomials of degree <= 19.
gauss_penalty <- function(basis, nderiv, m = 10L) {
    j <- seq_len(m - 1L)
    J <- matrix(0, m, m)
    J[cbind(j, j + 1L)] <- J[cbind(j + 1L, j)] <- j / sqrt(4 * j^2 - 1)
    e <- eigen(J, symmetric = TRUE)
    t <- (e$values + 1) / 2
    w <- e$vectors[1, ]^2
    D <- eval.basis(t, basis, nderiv)
    crossprod(D * w, D)
}
fos_case <- function(basis, lambdas, Lfd, name, description) {
    k <- basis$nbasis
    j <- seq_len(k)
    b0 <- sin(j); b1 <- cos(2 * j) / 2; b2 <- (j - (k + 1) / 2) / k
    noise <- .1 * outer(sin(3 * seq_len(fn)), cos(j)) + .05 * outer(cos(5 * seq_len(fn)), sin(2 * j))
    ycoef <- outer(rep(1, fn), b0) + outer(z1, b1) + outer(z2, b2) + noise
    yfd <- fd(t(ycoef), basis)
    xfdlist <- list(const = rep(1, fn), z1 = z1, z2 = z2)
    betalist <- lapply(seq_along(lambdas), function(i)
        if (is.null(Lfd)) fdPar(basis, lambda = lambdas[i]) else fdPar(basis, Lfd, lambdas[i]))
    fit <- fRegress(yfd, xfdlist, betalist)
    q <- length(xfdlist)
    coefs <- do.call(cbind, lapply(fit$betaestlist, function(p) p$fd$coefs))
    # Exact inner products for the same normal equations. inprod(basis, basis)
    # passes Lfd objects, so it skips inprod.bspline and uses Romberg
    # integration; bsplinepen's piecewise-polynomial algorithm is exact for
    # bases with interior knots. Without interior knots bsplinepen substitutes
    # a monomial basis (its penalty does not match B-spline coefficients), while
    # Romberg integration of these degree-8 integrands is exact.
    # For that single piece, a 10-point Gauss-Legendre rule is exact instead.
    X <- cbind(1, z1, z2)
    nderiv <- if (is.null(Lfd)) max(0, norder(basis) - 2) else Lfd$nderiv
    no_knots <- length(basis$params) == 0
    G <- if (no_knots) gauss_penalty(basis, 0) else eval.penalty(basis, int2Lfd(0))
    pen <- if (no_knots) gauss_penalty(basis, nderiv) else eval.penalty(basis, int2Lfd(nderiv))
    Cexact <- kronecker(crossprod(X), G) + kronecker(diag(lambdas), pen)
    Dexact <- as.vector(G %*% t(ycoef) %*% X)
    exact <- solve(Cexact, Dexact)
    fitted_coef <- X %*% t(matrix(exact, k, q))
    E <- ycoef - fitted_coef
    varc <- crossprod(E) / (fn - q)
    Cinv <- solve(Cexact)
    middle <- kronecker(crossprod(X), G %*% varc %*% G)
    bvar <- Cinv %*% middle %*% Cinv
    native <- fRegress.stderr(fit, diag(k), varc)
    tplot <- seq(0, 1, length.out = 11L)
    Phi <- eval.basis(tplot, basis)
    beta_se <- sapply(seq_len(q), function(i) {
        idx <- (i - 1) * k + seq_len(k)
        sqrt(rowSums((Phi %*% bvar[idx, idx]) * Phi))
    })
    znew <- rbind(c(1, .4, -.3), c(1, -.8, .5))
    A <- function(zrow) do.call(cbind, lapply(seq_len(q), function(i) zrow[i] * Phi))
    pred_se <- t(sapply(seq_len(nrow(znew)), function(r) {
        Ar <- A(znew[r, ]); sqrt(rowSums((Ar %*% bvar) * Ar))
    }))
    pred <- znew %*% t(matrix(exact, k, q)) %*% t(Phi)
    resid_var <- rowSums((Phi %*% varc) * Phi)
    crit <- qt(.975, fn - q)
    write_ref(rbind(matrix_rows("response_coef", ycoef), rows("z1", z1), rows("z2", z2),
        rows("lambda", lambdas), matrix_rows("native_beta_coef", coefs),
        matrix_rows("native_fitted_coef", t(fit$yhatfdobj$coefs)),
        matrix_rows("beta_coef", matrix(exact, k, q)),
        matrix_rows("residual_covariance", varc), matrix_rows("coefficient_covariance", bvar),
        rows("grid", tplot), matrix_rows("beta_se", beta_se), matrix_rows("prediction", t(pred)),
        matrix_rows("prediction_se", t(pred_se)), rows("residual_variance", resid_var),
        matrix_rows("new_covariates", znew), rows("critical_value", crit)),
        name, description)
    c(native_vs_exact_coef = max(abs(coefs - matrix(exact, k, q))) / max(abs(exact)),
      native_vs_exact_bvar = max(abs(native$Bvar - bvar)) / max(abs(bvar)))
}
# fda's inprod uses Romberg integration (EPS = 1e-4). For a B-spline basis
# without interior knots (one polynomial piece of degree 4), five Romberg
# levels integrate the degree-8 integrands exactly, so the unpenalized
# fRegress itself is the golden value. With interior knots fRegress carries
# Romberg errors near 1e-6, so those fixtures compare the exact system above
# ("beta_coef"); "native_*" rows record fRegress's own output.
# fRegress.stderr uses a full-weight Riemann sum on 501 points; the package
# uses exact inner products (compared with "coefficient_covariance").
# fda 6.3.0 bsplinepen returns the monomial-basis penalty for a B-spline basis
# without interior knots, so penalized fRegress on such a basis ("native_*"
# rows of fos_polynomial_penalized) differs from the B-spline penalty.
poly_basis <- create.bspline.basis(c(0, 1), nbasis = 5L, norder = 5L)
d1 <- fos_case(poly_basis, c(0, 0, 0), NULL, "fos_polynomial",
    "quantity,index,component,value; fRegress lambda=0 polynomial B-spline order 5, no interior knots")
d2 <- fos_case(poly_basis, c(0, 1e-3, 1e-2), NULL, "fos_polynomial_penalized",
    "quantity,index,component,value; lambda=(0,1e-3,1e-2), default fdPar Lfd=norder-2=3; exact rows use the B-spline penalty, native rows fda's monomial penalty")
cubic_basis <- create.bspline.basis(c(0, 1), nbasis = 7L, norder = 4L)
d3 <- fos_case(cubic_basis, c(1e-2, 1e-2, 1e-2), int2Lfd(1), "fos_cubic",
    "quantity,index,component,value; fRegress cubic B-spline 7 basis, lambda=1e-2, Lfd=1; exact rows use bsplinepen")
print(rbind(d1, d2, d3))
cat(header, "\nGenerated FLM functional-response, FCReg and fRegress references.\n")
