# Run from the repository root (inputs from test/reference/mgcv_m7_inputs.jl):
#     Rscript test/reference/mgcv_m7g.R
# Audited reference: mgcv 1.9-3 gam(family=binomial/poisson/gaussian(link="log"),
# method="REML") and method="GCV.Cp" (UBRE for known scale, GCV otherwise),
# with paraPen penalties built from the package's own bases (m7_pls_input.csv).
# Sources audited: gam.fit3 (penalized IRLS, Newton weights for non-canonical
# links), gam.fit3.post.proc (Vp, edf), fix.family.ls, initial.spg. R 4.5.1.
# Tight IRLS and outer tolerances below make R's own coefficients accurate to
# well below the 1e-8 comparison tolerance. The RNG seed is consumed only by
# R; Julia reads the realized responses. CI consumes the committed CSV only.
library(mgcv)
stopifnot(packageVersion("mgcv") == package_version("1.9-3"))
options(digits=17)
header <- paste("#", R.version.string, "; mgcv", packageVersion("mgcv"))

raw <- read.csv("test/data/m7_pls_input.csv", comment.char="#", header=FALSE,
                col.names=c("quantity", "row", "column", "value"))
input <- function(name) {
    dims <- raw[raw$quantity == paste0("dim_", name), ]
    out <- matrix(0, dims$row, dims$column)
    entries <- raw[raw$quantity == name, ]
    out[cbind(entries$row, entries$column)] <- entries$value
    if (dims$column == 1) drop(out) else out
}
x1 <- input("x1"); x2 <- input("x2"); z <- input("z")
B <- input("spline_design"); SB <- input("spline_penalty")
Fd <- input("fourier_design"); SF <- input("fourier_penalty")
n <- length(x1)

set.seed(7207)
eta_single <- 0.4 * z + sin(2 * pi * x1)
eta_disjoint <- 0.3 + 0.8 * sin(2 * pi * x1) + 0.5 * cos(4 * pi * x2)
trials <- sample(1:6, n, replace=TRUE)
cases <- list(
    binomial_single=list(family=binomial(), design="single", weights=rep(1, n),
        y=rbinom(n, 1, plogis(eta_single))),
    binomial_trials=list(family=binomial(), design="disjoint", weights=trials,
        y=rbinom(n, trials, plogis(eta_disjoint)) / trials),
    probit_single=list(family=binomial(link="probit"), design="single", weights=rep(1, n),
        y=rbinom(n, 1, pnorm(1.5 * eta_single))),
    poisson_single=list(family=poisson(), design="single", weights=rep(1, n),
        y=rpois(n, exp(0.5 + 0.6 * eta_single))),
    poisson_disjoint=list(family=poisson(), design="disjoint", weights=rep(1, n),
        y=rpois(n, exp(0.5 + 0.6 * eta_disjoint))),
    gaussian_log=list(family=gaussian(link="log"), design="single", weights=rep(1, n),
        y=exp(1 + 0.5 * eta_single) + 0.3 * rnorm(n)))
stopifnot(all(cases$gaussian_log$y > 0))
control <- gam.control(epsilon=1e-12, newton=list(conv.tol=1e-10))

fit_case <- function(case, method) {
    spec <- cases[[case]]
    if (spec$design == "single") {
        d <- list(y=spec$y, Z=cbind(z), B=B); form <- y ~ Z + B - 1; pen <- list(B=list(SB))
    } else {
        d <- list(y=spec$y, B=B, Fd=Fd); form <- y ~ B + Fd - 1; pen <- list(B=list(SB), Fd=list(SF))
    }
    d$w <- spec$weights
    gam(form, data=d, paraPen=pen, weights=w, family=spec$family, method=method, control=control)
}

rows <- function(case, method, q, x) {
    x <- as.matrix(x)
    data.frame(case=case, method=method, quantity=q,
               row=rep(seq_len(nrow(x)), ncol(x)),
               column=rep(seq_len(ncol(x)), each=nrow(x)), value=as.vector(x))
}
upper <- function(case, method, q, x) {
    keep <- upper.tri(x, diag=TRUE)
    data.frame(case=case, method=method, quantity=q, row=row(x)[keep],
               column=col(x)[keep], value=x[keep])
}

output <- list()
for (case in names(cases)) {
    output[[length(output) + 1]] <- rbind(rows(case, "input", "y", cases[[case]]$y),
        rows(case, "input", "weights", cases[[case]]$weights))
    for (method in c("REML", "GCV.Cp")) {
        fit <- fit_case(case, method)
        stopifnot(fit$converged, fit$outer.info$conv == "full convergence")
        cat(case, method, "sp", fit$sp, "score", fit$gcv.ubre, "edf", sum(fit$edf),
            "scale", fit$sig2, "\n")
        out <- rbind(
            rows(case, method, "lambda", fit$sp), rows(case, method, "sig2", fit$sig2),
            rows(case, method, "score", fit$gcv.ubre), rows(case, method, "coef", coef(fit)),
            rows(case, method, "fitted", fit$fitted.values),
            rows(case, method, "eta", fit$linear.predictors),
            rows(case, method, "edf", fit$edf), rows(case, method, "deviance", fit$deviance),
            upper(case, method, "Vp", fit$Vp))
        if (!is.null(fit$reml.scale) && method == "REML")
            out <- rbind(out, rows(case, method, "reml_scale", fit$reml.scale))
        output[[length(output) + 1]] <- out
    }
}
path <- "test/data/m7g_mgcv.csv"
writeLines(c(header, "# case,method,quantity,row,column,value; Vp upper triangle; lambda multiplies the penalty"), path)
output <- do.call(rbind, output)
# write.table keeps only 15 significant digits; 17 round-trip doubles exactly.
output$value <- sprintf("%.17g", output$value)
write.table(output, path, append=TRUE, sep=",", row.names=FALSE, col.names=FALSE, quote=FALSE)
