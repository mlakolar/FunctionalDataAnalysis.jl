# Run from the repository root after test/reference/mgcv_m7_inputs.jl:
#     Rscript test/reference/mgcv_m7.R
# Audited reference: mgcv 1.9-3 gam(method="REML") and gam(method="GCV.Cp")
# with paraPen penalties built from the package's own bases (m7_pls_input.csv).
# R 4.5.1. CI consumes committed CSVs only.
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
x1 <- input("x1"); x2 <- input("x2"); z <- input("z"); w <- input("w")
B <- input("spline_design"); SB <- input("spline_penalty")
Fd <- input("fourier_design"); SF <- input("fourier_penalty")
Xt <- input("tensor_design"); St1 <- input("tensor_penalty1"); St2 <- input("tensor_penalty2")
n <- length(x1)

# Fixed RNG seed is consumed ONLY by R. Julia reads the realized responses.
set.seed(7203)
e <- rnorm(n)
# The x1 effect of "bound" is linear, in the null space of its penalty. Seed 1
# is a draw for which both criteria send that smoothing parameter to the
# effective infinity (about 1e7 times the other cases' value).
set.seed(1)
e_bound <- rnorm(n)
responses <- list(
    single=z + sin(2*pi*x1) + .3*e,
    weighted=z + sin(2*pi*x1) + .3*e/sqrt(w),
    disjoint=sin(2*pi*x1) + .5*cos(4*pi*x2) + .3*e,
    tensor=sin(2*pi*x1)*cos(2*pi*x2) + .2*e,
    bound=1 + 2*x1 + .5*cos(4*pi*x2) + .3*e_bound,
    lower=z + sin(2*pi*x1) + .3*e)

fit_case <- function(case, method, min.sp=NULL) {
    y <- responses[[case]]
    if (case %in% c("single", "weighted", "lower")) {
        d <- list(y=y, Z=cbind(z), B=B)
        form <- y ~ Z + B - 1; pen <- list(B=list(SB))
    } else if (case %in% c("disjoint", "bound")) {
        d <- list(y=y, B=B, Fd=Fd)
        form <- y ~ B + Fd - 1; pen <- list(B=list(SB), Fd=list(SF))
    } else {
        d <- list(y=y, Xt=Xt)
        form <- y ~ Xt - 1; pen <- list(Xt=list(St1, St2))
    }
    weights <- if (case == "weighted") w else rep(1, n)
    gam(form, data=d, paraPen=pen, weights=weights, method=method, min.sp=min.sp)
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

output <- list(rows("all", "input", "weights_weighted", w))
for (case in names(responses)) {
    output[[length(output) + 1]] <- rows(case, "input", "y", responses[[case]])
    for (method in c("REML", "GCV.Cp")) {
        min.sp <- NULL
        if (case == "lower") {
            # A lower bound above the unconstrained optimum, so that it binds.
            min.sp <- 20*fit_case("single", method)$sp
            output[[length(output) + 1]] <- rows(case, method, "min_sp", min.sp)
        }
        fit <- fit_case(case, method, min.sp)
        # With min.sp, mgcv's sp excludes the bound: the penalty multiplier is
        # min.sp + sp (?gam.control / ?gam, argument min.sp).
        lambda <- fit$sp + if (is.null(min.sp)) 0 else min.sp
        cat(case, method, "lambda", lambda, "sig2", fit$sig2, "score", fit$gcv.ubre,
            "edf", sum(fit$edf), "conv", fit$outer.info$conv, "iter", fit$outer.info$iter, "\n")
        output[[length(output) + 1]] <- rbind(
            rows(case, method, "lambda", lambda), rows(case, method, "sig2", fit$sig2),
            rows(case, method, "score", fit$gcv.ubre), rows(case, method, "coef", coef(fit)),
            rows(case, method, "fitted", fit$fitted.values), rows(case, method, "edf", fit$edf),
            upper(case, method, "Vp", fit$Vp))
    }
}
writeLines(c(header, "# case,method,quantity,row,column,value; Vp upper triangle; lambda multiplies the penalty"),
           "test/data/m7_mgcv.csv")
output <- do.call(rbind, output)
# write.table keeps only 15 significant digits; 17 round-trip doubles exactly.
output$value <- sprintf("%.17g", output$value)
write.table(output, "test/data/m7_mgcv.csv", append=TRUE, sep=",",
            row.names=FALSE, col.names=FALSE, quote=FALSE)
