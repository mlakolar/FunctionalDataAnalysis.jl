# Run from the repository root: Rscript test/reference/refund_m6.R
# Audited reference: refund 0.1-40 fpca.face and pspline.setting.
# R 4.5.1; stats 4.5.1; splines 4.5.1. CI consumes committed CSVs only.
library(refund)
library(splines)
stopifnot(packageVersion("refund") == package_version("0.1-40"))
options(digits=17)
header <- paste("#", R.version.string, "; refund", packageVersion("refund"))
write_ref <- function(x, file, description) {
    writeLines(c(header, paste("#", description)), file)
    write.table(x, file, append=TRUE, sep=",", row.names=FALSE,
                col.names=FALSE, quote=FALSE, na="NA")
}
dir.create("test/data", showWarnings=FALSE)

# Fixed RNG seed is consumed ONLY by R. Julia reads the realized observations.
set.seed(61025)
n <- 40L
J <- 50L
grid <- seq(0, 2, length.out=J)
phi <- cbind(rep(1/sqrt(2), J), cos(pi*grid), sin(pi*grid), cos(2*pi*grid))
xi <- sweep(matrix(rnorm(n*4), n, 4), 2, sqrt(c(4, 1, .4, .08)), "*")
Y <- sweep(xi %*% t(phi), 2, 1 + .25*grid + .2*sin(pi*grid), "+") +
    matrix(rnorm(n*J, sd=.35), n, J)
write_ref(cbind(grid, t(Y)), "test/data/m6_face_input.csv", "time followed by 40 subject values")
# Keep complete eigensystems and scores, but only five reconstructed curves.
# The first four cover every missing-data pattern and the last is complete.
fitted_curves <- c(1:4, n)

# Instrument only the return: all calculations below are refund's unchanged
# function body. Capture otherwise unreported lambda, GCV, and low-rank factors.
trace("fpca.face", where=asNamespace("refund"), print=FALSE,
      exit=quote(assign(".face_audit", list(lambda=lambda,
          gcv=if (exists("Gcv", inherits=FALSE)) Gcv else vapply(seq(-20,20,length.out=100), face_gcv, 0.0),
          Q=as.matrix(B %*% A0), YS=YS, completed=Y,
          iterations=iter.miss-1L, s=s, Sigma=Sigma), envir=.GlobalEnv)))

rows <- function(q, x) {
    x <- as.matrix(x)
    if (length(x) == 0) return(data.frame(quantity=character(), row=integer(), column=integer(), value=numeric()))
    data.frame(quantity=q, row=rep(seq_len(nrow(x)), ncol(x)),
               column=rep(seq_len(ncol(x)), each=nrow(x)), value=as.vector(x))
}
w <- c(diff(grid)[1]/2, (grid[3:J]-grid[1:(J-2)])/2, diff(grid)[J-1]/2)

# Independently evaluate the package's documented mean departure: cubic
# clamped basis with 35 intervals, integrated squared-second-derivative
# penalty, existing smoother's GCV over exp(seq(-20,20,100))*domain_width^3.
breaks <- seq(min(grid), max(grid), length.out=36)
clamped <- c(rep(breaks[1], 3), breaks, rep(tail(breaks, 1), 3))
X <- splineDesign(clamped, grid, ord=4, outer.ok=TRUE)
nodes <- unlist(lapply(seq_len(length(breaks)-1), function(i)
    (breaks[i]+breaks[i+1])/2 + c(-1, 1)*diff(breaks)[i]/(2*sqrt(3))))
qw <- rep(diff(breaks)/2, each=2)
X2 <- splineDesign(clamped, nodes, ord=4, derivs=rep(2, length(nodes)), outer.ok=TRUE)
P <- crossprod(X2, qw*X2)
mean_lambdas <- exp(seq(-20,20,length.out=100))*diff(range(grid))^3
rawmean <- colMeans(Y)
mean_fits <- lapply(mean_lambdas, function(lam) {
    A <- crossprod(X)+lam*P
    coefs <- solve(A, crossprod(X, rawmean))
    df <- sum(diag(solve(A, crossprod(X))))
    fitted <- as.vector(X %*% coefs)
    list(fitted=fitted, gcv=J*sum((rawmean-fitted)^2)/(J-df)^2)
})
mean_index <- which.min(vapply(mean_fits, function(x) x$gcv, 0.0))
package_mean <- mean_fits[[mean_index]]$fitted

for (case in c("default", "fixed", "missing", "package_mean")) {
    data <- Y
    if (case == "missing") {
        # Missing interiors and boundaries in a few curves; no RNG is used.
        data[1, 1:3] <- NA
        data[2, 17:25] <- NA
        data[3, 47:50] <- NA
        data[4, seq(5, 45, by=4)] <- NA
        write_ref(which(is.na(data), arr.ind=TRUE), "test/data/m6_face_missing_input.csv",
                  "subject,time indices of missing observations in m6_face_input.csv")
    }
    if (case == "package_mean") {
        fit <- refund::fpca.face(sweep(data, 2, package_mean), argvals=grid,
                                 center=FALSE, var=TRUE)
        fit$mu <- package_mean
        fit$Yhat <- sweep(fit$Yhat, 2, package_mean, "+")
    } else {
        fit <- refund::fpca.face(data, argvals=grid, var=TRUE,
                                 lambda=if (case == "fixed") 2.5 else NULL)
    }
    capture <- .face_audit
    # Euclidean reference outputs are retained too, exposing the scale
    # conversion and the difference from refund's published band variance.
    C <- tcrossprod(capture$Q %*% capture$YS)/n
    eig <- eigen(sqrt(w)*t(sqrt(w)*t(C)), symmetric=TRUE)
    positive <- which(eig$values > 1e-10*max(abs(eig$values)))
    values <- eig$values[positive]
    functions <- eig$vectors[, positive, drop=FALSE]/sqrt(w)
    K <- which(cumsum(values)/sum(values) >= .99)[1]
    functions <- functions[, seq_len(K), drop=FALSE]
    values_selected <- values[seq_len(K)]
    for (k in seq_len(K)) {
        align <- sum(functions[,k]*fit$mu)
        orient <- if (abs(align) <= 1e-12*sqrt(sum(functions[,k]^2))*sqrt(sum(fit$mu^2)))
            functions[which.max(abs(functions[,k])),k] else align
        if (orient < 0) functions[,k] <- -functions[,k]
    }
    precision <- crossprod(functions) + diag(fit$sigma2/values_selected, K)
    score <- t(solve(precision, crossprod(functions, t(capture$completed))))
    omega <- fit$sigma2*solve(precision)
    fitted <- sweep(score %*% t(functions), 2, fit$mu, "+")
    posterior <- diag(functions %*% omega %*% t(functions))
    # Also compare the two formulas on the SAME converted eigensystem.
    refund_variance <- diag(fit$sigma2*functions %*%
        solve(crossprod(functions)+fit$sigma2*diag(values_selected, K)) %*% t(functions))
    output <- rbind(rows("mean", fit$mu), rows("eigenvalue", values_selected),
        rows("all_eigenvalue", values), rows("eigenfunction", functions),
        rows("score", score), rows("sigma2", fit$sigma2), rows("lambda", capture$lambda),
        rows("gcv", capture$gcv), rows("fitted_curves", fitted_curves),
        rows("fitted", fitted[fitted_curves, , drop=FALSE]), rows("posterior_variance", posterior),
        # Observed centered entries are exactly input minus mean. Only the
        # imputed entries require a golden value (in R's column-major order).
        rows("refund_variance", refund_variance), rows("imputed", capture$completed[is.na(data)]),
        rows("iterations", capture$iterations), rows("spectrum", capture$s),
        rows("refund_eigenvalue", fit$evalues), rows("refund_eigenfunction", fit$efunctions),
        rows("refund_score", fit$scores),
        rows("refund_fitted", fit$Yhat[fitted_curves, , drop=FALSE]),
        rows("refund_diag_var", fit$diag.var[1,]))
    write_ref(output, paste0("test/data/m6_face_", case, ".csv"),
        "quantity,row,column,value; weighted operator conversion documented on FPCA page")
    cat(case, "lambda", capture$lambda, "sigma2", fit$sigma2, "K", K,
        "iterations", capture$iterations, "posterior/refund variance at first point",
        posterior[1], refund_variance[1], "actual refund diag.var", fit$diag.var[1,1], "\n")
}

# Full pspline.setting matrices and initializer outputs are independent
# references for boundary handling, coefficient penalty, and R's spar search.
seqknots <- seq(-3, 38, length.out=42)/35*diff(range(grid))+min(grid)
ps <- refund:::pspline.setting(grid, seqknots)
write_ref(rbind(rows("basis", as.matrix(ps$B)), rows("penalty", as.matrix(ps$P)),
                rows("knots", seqknots)), "test/data/m6_pspline.csv",
          "quantity,row,column,value; refund 0.1-40 pspline.setting")
untrace("fpca.face", where=asNamespace("refund"))
