# Recreate from the repository root: Rscript test/reference/fda_m6.R
# Audited fda 6.3.0: R/pca.fd.R, fdPar.R, cca.fd.R, geigen.R.
# R 4.5.1. CI consumes small committed outputs, without running R.
suppressPackageStartupMessages(library(fda))
stopifnot(packageVersion("fda") == package_version("6.3.0"))
options(digits=17)
data(growth)
data(CanadianWeather)
header <- paste("#", R.version.string, "; fda", packageVersion("fda"))
write_ref <- function(x, file, description) {
    writeLines(c(header, paste("#", description)), file)
    write.table(x, file, append=TRUE, sep=",", row.names=FALSE,
                col.names=FALSE, quote=FALSE)
}
rows <- function(name, x) {
    x <- as.matrix(x)
    data.frame(quantity=name, row=rep(seq_len(nrow(x)), ncol(x)),
               column=rep(seq_len(ncol(x)), each=nrow(x)), value=as.vector(x))
}
orient_pca <- function(fit, basis, grid) {
    G <- eval.penalty(basis, 0)
    coefs <- fit$harmonics$coefs
    q <- sqrt(colSums(coefs*(G %*% coefs)))
    H <- sweep(coefs, 2, q, "/")
    score <- sweep(fit$scores, 2, q, "/")
    values <- eval.basis(grid, basis) %*% H
    mu <- as.vector(eval.fd(grid, fit$meanfd))
    for (j in seq_len(ncol(H))) {
        phi <- values[,j]/max(abs(values[,j]))
        smu <- if (max(abs(mu)) == 0) mu else mu/max(abs(mu))
        align <- sum(phi*smu)
        near <- abs(align) <= 1e-12*sqrt(sum(phi^2))*sqrt(sum(smu^2))
        sign <- if (near) values[which.max(abs(values[,j])),j] else align
        if (sign < 0) {
            H[,j] <- -H[,j]
            score[,j] <- -score[,j]
        }
    }
    list(coefs=H, values=eval.basis(grid, basis) %*% H,
         scores=score,
         norms=q)
}
orient_cca <- function(fit, transform) {
    A <- fit$ccawtfd1$coefs
    B <- fit$ccawtfd2$coefs
    v1 <- fit$ccavar1
    v2 <- fit$ccavar2
    for (j in seq_len(ncol(A))) {
        rotated <- transform %*% A[,j]
        if (rotated[which.max(abs(rotated))] < 0) {
            A[,j] <- -A[,j]
            B[,j] <- -B[,j]
            v1[,j] <- -v1[,j]
            v2[,j] <- -v2[,j]
        }
    }
    list(A=A, B=B, variates1=v1, variates2=v2)
}

# pca.fd uses inprod's hard-coded EPS=1e-4 Romberg rule, even for
# matching bases (the spline fast-path checks is.integer on an Lfd object).
# Keep pca.fd's entire algorithm, but use exact matching-basis inner products
# and high-accuracy operator integrals, to compare the package's exact-Gram
# convention. Native eigenvalues are retained separately below.
pca_exact <- fda::pca.fd
pca_environment <- new.env(parent=environment(pca_exact))
pca_environment$inprod <- function(fdobj1, fdobj2, ...) {
    b1 <- if (inherits(fdobj1,"fd")) fdobj1$basis else fdobj1
    b2 <- if (inherits(fdobj2,"fd")) fdobj2$basis else fdobj2
    stopifnot(fda:::is.eqbasis(b1,b2))
    A <- if (inherits(fdobj1,"fd")) fdobj1$coefs else diag(b1$nbasis)
    B <- if (inherits(fdobj2,"fd")) fdobj2$coefs else diag(b2$nbasis)
    t(A) %*% eval.penalty(b1,0) %*% B
}
pca_environment$eval.penalty <- function(basisobj, Lfdobj=0) {
    operator <- int2Lfd(Lfdobj)
    if (basisobj$type == "fourier" && !fda:::is.integerLfd(operator)) {
        # Cached high-precision integral independently computed below.
        return(harmonic_penalty)
    }
    fda::eval.penalty(basisobj,Lfdobj)
}
environment(pca_exact) <- pca_environment

for (dataset in c("growth", "weather")) {
    if (dataset == "growth") {
        times <- growth$age
        Y <- growth$hgtf
        # Paired height and interval growth-rate records of the same 54 girls.
        other_times <- (head(times,-1)+tail(times,-1))/2
        Z <- (Y[-1,]-Y[-nrow(Y),])/diff(times)
        interval <- range(times)
    } else {
        times <- day.5
        other_times <- times
        Y <- CanadianWeather$dailyAv[,,"Temperature.C"]
        Z <- CanadianWeather$dailyAv[,,"log10precip"]
        interval <- c(0,365)
    }
    for (kind in c("bspline", "fourier")) {
        basis <- if (kind == "bspline") create.bspline.basis(interval, nbasis=5, norder=4) else
            create.fourier.basis(interval, nbasis=5)
        harmonic_penalty <- matrix(0,5,5)
        if (kind == "fourier") {
            L <- fdPar(basis)$Lfd
            for (j in 1:5) for (k in 1:5) {
                harmonic_penalty[j,k] <- integrate(function(t) {
                    values <- eval.basis(t,basis,L)
                    values[,j]*values[,k]
                }, interval[1],interval[2],rel.tol=1e-12,abs.tol=1e-14)$value
            }
        }
        # Inputs exported as small coefficient matrices, so Julia does not
        # need to reproduce a separate curve-smoothing step or download data.
        left <- smooth.basis(times, Y, fdPar(basis, 0, 0))$fd
        right <- smooth.basis(other_times, Z, fdPar(basis, 0, 0))$fd
        # fda uses sin(omega*t); Julia uses sin(omega*(t-lower)).
        # Orthogonal coefficient rotation preserves all inner products and
        # Euclidean coefficient normalizations.
        transform <- diag(5)
        if (kind == "fourier") for (k in 1:2) {
            delta <- 2*pi*k*interval[1]/diff(interval)
            transform[(2*k):(2*k+1),(2*k):(2*k+1)] <-
                matrix(c(cos(delta),sin(delta),-sin(delta),cos(delta)),2,2)
        }
        prefix <- paste0("test/data/m6_fda_", dataset, "_", kind)
        write_ref(t(transform %*% left$coefs), paste0(prefix,"_left.csv"), "subjects x 5 input coefficients")
        write_ref(t(transform %*% right$coefs), paste0(prefix,"_right.csv"), "subjects x 5 input coefficients")
        grid <- seq(interval[1], interval[2], length.out=101)
        # Time units strongly affect lambda; choose meaningful positive values
        # for each dataset. All three are user-specified, with no R optimizer.
        lambdas <- if (dataset == "growth") c(0,.1,10) else c(0,1e4,1e6)
        for (index in seq_along(lambdas)) {
            lam <- lambdas[index]
            native <- pca.fd(left, nharm=2, harmfdPar=fdPar(left, lambda=lam))
            pc <- pca_exact(left, nharm=2, harmfdPar=fdPar(left, lambda=lam))
            converted <- orient_pca(pc, basis, grid)
            cc <- cca.fd(left, right, ncan=2,
                         ccafdPar1=fdPar(basis, 2, lam),
                         ccafdPar2=fdPar(basis, 2, lam))
            cw <- orient_cca(cc, transform)
            G <- eval.penalty(basis, 0)
            output <- rbind(rows("lambda",lam), rows("grid",grid),
                rows("eigenvalues",pc$values), rows("native_eigenvalues",native$values), rows("varprop",pc$varprop),
                rows("harmonics", converted$values), rows("scores",converted$scores),
                rows("harmonic_norms",converted$norms),
                rows("correlations",cc$ccacorr[1:2]),
                rows("weights1",eval.basis(grid,basis) %*% cw$A),
                rows("weights2",eval.basis(grid,basis) %*% cw$B),
                rows("variates1",cw$variates1),
                rows("variates2",cw$variates2))
            write_ref(output, paste0(prefix,"_",index,".csv"),
                "quantity,row,column,value; exact-L2 PCA harmonics/scores; R coefficient-normalized CCA")
        }
        # Independent CV criteria via the R fits. FPCA: center once, then
        # orthogonal L2 projections onto m=1 and m=2 subspaces, not sum scores*H.
        C <- left$coefs-rowMeans(left$coefs)
        pc_cv <- cc_cv <- numeric(length(lambdas))
        for (c in seq_along(lambdas)) {
            x <- y <- numeric(ncol(C))
            for (i in seq_len(ncol(C))) {
                held <- fd(C[,-i], basis)
                pc <- pca_exact(held, 2, fdPar(held, lambda=lambdas[c]), centerfns=FALSE)
                for (m in 1:2) {
                    A <- pc$harmonics$coefs[,seq_len(m),drop=FALSE]
                    residual <- C[,i] - A %*% solve(t(A)%*%G%*%A, t(A)%*%G%*%C[,i])
                    pc_cv[c] <- pc_cv[c] + as.numeric(t(residual)%*%G%*%residual)
                }
                cc <- cca.fd(fd(left$coefs[,-i],basis), fd(right$coefs[,-i],basis),2,
                             fdPar(basis,2,lambdas[c]),fdPar(basis,2,lambdas[c]))
                cw <- orient_cca(cc, transform)
                cw$A <- cw$A[,1,drop=FALSE]
                cw$B <- cw$B[,1,drop=FALSE]
                # Fold-centered held-out variates, using R's coefficient norm.
                x[i] <- t(left$coefs[,i]-rowMeans(left$coefs[,-i]))%*%G%*%cw$A
                y[i] <- t(right$coefs[,i]-rowMeans(right$coefs[,-i]))%*%G%*%cw$B
            }
            cc_cv[c] <- cor(x,y)^2
        }
        write_ref(cbind(lambdas, pc_cv, cc_cv), paste0(prefix,"_cv.csv"),
                  "lambda, summed L2 PCA CV_m (m=1:2), CCA held-out leading squared correlation")
    }
}
