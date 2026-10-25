# Run from the repository root: Rscript test/reference/face_m6.R
# Audited sources: face 0.1-8 (face.sparse, pspline, predict.face.sparse,
# raw.construct, select.knots, pspline.setting); refund 0.1-40 CD4 inputs.
# R 4.5.1; Matrix 1.7-3; mgcv 1.9-3; splines 4.5.1.
# CI reads the small committed CSVs; it does not run R.
library(face)
library(splines)
stopifnot(packageVersion("face") == package_version("0.1-8"),
          packageVersion("refund") == package_version("0.1-40"))
options(digits=17)
header <- paste("#", R.version.string, "; face", packageVersion("face"),
                "; refund", packageVersion("refund"), "; mgcv", packageVersion("mgcv"))
write_ref <- function(x, file, description) {
    writeLines(c(header, paste("#", description)), file)
    write.table(x, file, append=TRUE, sep=",", row.names=FALSE,
                col.names=FALSE, quote=FALSE)
}
rows <- function(q, x) {
    x <- as.matrix(x)
    data.frame(quantity=q, row=rep(seq_len(nrow(x)), ncol(x)),
               column=rep(seq_len(ncol(x)), each=nrow(x)), value=as.vector(x))
}
read_fixture <- function(file) as.matrix(read.csv(file, header=FALSE, comment.char="#"))
sparse <- as.data.frame(read_fixture("test/data/m3_sparse_input.csv"))
names(sparse) <- c("subj", "argvals", "y")
# Identical export convention to data/build_datasets.R and load_dataset(:cd4):
# native months, positional subject IDs, every observed cell, no transformations.
data("cd4", package="refund")
cd4_long <- do.call(rbind, lapply(seq_len(nrow(cd4)), function(i) {
    keep <- which(!is.na(cd4[i,]))
    data.frame(subj=i, argvals=as.numeric(colnames(cd4))[keep], y=cd4[i,keep])
}))
stopifnot(nrow(cd4_long)==1888)
write_ref(cd4_long, "test/data/m6_sparse_face_cd4_input.csv",
          "columns: subject,time,value; all refund 0.1-40 cd4 observations, matching load_dataset(:cd4)")

# Trace at return only: capture otherwise unreported criterion curves and raw
# noise before truncation, without changing any reference calculations.
trace("face.sparse.inner", where=asNamespace("face"), print=FALSE,
    exit=quote(assign(".sparse_audit", list(
        gcv=vapply(seq(-3,10,length.out=14), gcv, 0.0),
        raw_sigma2=alpha[length(alpha)]), envir=.GlobalEnv)))
trace("pspline", where=asNamespace("face"), print=FALSE,
    exit=quote(assign(".mean_audit", list(
        gcv=vapply(seq(-20,20,length.out=100), pspline_gcv, 0.0)), envir=.GlobalEnv)))
for (dataset in c("sparse", "cd4")) {
    d <- if (dataset=="sparse") sparse else cd4_long
    grid <- seq(min(d$argvals), max(d$argvals), length.out=100)
    J <- length(grid)
    w <- c(diff(grid)[1]/2, (grid[3:J]-grid[1:(J-2)])/2, diff(grid)[J-1]/2)
    for (case in c("default", "fixed")) {
        f <- face::face.sparse(d, argvals.new=grid, calculate.scores=TRUE,
            lambda=if (case=="fixed") 2.5 else NULL,
            lambda_mean=if (case=="fixed") .75 else NULL)
        capture <- .sparse_audit
        mean_capture <- .mean_audit
        B <- splineDesign(f$knots, grid, ord=4, outer.ok=TRUE)
        # Weighted L2 conversion: diagonalize sqrt(W) Chat sqrt(W). R's
        # reported eigenvalues use B'B/J, so scalar conversion works only for
        # constant weights: lambda_L2=J*w*lambda_R, phi_L2=phi_R/sqrt(J*w).
        C <- B %*% f$Theta %*% t(B)
        E <- eigen(sqrt(w)*t(sqrt(w)*t(C)), symmetric=TRUE)
        keep <- which(E$values>1e-10*max(abs(E$values)))
        values <- E$values[keep]
        phi <- E$vectors[,keep,drop=FALSE]/sqrt(w)
        K <- which(cumsum(values)/sum(values)>=.99)[1]
        phi <- phi[,seq_len(K),drop=FALSE]
        for (k in seq_len(K)) {
            a <- sum(phi[,k]*f$mu.new)
            if (abs(a)<=1e-12*sqrt(sum(phi[,k]^2))*sqrt(sum(f$mu.new^2)))
                a <- phi[which.max(abs(phi[,k])),k]
            if (a<0) phi[,k] <- -phi[,k]
        }
        subjects <- if (dataset=="sparse") unique(d$subj) else unique(c(1:6,
            d$subj[which.min(ave(d$subj,d$subj,FUN=length))]))
        score <- matrix(0,length(subjects),K)
        omega <- matrix(0,K*K,length(subjects))
        fitted <- variance <- matrix(0,length(subjects),J)
        # predict.face.sparse uses the FULL PSD Theta in the observation solve.
        # Transform its coefficient random effects and their posterior covariance
        # to our weighted eigenbasis. The R prediction is also retained directly.
        refpred <- predict(f,d)
        reference_prediction <- reference_variance <- numeric()
        for (ii in seq_along(subjects)) {
            i <- subjects[ii]
            sel <- which(d$subj==i)
            Bi <- splineDesign(f$knots,d$argvals[sel],ord=4,outer.ok=TRUE)
            V <- Bi %*% f$Theta %*% t(Bi)+diag(f$sigma2,length(sel))
            cross <- (Bi %*% f$Theta %*% t(B)) %*% (w*phi)
            r <- d$y[sel]-as.vector(predict(f$fit_mean,d$argvals[sel]))
            score[ii,] <- as.vector(t(cross) %*% solve(V,r))
            post <- diag(values[seq_len(K)],K)-t(cross) %*% solve(V,cross)
            omega[,ii] <- as.vector(post)
            fitted[ii,] <- f$mu.new + as.vector(phi %*% score[ii,])
            variance[ii,] <- diag(phi %*% post %*% t(phi))
            reference_prediction <- c(reference_prediction, refpred$y.pred[sel])
            reference_variance <- c(reference_variance, refpred$se.pred[sel]^2)
        }
        # These full-covariance predictions agree with the source at its native
        # observation times; selected-K reconstruction is a different quantity.
        out <- rbind(rows("grid",grid), rows("mean",f$mu.new),
            rows("eigenvalue",values), rows("eigenfunction",phi),
            rows("sigma2",f$sigma2), rows("raw_sigma2",capture$raw_sigma2),
            rows("lambda",f$lambda), rows("lambda_mean",f$lambda_mean),
            rows("gcv",capture$gcv), rows("mean_gcv",mean_capture$gcv),
            rows("subjects",subjects), rows("score",score), rows("score_covariance",omega),
            rows("fitted",fitted[seq_len(min(3,length(subjects))),,drop=FALSE]),
            rows("variance",variance[seq_len(min(3,length(subjects))),,drop=FALSE]),
            rows("source_prediction",reference_prediction), rows("source_variance",reference_variance),
            rows("theta",f$Theta), rows("knots",f$knots))
        write_ref(out,paste0("test/data/m6_sparse_face_",dataset,"_",case,".csv"),
            "quantity,row,column,value; weighted L2 conversion; score_covariance is vec(K x K) per subject; source_prediction/variance use full Theta")
    }
}
untrace("face.sparse.inner", where=asNamespace("face"))
untrace("pspline", where=asNamespace("face"))
