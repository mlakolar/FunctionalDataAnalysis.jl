# Recreate from repository root: Rscript test/reference/fdapace_m5.R
# Reference: R 4.5.1, fdapace 0.6.0. CI reads CSVs and does not require R.
# Native FPCAder is called for FPC (orders 1,2) and DPC (order 1), dense/sparse.
# Aligned DPC uses fdapace Lwls1D in BOTH arguments, trapezoidal eigenanalysis,
# stored mean and the same covariance for its prior and CE cross-covariance.
# See docs/src/fdapace.md for the deliberate departures from native DPC.
library(fdapace)
stopifnot(as.character(packageVersion("fdapace")) == "0.6.0")
options(digits=17)
read_fixture <- function(name) unname(as.matrix(read.csv(paste0("test/data/", name),
    comment.char="#", header=FALSE)))
weights <- function(t) c(diff(t)[1]/2, (diff(t)[-1]+diff(t)[-length(diff(t))])/2,
    tail(diff(t),1)/2)
system <- function(G, w, mu, cap=length(w)) {
    e <- eigen(sqrt(w) * G * rep(sqrt(w), each=length(w)), symmetric=TRUE)
    keep <- head(which(e$values > 1e-10 * max(abs(e$values))), cap)
    phi <- e$vectors[,keep,drop=FALSE]/sqrt(w)
    for (k in seq_len(ncol(phi))) {
        phi[,k] <- phi[,k]/sqrt(sum(w*phi[,k]^2))
        a <- sum(phi[,k]*mu)
        near <- abs(a) <= 1e-12 * sqrt(sum(phi[,k]^2)*sum(mu^2))
        if (if (near) phi[which.max(abs(phi[,k])),k] < 0 else a < 0)
            phi[,k] <- -phi[,k]
    }
    list(lambda=e$values[keep], phi=phi)
}
rows <- function(design, method, quantity, x) {
    x <- as.matrix(x)
    data.frame(design, method, quantity, index=rep(seq_len(nrow(x)),ncol(x)),
        component=rep(seq_len(ncol(x)),each=nrow(x)), value=as.vector(x))
}
output <- list()
record <- function(design, method, quantity, x) {
    output[[length(output)+1L]] <<- rows(design,method,quantity,x)
}
for (design in c("dense","sparse")) {
    input <- read_fixture(paste0("m3_",design,"_input.csv"))
    n <- max(input[,1]); Lt <- lapply(seq_len(n),function(i) input[input[,1]==i,2])
    Ly <- lapply(seq_len(n),function(i) input[input[,1]==i,3])
    if (design == "dense") {
        grid <- Lt[[1]]; mu <- colMeans(do.call(rbind,Ly)); G <- cov(do.call(rbind,Ly)); sigma2 <- 0
        native <- FPCA(Ly,Lt,list(dataType="Dense",methodMuCovEst="cross-sectional",
            error=FALSE,methodXi="IN",maxK=5L,methodSelectK=3L,useBinnedData="OFF",verbose=FALSE))
    } else {
        m <- read_fixture("m3_sparse_mean.csv"); grid <- m[,1]; mu <- m[,2] + .2*grid^2
        G <- read_fixture("m3_sparse_surface.csv"); sigma2 <- .12
        native <- FPCA(Ly,Lt,list(dataType="Sparse",error=TRUE,methodXi="CE",
            methodMuCovEst="smooth",nRegGrid=length(grid),maxK=5L,methodSelectK=3L,
            useBinnedData="OFF",methodRho="vanilla",userMu=list(t=grid,mu=mu),
            userCov=list(t=grid,cov=G),userSigma2=sigma2,verbose=FALSE))
    }
    w <- weights(grid); es <- system(G,w,mu,5L)
    Gfull <- es$phi %*% diag(es$lambda) %*% t(es$phi)
    aligned <- native
    aligned$phi <- es$phi[,1:3]; aligned$lambda <- es$lambda[1:3]
    aligned$mu <- mu; aligned$smoothedCov <- G; aligned$fittedCov <- Gfull
    if (design=="dense") {
        xi <- sweep(do.call(rbind,Ly),2,mu,"-") %*% (w*aligned$phi)
    } else {
        og <- sort(unique(unlist(Lt)))
        ce <- fdapace:::GetCEScores(Ly,Lt,list(verbose=FALSE),
            approx(grid,mu,og)$y,og,ConvertSupport(grid,og,Cov=Gfull),
            aligned$lambda,ConvertSupport(grid,og,phi=aligned$phi),sigma2)
        xi <- t(do.call(cbind,ce["xiEst",]))
    }
    aligned$xiEst <- xi
    aligned$sigma2 <- sigma2
    for (nu in 1:2) {
        method <- paste0("FPC_order",nu)
        d <- suppressWarnings(FPCAder(aligned,list(method="FPC",p=nu,bwMu=.12,bwCov=.15,kernelType="gauss")))
        record(design,method,"mean",d$muDer)
        record(design,method,"functions",d$phiDer)
        record(design,method,"eigenvalues",aligned$lambda)
        record(design,method,"scores",xi)
        record(design,method,"curves",sweep(xi %*% t(d$phiDer),2,d$muDer,"+"))
    }
    # Native DPC is retained to expose the implementation difference.
    # Dense native CE can use its reference solver's generalized inverse.
    native_der <- FPCAder(aligned,list(method="DPC",p=1,bwMu=.12,bwCov=.15,kernelType="gauss"))
    record(design,"native_DPC","mean",native_der$muDer)
    record(design,"native_DPC","eigenvalues",native_der$lambdaDer)
    record(design,"native_DPC","functions",native_der$phiDer)
    record(design,"native_DPC","scores",native_der$xiDer)
    # Aligned quantity: same local derivative operator on mean and covariance.
    deriv_cols <- function(x,h=.15,p=2L) apply(as.matrix(x),2,function(y)
        fdapace:::Lwls1D(h,"gauss",xin=grid,yin=y,xout=grid,npoly=p,nder=1L))
    dm <- as.vector(deriv_cols(mu,.12)); source <- if(design=="dense") G else Gfull
    C10 <- deriv_cols(source); C11 <- deriv_cols(t(C10)); C11 <- (C11+t(C11))/2
    ds <- system(C11,w,dm); dl <- ds$lambda[1:3]; dp <- ds$phi[,1:3]
    if (design=="dense") {
        centered <- sweep(do.call(rbind,Ly),2,mu,"-")
        dx <- t(deriv_cols(t(centered))) %*% (w*dp)
    } else {
        zeta <- t(C10) %*% (w*dp)
        og <- sort(unique(unlist(Lt)))
        ce <- fdapace:::GetCEScores(Ly,Lt,list(verbose=FALSE),
            approx(grid,mu,og)$y,og,ConvertSupport(grid,og,Cov=source),
            dl,ConvertSupport(grid,og,phi=sweep(zeta,2,dl,"/")),sigma2)
        dx <- t(do.call(cbind,ce["xiEst",]))
        for (i in seq_len(n)) record(design,"DPC",paste0("score_covariance_",i),ce["xiVar",][[i]])
    }
    record(design,"DPC","mean",dm)
    record(design,"DPC","covariance",C11)
    record(design,"DPC","eigenvalues",dl)
    record(design,"DPC","functions",dp)
    record(design,"DPC","scores",dx)
    record(design,"DPC","curves",sweep(dx %*% t(dp),2,dm,"+"))
}
path <- "test/data/m5_fdapace_der.csv"
writeLines(c(paste("#",R.version.string,"; fdapace",packageVersion("fdapace")),
    "# design,method,quantity,index,component,value; FPC uses aligned original eigensystem; DPC=aligned two-sided polynomial derivative and weighted eigenstep; native_DPC retained separately",
    "# fixed bwMu=.12,bwCov=.15,gauss; FPC degree=order; DPC degree=2; K=3; inputs=m3_dense_input/m3_sparse_*; sparse supplied mean adds .2*t^2"),path)
write.table(do.call(rbind,output),path,append=TRUE,sep=",",row.names=FALSE,col.names=FALSE,quote=FALSE)
cat(path,"\n")
