# Run from the repository root: Rscript test/reference/fdapace_m6.R
# Source/reference: fdapace 0.6.0. R version is recorded in every CSV header.
# Dense endpoints vanish: fdapace's rectangular and FDA's weighted SVD agree.
# Sparse: preserve native FSVD outputs and compare the documented weighted
# operator/CE quantity computed below from the native cross surface. No optimizer
# is used: all bandwidths and marginal inputs are fixed. Signs use the largest
# absolute entry of the left singular function, positive on both sides.
# m6_sparse_native_{left,right,scores1,scores2,values}.csv are reference-only:
# Julia tests do not read them because native and weighted discretizations
# differ. Tests use m6_sparse_*.csv (without "native") for weighted comparisons.
library(fdapace)
stopifnot(packageVersion("fdapace") == "0.6.0")
options(digits=17)
write_ref <- function(x, name) {
    path <- paste0("test/data/m6_", name, ".csv")
    writeLines(paste("#", R.version.string, "; fdapace 0.6.0; fdapace_m6.R"), path)
    write.table(x, path, append=TRUE, sep=",", row.names=FALSE, col.names=FALSE, quote=FALSE)
}
weights <- function(t) c(diff(t)[1]/2, (head(diff(t),-1)+tail(diff(t),-1))/2, tail(diff(t),1)/2)
orient <- function(u,v,s1,s2) {
    for (j in seq_len(ncol(u))) if(u[which.max(abs(u[,j])),j]<0) {
        u[,j]<- -u[,j]; v[,j]<- -v[,j]; s1[,j]<- -s1[,j]; s2[,j]<- -s2[,j]
    }
    list(u=u,v=v,s1=s1,s2=s2)
}
output <- function(name, g1,g2,d,u,v,s1,s2,fve) {
    write_ref(cbind(g1,u),paste0(name,"_left"))
    write_ref(cbind(g2,v),paste0(name,"_right"))
    write_ref(cbind(d,fve),paste0(name,"_values"))
    write_ref(s1,paste0(name,"_scores1"));write_ref(s2,paste0(name,"_scores2"))
}
n<-18L; g<-seq(0,1,length.out=21)
p<-sqrt(2)*outer(g,1:2,function(t,k) sin(pi*t*k))
a<-outer(1:n,1:2,function(i,k) cos(2*pi*i*k/n))
b<-outer(1:n,1:2,function(i,k) sin(2*pi*i*k/n))
x<-a%*%t(p)+matrix(1+g,n,length(g),byrow=TRUE)
y<-(a%*%matrix(c(1,.2,-.3,.6),2,2)+.2*b)%*%t(p)+matrix(.3-g,n,length(g),byrow=TRUE)
ly1<-lapply(1:n,function(i)x[i,]);ly2<-lapply(1:n,function(i)y[i,]);lt<-rep(list(g),n)
opts<-list(dataType="Dense", methodMuCovEst="cross-sectional", error=FALSE,
    methodXi="IN",shrink=FALSE,nRegGrid=length(g),maxK=2L,methodSelectK=2L,useBinnedData="OFF",verbose=FALSE)
f<-FSVD(ly1,lt,ly2,lt,opts,opts,list(maxK=2L,methodSelectK=2L))
o<-orient(f$sFun1,f$sFun2,f$sScores1,f$sScores2)
output("dense",g,g,f$sValues,o$u,o$v,o$s1,o$s2,cumsum(f$sValues^2)/sum(f$sValues^2))
write_ref(x,"dense_input1");write_ref(y,"dense_input2")
# Simultaneous irregular sampling for FCCor; five distinct bandwidths.
ltc<-lapply(1:n,function(i)g[sort(unique(c(1,21,((i+seq_len(6)*3)%%19)+2)))])
xc<-lapply(1:n,function(i)x[i,match(ltc[[i]],g)])
yc<-lapply(1:n,function(i)y[i,match(ltc[[i]],g)])
h<-c(.5,.6,.55,.65,.7)
cc<-FCCor(xc,yc,ltc,bw=h,Tout=g)
write_ref(cbind(g,cc$corr),"concurrent")
write_ref(do.call(rbind,lapply(1:n,function(i)cbind(i,ltc[[i]],xc[[i]],yc[[i]]))),"concurrent_input")
write_ref(DynCorr(x,y,g),"dynamic")
# Reuse milestone 4 observations; affine supplied means agree under R's
# userMu interpolation and the package's explicit linear interpolation.
input<-read.csv("test/data/m4_cross_input.csv",comment.char="#",header=FALSE)
get_node<-function(j,col) lapply(1:12,function(i)input[input[,1]==j & input[,2]==i,col])
t1<-get_node(1,3); t2<-get_node(2,3); y1<-get_node(1,4);y2<-get_node(2,4)
g<-seq(0,1,length.out=51); w<-weights(g); mu1<-1+g;mu2<- -.2+.3*g
phi<-cbind(rep(1,51),g-.5)
cov1<-phi%*%diag(c(.8,.3))%*%t(phi)
cov2<-phi%*%diag(c(.7,.2))%*%t(phi)
sopts<-function(mu,cov)list(dataType="Sparse", nRegGrid=51L,maxK=2L,methodSelectK=2L,
    userMu=list(t=g,mu=mu),userCov=list(t=g,cov=cov),userSigma2=.2,
    useBinnedData="OFF",verbose=FALSE)
f<-FSVD(y1,t1,y2,t2,sopts(mu1,cov1),sopts(mu2,cov2),
    list(bw1=.55,bw2=.8,kernel="epan",maxK=2L,methodSelectK=2L))
o<-orient(f$sFun1,f$sFun2,f$sScores1,f$sScores2)
output("sparse_native",g,g,f$sValues,o$u,o$v,o$s1,o$s2,cumsum(f$sValues^2)/sum(f$sValues^2))
# Weighted SVD and joint CE, independent R implementation of the package
# discretization. StackSmoothCov's PSD projection remains unweighted, as in R.
cross<-f$CrCov
d<-svd(cross*outer(sqrt(w),sqrt(w)))
val<-d$d[1:2];u<-d$u[,1:2]/sqrt(w);v<-d$v[,1:2]/sqrt(w)
stack<-rbind(cbind(cov1,cross),cbind(t(cross),cov2))
e<-eigen(stack,symmetric=TRUE);keep<-e$values>0
projected<-e$vectors[,keep,drop=FALSE]%*%diag(e$values[keep])%*%t(e$vectors[,keep,drop=FALSE])
reg<-projected+diag(.2,102)
p<-cov1%*%(u*w);q<-cov2%*%(v*w)
interp<-function(tt) sapply(1:51,function(j)approx(g,diag(51)[,j],tt)$y)
s1<-s2<-matrix(0,12,2)
for(i in 1:12) {
    h1<-interp(t1[[i]]);h2<-interp(t2[[i]])
    h<-rbind(cbind(h1,matrix(0,nrow(h1),51)),cbind(matrix(0,nrow(h2),51),h2))
    z<-solve(h%*%reg%*%t(h),c(y1[[i]]-as.vector(h1%*%mu1),y2[[i]]-as.vector(h2%*%mu2)))
    num1<-rbind(h1%*%p,sweep(h2%*%v,2,val,"*"))
    num2<-rbind(sweep(h1%*%u,2,val,"*"),h2%*%q)
    s1[i,]<-as.vector(t(num1)%*%z);s2[i,]<-as.vector(t(num2)%*%z)
}
o<-orient(u,v,s1,s2)
output("sparse",g,g,val,o$u,o$v,o$s1,o$s2,cumsum(val^2)/sum(val^2))
write_ref(cross,"sparse_cross")
cat("Generated dense FSVD, sparse native and weighted FSVD, FCCor, DynCorr references.\n")
