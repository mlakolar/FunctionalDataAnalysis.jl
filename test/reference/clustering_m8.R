# Run from the repository root: Rscript test/reference/clustering_m8.R
# References: R 4.5.1; fdapace 0.6.0 (kCFC, FPCA); fda.usc 2.2.0 (kmeans.fd,
# metric.lp); EMCluster 0.2-18 (emcluster, e.step, logL, em.aic, em.bic);
# stats::kmeans. Sources audited: fdapace R/kCFC.R, R/FClust.R,
# R/GetISEfromFPCA.R, R/GetINScores.R; fda.usc R/kmeans.fd.R,
# R/kmeans.center.ini.R, R/kmeans.assig.groups.R, R/kmeans.centers.update.R;
# EMCluster src/{init_other.c, M_init_other.c, M_emcluster.c, emcluster.c,
# initials.c, it_init_EM.c}. CI reads the committed CSVs and does not need R.
#
# Data: three clusters of twelve noise-free curves on 26 equally spaced points.
# All deviations from the common endpoint values vanish at both endpoints, so
# fdapace's rectangular eigen discretization and the package's trapezoidal one
# coincide (milestone 3), globally and within every cluster.
#
# The goldens use supplied initializations wherever the reference draws random
# numbers: kmeans.fd and stats::kmeans receive seed curve indices, kCFC's
# initial k-means labels (set.seed(kSeed) inside kCFC) are recorded and
# supplied to the package, and EMCluster starts from parameters computed from
# supplied labels exactly as its `initials` C routine does (class means,
# covariances with denominator n_c - 1, proportions n_c/n). EMCluster's EM is
# run to its fixed point (em.eps = 0) and the fixed point is verified below.
suppressMessages({library(fdapace); library(fda.usc); library(EMCluster)})
stopifnot(packageVersion("fdapace") == "0.6.0", packageVersion("fda.usc") == "2.2.0",
          packageVersion("EMCluster") == "0.2.18")
options(digits = 17)
header <- paste("#", R.version.string, "; fdapace", packageVersion("fdapace"),
                "; fda.usc", packageVersion("fda.usc"), "; EMCluster",
                packageVersion("EMCluster"), "; clustering_m8.R")
rows <- function(quantity, index, component, value) {
    data.frame(quantity = quantity, index = index, component = component, value = as.vector(value))
}
matrix_rows <- function(quantity, x) {
    x <- as.matrix(x)
    rows(quantity, rep(seq_len(nrow(x)), ncol(x)), rep(seq_len(ncol(x)), each = nrow(x)), x)
}
vector_rows <- function(quantity, x, component = 0L) rows(quantity, seq_along(x), component, x)

set.seed(20261027)
grid <- seq(0, 1, length.out = 26)
n_per <- 12L
labels <- rep(1:3, each = n_per)
n <- length(labels)
s <- sqrt(2) * outer(grid, 1:4, function(t, j) sin(j * pi * t))
base <- 1 + grid
shift <- rbind(c(0, 0), c(1.2, 0), c(0.3, 1.0))
sds <- rbind(c(1.0, 0.45, 0.25, 0.10), c(0.35, 0.90, 0.30, 0.10), c(0.40, 0.30, 0.80, 0.25))
y <- t(sapply(seq_len(n), function(i) {
    c <- labels[i]
    base + s[, 1:2] %*% shift[c, ] + s %*% (sds[c, ] * rnorm(4))
}))
w <- c(diff(grid)[1] / 2, (head(diff(grid), -1) + tail(diff(grid), -1)) / 2, tail(diff(grid), 1) / 2)
Ly <- lapply(seq_len(n), function(i) y[i, ])
Lt <- rep(list(grid), n)

# fda.usc kmeans.fd with seed curves and trim = 0 (func.trim.FM with trim 0
# is the cross-sectional mean). kmeans.fd never recomputes its distance matrix
# after the first centre update, so its labels are the nearest initial seed.
# The data are chosen so that one further Lloyd step changes nothing, which is
# checked here; the package's converged Lloyd iteration then agrees.
fd <- fdata(y, grid)
D <- metric.lp(fd)
stopifnot(max(abs(D[1, ] - sqrt(colSums(w * (t(y) - y[1, ])^2)))) < 1e-12)
# First seed triple (one curve per simulated cluster, lexicographic order)
# whose nearest-seed partition is a Lloyd fixed point with clusters of size >= 5.
lloyd_fixed <- function(seeds) {
    lab <- apply(D[seeds, , drop = FALSE], 2, which.min)
    if (length(unique(lab)) < 3 || min(table(lab)) < 5) return(FALSE)
    cen <- t(sapply(1:3, function(j) colMeans(y[lab == j, , drop = FALSE])))
    nd <- sapply(1:3, function(j) colSums(w * (t(y) - cen[j, ])^2))
    all(apply(nd, 1, which.min) == lab)
}
candidates <- expand.grid(a = 1:12, b = 13:24, c = 25:36)
seeds_curves <- NULL
for (r in seq_len(nrow(candidates))) {
    if (lloyd_fixed(unlist(candidates[r, ]))) { seeds_curves <- as.integer(unlist(candidates[r, ])); break }
}
stopifnot(!is.null(seeds_curves))
kfd <- kmeans.fd(fd, ncl = seeds_curves, par.dfunc = list(trim = 0), draw = FALSE)
seed_distance <- t(D[seeds_curves, ])
stopifnot(all(kfd$cluster == apply(seed_distance, 1, which.min)))
centres <- kfd$centers$data
next_distance <- sapply(1:3, function(j) sqrt(colSums(w * (t(y) - centres[j, ])^2)))
stopifnot(all(apply(next_distance, 1, which.min) == kfd$cluster))

# Global FPCA (dense cross-sectional, IN scores) as in FClust's EMCluster path
# but without smoothing; FVE threshold 0.9 is FClust's default optnsFPCA.
dense <- list(dataType = "Dense", methodMuCovEst = "cross-sectional", error = FALSE,
              methodXi = "IN", useBinnedData = "OFF", verbose = FALSE)
global <- FPCA(Ly, Lt, c(dense, list(FVEthreshold = 0.9)))
stopifnot(length(global$workGrid) == length(grid), max(abs(global$workGrid - grid)) < 1e-14)
xi <- global$xiEst
p <- ncol(xi)

# stats::kmeans (Lloyd) on FPC scores from supplied seed rows.
seeds_scores <- c(1L, 2L, 25L)
km <- kmeans(xi, centers = xi[seeds_scores, , drop = FALSE], algorithm = "Lloyd", iter.max = 100)

# fdapace kCFC on dense data with cross-sectional FPCAs.
kc <- kCFC(Ly, Lt, k = 3, optnsSW = c(dense, list(FVEthreshold = 0.9)),
           optnsCS = c(dense, list(FVEthreshold = 0.7)))
ymat <- fdapace:::List2Mat(Ly, Lt)
costs <- sapply(kc$fpcaList, function(u) fdapace:::GetISEfromFPCA(u, Ly, Lt, ymat))
stopifnot(all(as.integer(kc$cluster) == apply(costs, 1, which.min)))
history <- sapply(kc$prevConf, function(u) as.integer(as.character(u)))

# EMCluster from labels with two curves moved, run to its fixed point.
em_labels <- labels
em_labels[c(3, 20)] <- c(2L, 3L)
pi0 <- as.vector(table(em_labels)) / n
Mu0 <- t(sapply(1:3, function(c) colMeans(xi[em_labels == c, , drop = FALSE])))
LT0 <- variance2LTSigma(simplify2array(lapply(1:3, function(c) cov(xi[em_labels == c, , drop = FALSE]))))
# emcluster's stopping rule is a relative log-likelihood change. Even with
# em.eps = 0 it stops once the log likelihood is flat to rounding, while one
# more EM step still moves the means by about 1e-8. The golden is therefore
# the fixed point of EMCluster's own E and M steps (e.step, m.step), reached
# from emcluster's result; its labels equal emcluster's.
native <- emcluster(xi, pi = pi0, Mu = Mu0, LTSigma = LT0,
                    EMC = .EMControl(em.iter = 5000, em.eps = 0), assign.class = TRUE)
em <- native
for (r in 1:10000) {
    previous <- em
    em <- m.step(xi, Gamma = e.step(xi, emobj = em)$Gamma)
    change <- max(abs(em$Mu - previous$Mu), abs(em$pi - previous$pi),
                  abs(em$LTSigma - previous$LTSigma))
    if (change < 1e-15) break
}
stopifnot(change < 1e-15)
cat("emcluster stopped after", native$conv.iter, "iterations; fixed point moved means by",
    max(abs(em$Mu - native$Mu)), "\n")
em <- assign.class(xi, em)
stopifnot(all(em$class == native$class))
posterior <- e.step(xi, emobj = em)$Gamma
loglik <- logL(xi, emobj = em)

out <- rbind(
    vector_rows("grid", grid), matrix_rows("y", y), vector_rows("label", labels),
    vector_rows("fdausc_seeds", seeds_curves), vector_rows("fdausc_cluster", kfd$cluster),
    matrix_rows("fdausc_centers", centres), matrix_rows("fdausc_seed_distance", seed_distance),
    matrix_rows("fpca_scores", xi), vector_rows("fpca_mean", global$mu),
    vector_rows("fpca_lambda", global$lambda),
    vector_rows("kmeans_seeds", seeds_scores), vector_rows("kmeans_cluster", km$cluster),
    matrix_rows("kmeans_centers", km$centers), vector_rows("kmeans_withinss", km$withinss),
    vector_rows("kcfc_initial", as.integer(as.character(kc$clustConf0))),
    vector_rows("kcfc_cluster", as.integer(as.character(kc$cluster))),
    vector_rows("kcfc_iterations", kc$iterToConv),
    matrix_rows("kcfc_costs", costs), matrix_rows("kcfc_history", history),
    do.call(rbind, lapply(1:3, function(c) rbind(
        vector_rows("kcfc_mean", kc$fpcaList[[c]]$mu, c),
        vector_rows("kcfc_lambda", kc$fpcaList[[c]]$lambda, c)))),
    vector_rows("em_initial", em_labels), vector_rows("em_pi", em$pi),
    matrix_rows("em_mu", em$Mu),
    matrix_rows("em_sigma", do.call(rbind, lapply(1:3, function(c)
        LTSigma2variance(em$LTSigma[c, , drop = FALSE])[, , 1]))),
    matrix_rows("em_posterior", posterior), vector_rows("em_class", em$class),
    vector_rows("em_loglik", loglik), vector_rows("em_aic", em.aic(xi, emobj = em)),
    vector_rows("em_bic", em.bic(xi, emobj = em)))
cat("n =", n, "; FPCA K =", p, "; kmeans iterations =", km$iter,
    "; kCFC iterations =", kc$iterToConv, "; changed initial labels =",
    sum(as.integer(as.character(kc$clustConf0)) != as.integer(as.character(kc$cluster))),
    "; EM loglik =", loglik, "\n")
path <- "test/data/m8_clustering.csv"
writeLines(c(header, "# quantity,index,component,value; em_sigma stacks the cluster covariances by rows"), path)
out$value <- sprintf("%.17g", out$value)
write.table(out, path, append = TRUE, sep = ",", row.names = FALSE, col.names = FALSE, quote = FALSE)
