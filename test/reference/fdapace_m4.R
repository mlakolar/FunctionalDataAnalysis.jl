# Run from the repository root: Rscript test/reference/fdapace_m4.R
# Fixed two-node sparse data. Reference: R 4.5.1, fdapace 0.6.0.
# GetCrCovYX: supplied means, bw1=.55, bw2=.8, kern="epan", useGAM=FALSE;
# compare smoothedCC on its returned 51 x 51 grids, with both rmDiag settings.
# Exact-duplicate aggregation by BinRawCov retains observation-pair weights.
library(fdapace)
stopifnot(packageVersion("fdapace") == "0.6.0")
options(digits = 17)
header <- paste("#", R.version.string, "; fdapace", packageVersion("fdapace"),
                "; GetCrCovYX, epan, bw=(.55,.8), observation-pair weights")
write_reference <- function(x, path) {
    writeLines(header, path)
    write.table(x, path, append = TRUE, sep = ",", row.names = FALSE,
                col.names = FALSE, quote = FALSE)
}
times1 <- lapply(1:12, function(i) seq(0, 1, length.out = 3 + i %% 4))
times2 <- lapply(1:12, function(i) seq(0, 1, length.out = 4 + i %% 3))
mu1 <- function(t) 1 + t
mu2 <- function(t) -.2 + .3 * t
values1 <- lapply(1:12, function(i) {
    t <- times1[[i]]
    mu1(t) + (i - 6.5) * (.2 + .3 * t) + .15 * cos(6 * t + i)
})
values2 <- lapply(1:12, function(i) {
    t <- times2[[i]]
    mu2(t) + (i - 6.5) * (.4 - .2 * t) + .1 * sin(5 * t + i)
})
input <- do.call(rbind, lapply(1:2, function(node) {
    tt <- if (node == 1) times1 else times2
    yy <- if (node == 1) values1 else values2
    do.call(rbind, lapply(1:12, function(i) data.frame(node, subject = i,
        time = tt[[i]], value = yy[[i]])))
}))
write_reference(input, "test/data/m4_cross_input.csv")
reference <- lapply(c(FALSE, TRUE), function(exclude) {
    result <- GetCrCovYX(bw1 = .55, bw2 = .8, Ly1 = values1, Lt1 = times1,
        Ymu1 = mu1(sort(unique(unlist(times1)))), Ly2 = values2, Lt2 = times2,
        Ymu2 = mu2(sort(unique(unlist(times2)))), kern = "epan", rmDiag = exclude)
    stopifnot(all(is.finite(result$smoothedCC)), inherits(result$rawCC, "BinnedRawCC"))
    data.frame(exclude_same_time = as.integer(exclude), h1 = result$bw[1],
        h2 = result$bw[2], expand.grid(s = result$smoothGrid[, 1],
        t = result$smoothGrid[, 2]), value = as.vector(result$smoothedCC))
})
write_reference(do.call(rbind, reference), "test/data/m4_fdapace_crosscov.csv")
