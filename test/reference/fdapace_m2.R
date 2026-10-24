# Run from the repository root: Rscript test/reference/fdapace_m2.R
# Fixed irregular observations; no random generation or binning.
library(fdapace)
options(digits = 17)
dir.create("test/data", showWarnings = FALSE)
header <- paste("#", R.version.string, "; fdapace", packageVersion("fdapace"))
write_reference <- function(x, path) {
    writeLines(header, path)
    write.table(x, path, append = TRUE, sep = ",", row.names = FALSE,
                col.names = FALSE, quote = FALSE)
}
times <- list(seq(0, 1, length.out = 9), c(0, .15, .4, .7, 1),
              c(0, .2, .35, .6, .8, 1), c(0, .1, .3, .55, .75, .9, 1))
d <- do.call(rbind, lapply(seq_along(times), function(i) {
    t <- times[[i]]
    data.frame(curve = i, time = t,
               value = 1 + t + .4 * t^2 + (i - 2.5) * (.3 + .2 * t) +
                       .05 * cos(7 * t + i))
}))
write_reference(d, "test/data/m2_input.csv")
grid <- seq(0, 1, length.out = 9)
h <- .65
ord <- order(d$time)
mean_at <- function(points, weighting) {
    w <- if (weighting == "obs") rep(1, nrow(d)) else 1 / lengths(times)[d$curve]
    Lwls1D(h, "epan", win = w[ord], xin = d$time[ord],
           yin = d$value[ord], xout = points)
}
means <- lapply(c("obs", "subj"), function(weighting) {
    data.frame(weighting, bandwidth = h, time = grid, value = mean_at(grid, weighting))
})
write_reference(do.call(rbind, means), "test/data/m2_fdapace_mean.csv")
# Task 3 fixture: all ordered within-curve off-diagonal residual products.
# Subject weights give each curve's N_i*(N_i-1) pairs total weight one.
# Center at the matching pooled mean evaluated directly at observation times.
covs <- lapply(c("obs", "subj"), function(weighting) {
    mu <- mean_at(sort(unique(d$time)), weighting)
    residual <- d$value - mu[match(d$time, sort(unique(d$time)))]
    pairs <- do.call(rbind, lapply(seq_along(times), function(i) {
        ix <- which(d$curve == i)
        ij <- expand.grid(a = ix, b = ix)
        ij <- ij[ij$a != ij$b, ]
        data.frame(t = d$time[ij$a], s = d$time[ij$b],
                   z = residual[ij$a] * residual[ij$b],
                   w = if (weighting == "obs") 1 else 1 / nrow(ij))
    }))
    surface <- Lwls2D(h, "epan", xin = as.matrix(pairs[, c("t", "s")]),
                      yin = pairs$z, win = pairs$w, xout1 = grid, xout2 = grid,
                      method = "plain")
    data.frame(weighting, bandwidth = h, expand.grid(t = grid, s = grid),
               value = as.vector(surface))
})
write_reference(do.call(rbind, covs), "test/data/m2_fdapace_cov.csv")
