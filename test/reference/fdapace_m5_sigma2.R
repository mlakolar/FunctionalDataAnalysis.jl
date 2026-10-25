# Run from the repository root: Rscript test/reference/fdapace_m5_sigma2.R
# Reference: R 4.5.1, fdapace 0.6.0. Fixed milestone-2 observations.
# PC_CovE-level noise, not FPCA's negative-to-1e-6 replacement.
library(fdapace)
stopifnot(packageVersion("fdapace") == "0.6.0")
options(digits = 17)
d <- read.csv("test/data/m2_input.csv", comment.char = "#", header = FALSE)
names(d) <- c("curve", "time", "value")
counts <- tabulate(d$curve)
grids <- list(aligned = seq(0, 1, length.out = 9),
              unaligned = c(0, .18, .31, .56, .69, .84, 1),
              extended = c(-.1, .1, .35, .55, .7, .9, 1.1))
rows <- list()
for (weighting in c("obs", "subj")) {
    w <- if (weighting == "obs") rep(1, nrow(d)) else 1 / counts[d$curve]
    ord <- order(d$time)
    # As in m2: direct mean at each observation time, h=.65, epan.
    mu <- Lwls1D(.65, "epan", win = w[ord], xin = d$time[ord],
                 yin = d$value[ord], xout = sort(unique(d$time)))
    residual <- d$value - mu[match(d$time, sort(unique(d$time)))]
    pairs <- do.call(rbind, lapply(seq_along(counts), function(i) {
        ix <- which(d$curve == i)
        ij <- expand.grid(a = ix, b = ix)
        ij <- ij[ij$a != ij$b, ]
        data.frame(t = d$time[ij$a], s = d$time[ij$b],
                   z = residual[ij$a] * residual[ij$b],
                   w = if (weighting == "obs") 1 else 1 / nrow(ij))
    }))
    # BinnedRawCov is PC_CovE's weighted-input interface. No times are
    # snapped or merged here. Counts encode our observation/subject weights.
    # Sort both the raw squares AND their weights before passing to PC_CovE:
    # PC_CovE sorts diagonal values but does not permute win2 itself.
    rcov <- structure(list(tPairs = as.matrix(pairs[, c("t", "s")]),
        cxxn = pairs$z, meanVals = pairs$z, count = pairs$w,
        diag = cbind(d$time[ord], residual[ord]^2),
        tDiag = d$time[ord], diagMeans = residual[ord]^2, diagCount = w[ord]),
        class = "BinnedRawCov")
    for (kernel in c("epan", "gauss")) for (h in c(.35, .65)) {
        for (grid_name in names(grids)) for (interval in c("middle_half", "full")) {
            grid <- grids[[grid_name]]
            cut <- if (interval == "middle_half") c(.25, .75) else c(0, 1)
            fit <- fdapace:::PC_CovE(sort(unique(d$time)), grid, c(h, h),
                                    rotationCut = cut, kernel = kernel, rcov = rcov)
            buff <- .Machine$double.eps * max(abs(d$time)) * 10
            retained <- grid[grid > cut[1] - buff & grid < cut[2] + buff]
            stopifnot(all(is.finite(c(fit$sigma2, fit$xvar, fit$yvar))))
            add <- function(quantity, times, values) {
                data.frame(weighting, kernel, bandwidth = h, grid = grid_name,
                    interval, quantity, index = seq_along(values), time = times,
                    value = values)
            }
            rows <- c(rows, list(add("raw_variance", 0, fit$sigma2),
                add("latent_diagonal", retained, fit$xvar),
                add("observed_diagonal", retained, fit$yvar)))
        }
    }
}
path <- "test/data/m5_fdapace_sigma2.csv"
writeLines(c(paste("#", R.version.string, "; fdapace", packageVersion("fdapace")),
    "# PC_CovE; fixed m2_input.csv; mean epan h=.65; weighted BinnedRawCov interface; no binning",
    "# weighting,kernel,bandwidth,grid,interval,quantity,index,time,value"), path)
write.table(do.call(rbind, rows), path, append = TRUE, sep = ",",
            row.names = FALSE, col.names = FALSE, quote = FALSE)
