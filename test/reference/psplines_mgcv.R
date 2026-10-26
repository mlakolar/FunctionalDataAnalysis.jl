# Run from the repository root: Rscript test/reference/psplines_mgcv.R
# References: R 4.5.1; mgcv 1.9-3 (smooth.construct.ps.smooth.spec, audited).
# mgcv P-spline bases built from the data range (no knots supplied): knots,
# basis values, first derivatives and the unscaled difference penalty.
suppressMessages(library(mgcv))
stopifnot(packageVersion("mgcv") == "1.9.3")
options(digits = 17)
header <- paste("#", R.version.string, "; mgcv", packageVersion("mgcv"), "; psplines_mgcv.R")
out <- NULL
add <- function(case, quantity, value, component = 0L) {
    out <<- rbind(out, data.frame(case, quantity, index = seq_along(value), component,
                                  value = sprintf("%.17g", as.vector(value))))
}
cases <- list(
    list(x = seq(0, 1, length = 31), k = 5, m = c(2, 1)),    # pffr bs.yindex
    list(x = seq(0, 1, length = 31), k = 20, m = c(2, 1)),   # pffr bs.int
    list(x = c(-1, -0.6, -0.3, 0.1, 0.4, 0.8, 1.1, 1.5, 1.9, 2.5), k = 7, m = c(3, 2)))
for (j in seq_along(cases)) {
    cs <- cases[[j]]
    sm <- smooth.construct(s(x, bs = "ps", k = cs$k, m = cs$m), data.frame(x = cs$x), NULL)
    xe <- seq(min(cs$x), max(cs$x), length = 9)
    X <- splines::spline.des(sm$knots, xe, cs$m[1] + 2, 0 * xe)$design
    D <- splines::spline.des(sm$knots, xe, cs$m[1] + 2, 0 * xe + 1)$design
    stopifnot(max(abs(Predict.matrix(sm, data.frame(x = xe)) - X)) < 1e-14)
    add(j, "range", range(cs$x))
    add(j, "knots", sm$knots)
    add(j, "points", xe)
    for (c in seq_len(ncol(X))) add(j, "value", X[, c], c)
    for (c in seq_len(ncol(D))) add(j, "derivative", D[, c], c)
    for (c in seq_len(ncol(X))) add(j, "S", sm$S[[1]][, c], c)
}
path <- "test/data/psplines_mgcv.csv"
writeLines(c(header, "# columns: case, quantity, index, component, value"), path)
write.table(out, path, append = TRUE, sep = ",", row.names = FALSE, col.names = FALSE, quote = FALSE)
