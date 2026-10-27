# Run from the repository root: Rscript test/reference/depth_m8.R
# References: R 4.5.1; roahd 1.4.3 and fda.usc 2.2.0 (current CRAN).
# Functions audited (source of the installed namespaces): roahd BD, MBD,
# BD_relative, MBD_relative, MEI, MHI, multiMBD, multiBD, multiMEI; fda.usc
# depth.FM, depth.MB (internal), depth.RP, depth.mode, depth.FMp, depth.RPp,
# depth.modep, FM1, TD1, mdepth.TD, mdepth.MhD, metric.lp, metric.mfdata,
# euclidean, inprod.fdata, norm.fdata, rproc2fdata, int.simpson2, Ker.norm.
# https://cran.r-project.org/src/contrib/roahd_1.4.3.tar.gz
# https://cran.r-project.org/src/contrib/fda.usc_2.2.0.tar.gz
#
# Inputs are small simulated curves written to test/data/m8_depth_input.csv;
# the RNG seed is consumed only by R. "tied" rounds the curves to one decimal
# and duplicates a curve, so every rank-based depth meets ties. Projections
# for the random-projection depths are fixed here and vanish at both grid
# ends: depth.RP projects with the trapezoidal inner product, depth.RPp with a
# plain sum; for such projections the two differ by the positive factor of
# the grid step, which Tukey depth ignores. Projected values stay far below
# 16 in magnitude, where fda.usc's TD1 offset (x - 1e-15) still separates a
# value from itself; see the "Depth and outliers" docs page. roahd's BD counts only
# curve pairs with one curve strictly above and one strictly below on the
# whole grid; the band-depth definition (graph inside the band) is computed
# here directly from pairs as `BD_graph`. CI reads the committed CSVs and does
# not need R.
suppressMessages({library(roahd); library(fda.usc)})
stopifnot(packageVersion("roahd") == "1.4.3", packageVersion("fda.usc") == "2.2.0")
options(digits = 17)
header <- paste("#", R.version.string, "; roahd", packageVersion("roahd"),
                "; fda.usc", packageVersion("fda.usc"), "; depth_m8.R")

set.seed(8101)
m <- 15
grid <- seq(0, 1, length.out = m)
grid_nu <- c(0, 0.03, 0.08, 0.15, 0.2, 0.3, 0.38, 0.45, 0.55, 0.62, 0.7, 0.8, 0.88, 0.95, 1)
curves <- function(n, shift = 0) {
    a <- rnorm(n, shift, 0.25); b <- rnorm(n, 0, 0.2); c <- rnorm(n, 0, 0.1)
    outer(a, rep(1, m)) + outer(b, sin(2 * pi * grid)) + outer(c, cos(4 * pi * grid)) +
        matrix(rnorm(n * m, 0, 0.03), n, m)
}
smooth <- curves(12)
smooth_new <- rbind(curves(3), smooth[5, ] + 0.6)
tied <- round(smooth, 1)
tied[12, ] <- tied[3, ]
tied_new <- rbind(round(curves(3), 1), tied[7, ])
mv1 <- smooth
mv2 <- curves(12, 0.1) + 0.5 * smooth
mv1_new <- smooth_new
mv2_new <- curves(4, 0.1) + 0.5 * smooth_new
mvt1 <- round(mv1, 1)
mvt2 <- round(mv2, 1)
proj <- matrix(rnorm(5 * m), 5, m)
proj[, c(1, m)] <- 0

inputs <- list(grid = matrix(grid, 1), grid_nu = matrix(grid_nu, 1), smooth = smooth,
    smooth_new = smooth_new, tied = tied, tied_new = tied_new, mv1 = mv1, mv2 = mv2,
    mv1_new = mv1_new, mv2_new = mv2_new, mvt1 = mvt1, mvt2 = mvt2, proj = proj)
input_rows <- do.call(rbind, lapply(names(inputs), function(name) {
    x <- inputs[[name]]
    data.frame(dataset = name, row = as.vector(row(x)), col = as.vector(col(x)), value = as.vector(x))
}))
input_path <- "test/data/m8_depth_input.csv"
writeLines(c(header, "# dataset,row,col,value"), input_path)
input_rows$value <- sprintf("%.17g", input_rows$value)
write.table(input_rows, input_path, append = TRUE, sep = ",", row.names = FALSE,
            col.names = FALSE, quote = FALSE)

# Band depth by its definition: pairs of reference curves whose band contains
# the whole graph. The sample depth counts pairs that include the curve itself.
bd_graph <- function(target, reference) {
    n0 <- nrow(reference)
    apply(target, 1, function(x) {
        count <- 0
        for (j in 1:(n0 - 1)) for (k in (j + 1):n0) {
            lo <- pmin(reference[j, ], reference[k, ]); hi <- pmax(reference[j, ], reference[k, ])
            count <- count + all(lo <= x & x <= hi)
        }
        count / choose(n0, 2)
    })
}

out <- NULL
rows <- function(case, quantity, value) {
    data.frame(case = case, quantity = quantity, index = seq_along(value), value = as.vector(value))
}
add <- function(case, quantity, value) out <<- rbind(out, rows(case, quantity, value))
fd <- function(x, t = grid) fdata(x, t)

for (case in c("smooth", "tied")) {
    x <- inputs[[case]]; xnew <- inputs[[paste0(case, "_new")]]
    add(case, "BD", BD(x))
    add(case, "BD_relative", BD_relative(xnew, x))
    add(case, "BD_graph", bd_graph(x, x))
    add(case, "BD_graph_relative", bd_graph(xnew, x))
    add(case, "MBD", MBD(x))
    add(case, "MBD_ties", MBD(x, manage_ties = TRUE))
    add(case, "MBD_relative", MBD_relative(xnew, x))
    add(case, "MEI", MEI(x))
    add(case, "MHI", MHI(x))
    add(case, "depth.FM", depth.FM(fd(x))$dep)
    add(case, "depth.FM_lmed", depth.FM(fd(x))$lmed)
    add(case, "depth.FM_relative", depth.FM(fd(xnew), fd(x))$dep)
    add(case, "depth.MB", fda.usc:::depth.MB(fd(x))$dep)
    add(case, "depth.MB_relative", fda.usc:::depth.MB(fd(xnew), fd(x))$dep)
    add(case, "depth.RP", depth.RP(fd(x), proj = fd(proj))$dep)
    add(case, "depth.RP_relative", depth.RP(fd(xnew), fd(x), proj = fd(proj))$dep)
    mode <- depth.mode(fd(x))
    add(case, "depth.mode", mode$dep)
    add(case, "depth.mode_hq", mode$hq)
    add(case, "depth.mode_relative", depth.mode(fd(xnew), fd(x))$dep)
}
stopifnot(any(abs(BD(tied) - bd_graph(tied, tied)) > 0))

# Nonuniform grid: fda.usc weights depth.MB by trapezoidal rule and integrates
# the L2 metric by trapezoids; depth.FM averages grid points with equal weight.
nu <- fd(smooth, grid_nu)
add("nonuniform", "depth.MB", fda.usc:::depth.MB(nu)$dep)
add("nonuniform", "depth.FM", depth.FM(nu)$dep)
add("nonuniform", "MBD", MBD(smooth))
nu_mode <- depth.mode(nu)
add("nonuniform", "depth.mode", nu_mode$dep)
add("nonuniform", "depth.mode_hq", nu_mode$hq)
add("nonuniform", "depth.RP", depth.RP(nu, proj = fd(proj, grid_nu))$dep)

# Multivariate: two components on the shared uniform grid.
mf <- list(x1 = fd(mv1), x2 = fd(mv2))
mf_new <- list(x1 = fd(mv1_new), x2 = fd(mv2_new))
add("multivariate", "multiMBD", multiMBD(list(mv1, mv2)))
add("multivariate", "multiMBD_weighted", multiMBD(list(mv1, mv2), weights = c(0.3, 0.7)))
add("multivariate", "multiBD", multiBD(list(mv1, mv2)))
add("multivariate", "multiMBD_tied", multiMBD(list(mvt1, mvt2)))
add("multivariate", "multiMBD_tied_ties", multiMBD(list(mvt1, mvt2), manage_ties = TRUE))
add("multivariate", "depth.FMp", depth.FMp(mf)$dep)
add("multivariate", "depth.FMp_relative", depth.FMp(mf_new, mf)$dep)
invisible(capture.output(rpp <- depth.RPp(mf, proj = fd(proj))))
invisible(capture.output(rpp_new <- depth.RPp(mf_new, mf, proj = fd(proj))))
add("multivariate", "depth.RPp", rpp$dep)
add("multivariate", "depth.RPp_relative", rpp_new$dep)
modep <- depth.modep(mf)
add("multivariate", "depth.modep", modep$dep)
add("multivariate", "depth.modep_hq", modep$hq)
add("multivariate", "depth.modep_relative", depth.modep(mf_new, mf)$dep)

path <- "test/data/m8_depth_golden.csv"
writeLines(c(header, "# case,quantity,index,value"), path)
out$value <- sprintf("%.17g", out$value)
write.table(out, path, append = TRUE, sep = ",", row.names = FALSE, col.names = FALSE, quote = FALSE)
