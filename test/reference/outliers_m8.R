# Run from the repository root: Rscript test/reference/outliers_m8.R
# References: R 4.5.1; roahd 1.4.3, fda 6.3.0, fdaoutlier 0.2.1, MASS 7.3-65,
# robustbase 0.99-7 (current CRAN).
# Functions audited (source of the installed namespaces and of fdaoutlier's
# C++ and MASS's lqs.c): roahd fbplot.fData, fbplot.mfData, .fbplot_fData,
# .fbplot_mfData, outliergram, .outliergram, manage_high_MEI_data,
# manage_low_MEI_data, compute_min_diff_min, compute_max_diff_max,
# generate_gauss_fdata, MBD.default, MEI.default, multiMBD.default; fda fbplot,
# fMBD; fdaoutlier msplot, dir_out, projection_depth (projectionDepth,
# customMedian, customMAD), hardin_factor_numeric, croux_hesbroeck_asymptotic;
# MASS cov.rob and mve_fitlots; robustbase covOGK, Qn, Qn.finite.c, s_Qn, covGK,
# scaleTau2.
# https://cran.r-project.org/src/contrib/roahd_1.4.3.tar.gz
# https://cran.r-project.org/src/contrib/fdaoutlier_0.2.1.tar.gz
#
# Inputs are small simulated curves written to test/data/m8_outliers_input.csv;
# the RNG streams are consumed only by R. The univariate sample (24 curves on
# 20 points) has a magnitude outlier (5), a shape outlier (9) and a low curve
# (16) whose parameters were drawn, as below, by a search for data where
# roahd's shift check flags a curve the first stage misses. The script checks
# that roahd's first stage flags at least one curve (otherwise roahd skips the
# shift check) and that its grep-based matching of shift candidates is not
# confused by indices such as 1 and 16. MASS's MCD enumerates every subset
# here (fewer than 5000), so no RNG enters it. fdaoutlier's projection
# directions are regenerated here from its seed and stored with the inputs.
# The adjusted-factor trial reproduces one iteration of roahd's
# fbplot(adjust=...) with a stored simulated sample. CI reads the committed
# CSVs and does not need R.
suppressMessages({library(roahd); library(fda); library(fdaoutlier); library(MASS); library(robustbase)})
stopifnot(packageVersion("roahd") == "1.4.3", packageVersion("fda") == "6.3.0",
          packageVersion("fdaoutlier") == "0.2.1", packageVersion("MASS") == "7.3.65",
          packageVersion("robustbase") == "0.99.7")
options(digits = 17)
header <- paste("#", R.version.string, "; roahd", packageVersion("roahd"), "; fda", packageVersion("fda"),
                "; fdaoutlier", packageVersion("fdaoutlier"), "; MASS", packageVersion("MASS"),
                "; robustbase", packageVersion("robustbase"), "; outliers_m8.R")

m <- 20
grid <- seq(0, 1, length.out = m)
n <- 24
set.seed(7324)
a <- rnorm(n, 0, 0.5); b <- rnorm(n, 0, 0.3)
uni <- outer(a, rep(1, m)) + outer(b, sin(2 * pi * grid)) + matrix(rnorm(n * m, 0, 0.05), n, m)
uni[5, ] <- uni[5, ] + 3
uni[9, ] <- 0.1 + 0.6 * sin(8 * pi * grid)
amp <- runif(1, 0.1, 0.8); unused <- runif(1, 0.3, 1.5); fr <- sample(2:6, 1)
uni[16, ] <- apply(uni[-16, ], 2, min) - amp * (1 + sin(fr * pi * grid + runif(1, 0, 2 * pi))) / 2 +
    runif(1, -0.1, 0.4)

# Ties in depth at the median: roahd's central region (depth >= median depth)
# then holds more than half the curves. Search the first eleven typical
# curves, rounded, for a duplicated curve that creates such a tie.
typical <- setdiff(seq_len(n), c(5, 9, 16))
found <- FALSE
for (start in 0:10) {
    base <- round(uni[typical[start + 1:11], ], 1)
    for (k in seq_len(nrow(base))) {
        tied <- rbind(base, base[k, ])
        dk <- MBD(tied, manage_ties = TRUE)
        if (sum(dk >= quantile(dk, 0.5)) != ceiling(nrow(tied) / 2)) { found <- TRUE; break }
    }
    if (found) break
}
stopifnot(found)

# Bivariate sample: 20 curves, a magnitude outlier in the second component (3)
# and a shape outlier in the first (11).
nm <- 20
set.seed(8202)
s1 <- rnorm(nm, 0, 0.4); s2 <- rnorm(nm, 0, 0.3)
mv1 <- outer(s1, rep(1, m)) + outer(s2, cos(2 * pi * grid)) + matrix(rnorm(nm * m, 0, 0.05), nm, m)
mv2 <- 0.5 * mv1 + outer(rnorm(nm, 0, 0.3), grid) + matrix(rnorm(nm * m, 0, 0.05), nm, m)
mv2[3, ] <- mv2[3, ] + 2.5
mv1[11, ] <- 0.4 * sin(10 * pi * grid)
seed_projections <- 7
nproj <- 50
set.seed(seed_projections)
directions <- matrix(runif(2 * nproj, -1, 1), nproj, 2)

# One trial of roahd's adjusted factor, with the default rate.
trial_size <- 60
cov_ogk <- robustbase::covOGK(uni, sigmamu = robustbase::s_Qn)$cov
centerline <- uni[which.max(MBD(uni, manage_ties = TRUE)), ]
set.seed(8203)
sim <- roahd:::generate_gauss_fdata(trial_size, centerline, CholCov = chol(cov_ogk))
rate <- 2 * pnorm(4 * qnorm(0.25))
cost <- function(f) length(roahd:::.fbplot_fData(grid, sim, Depths = "MBD", Fvalue = f)$ID_out) / trial_size - rate
trial <- uniroot(cost, interval = c(0.5, 5), tol = 0.001, maxiter = 100)

# robustbase's Qn0 truncates candidate differences to single precision while
# searching and returns that value when the search ends early, so Qn can
# differ from the exact order statistic by about 1e-8. The OGK golden uses
# covOGK with the exact Qn (same constants); qn_column is a column of the
# OGK rotations of the curves where the two differ.
qn_exact <- function(x, mu.too = FALSE, ...) {
    nx <- length(x); h <- nx %/% 2 + 1
    r <- 2.21914 * sort(abs(outer(x, x, "-")[upper.tri(diag(nx))]))[choose(h, 2)]
    r <- if (nx <= 12) r * c(0.399356, 0.99365, 0.51321, 0.84401, 0.6122, 0.85877, 0.66993, 0.87344,
                             0.72014, 0.88906, 0.75743)[nx - 1L] else r / robustbase:::Qn.finite.c(nx)
    c(if (mu.too) median(x), r)
}
z <- uni
rotations <- NULL
for (iter in 1:2) {
    z <- sweep(z, 2L, apply(z, 2L, qn_exact), "/")
    u <- diag(m)
    for (i in 2:m) for (j in 1:(i - 1)) u[i, j] <- robustbase::covGK(z[, i], z[, j])
    z <- z %*% eigen(u, symmetric = TRUE)$vectors
    rotations <- cbind(rotations, z)
}
truncated <- which(apply(rotations, 2L, robustbase::Qn) != apply(rotations, 2L, qn_exact))
stopifnot(length(truncated) > 0)
qn_column <- matrix(rotations[, truncated[1]], 1)

inputs <- list(grid = matrix(grid, 1), uni = uni, tied = tied, mv1 = mv1, mv2 = mv2,
               directions = directions, sim = sim, qn_column = qn_column)
input_rows <- do.call(rbind, lapply(names(inputs), function(name) {
    x <- inputs[[name]]
    data.frame(dataset = name, row = as.vector(row(x)), col = as.vector(col(x)), value = as.vector(x))
}))
input_path <- "test/data/m8_outliers_input.csv"
writeLines(c(header, "# dataset,row,col,value"), input_path)
input_rows$value <- sprintf("%.17g", input_rows$value)
write.table(input_rows, input_path, append = TRUE, sep = ",", row.names = FALSE,
            col.names = FALSE, quote = FALSE)

out <- NULL
# An empty result (no flagged curve) is one row with index 0.
add <- function(case, quantity, value) {
    value <- as.vector(value)
    rows <- if (length(value)) data.frame(case = case, quantity = quantity, index = seq_along(value), value = value)
        else data.frame(case = case, quantity = quantity, index = 0, value = 0)
    out <<- rbind(out, rows)
}

# Functional boxplots.
fb <- roahd::fbplot(fData(grid, uni), display = FALSE)
env <- roahd:::.fbplot_fData(grid, uni, fb$Depth, 1.5)
add("uni", "fbplot_depth", fb$Depth)
add("uni", "fbplot_outliers", sort(fb$ID_outliers))
add("uni", "fbplot_central_lower", env$min_envelope_central)
add("uni", "fbplot_central_upper", env$max_envelope_central)
add("uni", "fbplot_fence_lower", env$fence_lower)
add("uni", "fbplot_fence_upper", env$fence_upper)
fdafb <- fda::fbplot(t(uni), x = grid, plot = FALSE)
add("uni", "fda_depth", fdafb$depth)
add("uni", "fda_outliers", fdafb$outpoint)
add("uni", "fda_median", fdafb$medcurve)
fbt <- roahd::fbplot(fData(grid, tied), display = FALSE)
envt <- roahd:::.fbplot_fData(grid, tied, fbt$Depth, 1.5)
add("tied", "fbplot_outliers", sort(fbt$ID_outliers))
add("tied", "fbplot_central_lower", envt$min_envelope_central)
add("tied", "fbplot_central_upper", envt$max_envelope_central)
add("tied", "fbplot_central_size", sum(fbt$Depth >= quantile(fbt$Depth, 0.5)))
fdat <- fda::fbplot(t(tied), x = grid, plot = FALSE)
add("tied", "fda_outliers", fdat$outpoint)
fbm <- roahd::fbplot(mfData(grid, list(mv1, mv2)), display = FALSE)
envm <- roahd:::.fbplot_mfData(grid, list(mv1, mv2), fbm$Depth, 1.5)
add("mv", "fbplot_depth", fbm$Depth)
add("mv", "fbplot_outliers", sort(fbm$ID_outliers))
for (j in 1:2) {
    add("mv", paste0("fbplot_central_lower", j), envm$min_envelope_central[j, ])
    add("mv", paste0("fbplot_central_upper", j), envm$max_envelope_central[j, ])
}

# Outliergram, its first stage and the shift-check candidates.
og <- roahd::outliergram(fData(grid, uni), display = FALSE)
st <- roahd:::.outliergram(fData(grid, uni), shift = FALSE)
stopifnot(length(st$ID_SO) > 0, length(setdiff(og$ID_outliers, st$ID_SO)) > 0)
kept <- setdiff(seq_len(n), st$ID_SO)
low <- kept[st$MEI_data[kept] >= quantile(st$MEI_data, 0.95)]
high <- kept[st$MEI_data[kept] <= quantile(st$MEI_data, 0.05)]
checked_low <- sort(unique(apply(uni, 2, which.min)[apply(uni, 2, which.min) %in% low]))
checked_high <- sort(unique(apply(uni, 2, which.max)[apply(uni, 2, which.max) %in% high]))
for (ids in list(checked_low, checked_high)) for (id in ids) stopifnot(length(grep(id, ids)) == 1)
add("uni", "outliergram_mei", st$MEI_data)
add("uni", "outliergram_mbd", st$MBD_data)
add("uni", "outliergram_d", og$d)
add("uni", "outliergram_quartiles", c(st$Q_d1, st$Q_d3))
add("uni", "outliergram_flagged", sort(st$ID_SO))
add("uni", "outliergram_outliers", sort(og$ID_outliers))

# Magnitude-shape plots. dir_out's MCD uses cov.rob(nsamp = "best").
ms <- msplot(uni, plot = FALSE)
dir <- dir_out(uni, return_distance = TRUE)
hf <- fdaoutlier:::hardin_factor_numeric(n, 2)
add("uni", "ms_mo", ms$mean_outlyingness)
add("uni", "ms_vo", ms$var_outlyingness)
add("uni", "ms_distance", dir$distance)
add("uni", "ms_center", dir$mcd_obj$center)
add("uni", "ms_cov", dir$mcd_obj$cov)
add("uni", "ms_best", dir$mcd_obj$best)
add("uni", "ms_raw_center", colMeans(dir$ms_matrix[dir$mcd_obj$best, ]))
add("uni", "ms_raw_cov", var(dir$ms_matrix[dir$mcd_obj$best, ]))
add("uni", "ms_cutoff", hf$factor2 / hf$factor1)
add("uni", "ms_outliers", ms$outliers)
add("uni", "ms_median", ms$median_curve)
# The second component alone, with a flagged magnitude outlier.
ms2 <- msplot(mv2, plot = FALSE)
dir2 <- dir_out(mv2, return_distance = TRUE)
hf2 <- fdaoutlier:::hardin_factor_numeric(nm, 2)
stopifnot(length(ms2$outliers) > 0)
add("uni2", "ms_mo", ms2$mean_outlyingness)
add("uni2", "ms_vo", ms2$var_outlyingness)
add("uni2", "ms_distance", dir2$distance)
add("uni2", "ms_best", dir2$mcd_obj$best)
add("uni2", "ms_cutoff", hf2$factor2 / hf2$factor1)
add("uni2", "ms_outliers", ms2$outliers)
add("uni2", "ms_median", ms2$median_curve)
arr <- array(c(mv1, mv2), dim = c(nm, m, 2))
msm <- msplot(arr, n_projections = nproj, seed = seed_projections, plot = FALSE)
dirm <- dir_out(arr, n_projections = nproj, seed = seed_projections, return_distance = TRUE)
hfm <- fdaoutlier:::hardin_factor_numeric(nm, 3)
add("mv", "ms_mo", msm$mean_outlyingness)
add("mv", "ms_vo", msm$var_outlyingness)
add("mv", "ms_distance", dirm$distance)
add("mv", "ms_center", dirm$mcd_obj$center)
add("mv", "ms_cov", dirm$mcd_obj$cov)
add("mv", "ms_best", dirm$mcd_obj$best)
add("mv", "ms_cutoff", hfm$factor2 / hfm$factor1)
add("mv", "ms_outliers", msm$outliers)
add("mv", "ms_median", msm$median_curve)

# Hardin-Rocke pieces: the F quantile (deterministic in fdaoutlier) and the
# exact consistency factor c.
for (nq in list(c(24, 2), c(20, 3), c(100, 2), c(1000, 3))) {
    hn <- nq[1]; q <- nq[2]; h <- floor((hn + q + 1) / 2)
    add("hardin", paste0("quantile_", hn, "_", q), fdaoutlier:::croux_hesbroeck_asymptotic(hn, q)$factor2)
    add("hardin", paste0("c_", hn, "_", q), pchisq(qchisq(h / hn, q), q + 2) / (h / hn))
}

# Adjusted factor: robust covariance and one trial.
add("adjust", "cov_ogk_exact_qn", robustbase::covOGK(uni, sigmamu = qn_exact)$cov)
add("adjust", "qn_robustbase", c(robustbase::Qn(uni[, 1]), robustbase::Qn(uni[1:11, 2]), robustbase::Qn(qn_column)))
add("adjust", "qn_exact", c(qn_exact(uni[, 1]), qn_exact(uni[1:11, 2]), qn_exact(qn_column)))
add("adjust", "trial_root", trial$root)
add("adjust", "trial_tol", trial$estim.prec)
add("adjust", "trial_flagged", length(roahd:::.fbplot_fData(grid, sim, Depths = "MBD", Fvalue = trial$root)$ID_out))
add("adjust", "rate", rate)

path <- "test/data/m8_outliers_golden.csv"
writeLines(c(header, "# case,quantity,index,value"), path)
out$value <- sprintf("%.17g", out$value)
write.table(out, path, append = TRUE, sep = ",", row.names = FALSE, col.names = FALSE, quote = FALSE)
