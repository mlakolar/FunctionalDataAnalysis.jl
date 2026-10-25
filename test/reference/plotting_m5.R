# R 4.5.1; fdapace 0.6.0. Run: Rscript test/reference/plotting_m5.R
# Deterministic design counts from the audited helper used by CreateDesignPlot.
stopifnot(as.character(packageVersion("fdapace")) == "0.6.0")
times <- list(c(0, .1, .7), c(.1, .4, 1), c(0, .4, .7, 1), .4)
grid <- sort(unique(unlist(times)))
rows <- do.call(rbind, lapply(c(FALSE, TRUE), function(diagonal) {
    counts <- fdapace:::DesignPlotCount(times, grid, !diagonal, TRUE)
    pairs <- which(counts != 0, arr.ind = TRUE)
    data.frame(diagonal = as.integer(diagonal), x = grid[pairs[, 1]],
               y = grid[pairs[, 2]], count = counts[pairs])
}))
rows <- rows[order(rows$diagonal, rows$x, rows$y), ]
path <- "test/data/m5_design.csv"
writeLines(sprintf("# %s; fdapace %s; reference/plotting_m5.R",
                   R.version.string, packageVersion("fdapace")), path)
write.table(rows, path, append = TRUE, sep = ",", row.names = FALSE,
            col.names = FALSE, quote = FALSE)
