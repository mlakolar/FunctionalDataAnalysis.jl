# Run after mfpca_m4.jl, from the repository root:
# Rscript test/reference/mfpca_m4.R
# Reference: R 4.5.1, MFPCA 1.3-11, funData 1.3-9, mgcv 1.9-3.
# Strict given-expansion comparison and separate cross-method uFPCA comparison.
library(MFPCA)
stopifnot(packageVersion("MFPCA") == "1.3.11")
options(digits = 17)
input <- read.csv("test/data/m4_mfpca_input.csv", comment.char = "#", header = FALSE,
                  col.names = c("quantity", "node", "row", "column", "value"))
get_matrix <- function(quantity, node) {
    a <- input[input$quantity == quantity & input$node == node, ]
    matrix(a$value, nrow = max(a$row), ncol = max(a$column))
}
grids <- lapply(1:3, function(j) as.vector(get_matrix("grid", j)))
data <- multiFunData(lapply(1:3, function(j) funData(grids[[j]], get_matrix("value", j))))
given <- lapply(1:3, function(j) list(type = "given", ortho = TRUE,
    functions = funData(grids[[j]], t(get_matrix("function", j))),
    scores = get_matrix("score", j)))
cross <- lapply(1:3, function(j) list(type = "uFPCA", npc = 2))
path <- "test/data/m4_mfpca.csv"
writeLines(paste("#", R.version.string, "; MFPCA", packageVersion("MFPCA"),
    "; funData", packageVersion("funData"), "; mgcv", packageVersion("mgcv"),
    "; mode,quantity,node,row,column,value; M=4, exact eigen, bootstrap=FALSE, fit=TRUE"), path)
write_matrix <- function(mode, quantity, node, a) {
    a <- as.matrix(a)
    indices <- expand.grid(row = seq_len(nrow(a)), column = seq_len(ncol(a)))
    write.table(data.frame(mode, quantity, node, indices, value = as.vector(a)), path,
                append = TRUE, sep = ",", row.names = FALSE, col.names = FALSE, quote = FALSE)
}
write_result <- function(mode, result, p, normalization = FALSE) {
    # R does not orient eigenvectors. Apply FDA's largest coefficient rule
    # for given expansions. Cross-method signs use node functions in Julia,
    # because R's own univariate bases may have different signs/orientations.
    signs <- apply(result$vectors, 2, function(c) if (c[which.max(abs(c))] < 0) -1 else 1)
    write_matrix(mode, "eigenvalue", 0, result$values)
    write_matrix(mode, "score", 0, sweep(result$scores, 2, signs, "*"))
    write_matrix(mode, "coefficient", 0, sweep(result$vectors, 2, signs, "*"))
    if (normalization) write_matrix(mode, "normalization_factor", 0, result$normFactors)
    for (j in seq_len(p)) {
        write_matrix(mode, "function", j, sweep(t(result$functions[[j]]@X), 2, signs, "*"))
        write_matrix(mode, "fit", j, result$fit[[j]]@X)
    }
}
for (mode in c("given_unit", "given_weighted", "cross_method")) {
    weights <- if (mode == "given_unit") rep(1, 3) else c(0.7, 1.5, 2.2)
    result <- MFPCA(data, M = 4, weights = weights,
        uniExpansions = if (mode == "cross_method") cross else given,
        fit = TRUE, approx.eigen = FALSE, bootstrap = FALSE, verbose = FALSE)
    write_result(mode, result, 3)
}

# Test the exact route's cov(allScores) centering and its *uncentered*
# normFactors, scores and function coefficients using actual PACE scores.
input <- read.csv("test/data/m4_mfpca_pace_input.csv", comment.char = "#", header = FALSE,
                  col.names = c("quantity", "node", "row", "column", "value"))
grids <- lapply(1:2, function(j) as.vector(get_matrix("grid", j)))
data <- multiFunData(lapply(1:2, function(j) funData(grids[[j]], get_matrix("value", j))))
given <- lapply(1:2, function(j) list(type = "given", ortho = TRUE,
    functions = funData(grids[[j]], t(get_matrix("function", j))),
    scores = get_matrix("score", j)))
stopifnot(abs(mean(given[[2]]$scores)) > 1)
path <- "test/data/m4_mfpca_pace.csv"
writeLines(paste("#", R.version.string, "; MFPCA", packageVersion("MFPCA"),
    "; funData", packageVersion("funData"), "; mgcv", packageVersion("mgcv"),
    "; mode,quantity,node,row,column,value; M=2, exact eigen, bootstrap=FALSE, fit=TRUE; fits rebased to supplied FDA means"), path)
for (mode in c("given_pace_unit", "given_pace_weighted")) {
    weights <- if (mode == "given_pace_unit") rep(1, 2) else c(0.8, 1.4)
    # givenBasis warns that these intentionally noncentered scores are not
    # demeaned. They must remain uncentered to exercise calcMFPCA faithfully.
    result <- MFPCA(data, M = 2, weights = weights, uniExpansions = given,
        fit = TRUE, approx.eigen = FALSE, bootstrap = FALSE, verbose = FALSE)
    for (j in 1:2) {
        # type="given" uses empirical means. Match FDA's supplied first-step
        # means when comparing reconstruction, without changing step two.
        result$fit[[j]]@X <- sweep(result$fit[[j]]@X, 2,
            as.vector(result$meanFunction[[j]]@X), "-")
        result$fit[[j]]@X <- sweep(result$fit[[j]]@X, 2,
            as.vector(get_matrix("mean", j)), "+")
    }
    write_result(mode, result, 2, normalization = TRUE)
}
