# Run from the repository root: Rscript test/reference/classification_m8.R
# References: R 4.5.1; fda.usc 2.2.0 (current CRAN); MASS 7.3-65.
# Functions audited (fda.usc source): classif.np, classif.knn, S.KNN,
# predict.classif, pred2np, weights4class, Ker.unif, classif.depth,
# classif.DD, predict.classif.DD, MCR0, classif.glm2boost, classif.lda,
# classif.qda, pred2ML, pred2lda, fdata2model, create.pc.basis, depth.mode.
# https://cran.r-project.org/src/contrib/fda.usc_2.2.0.tar.gz
#
# Inputs are small simulated curves written to test/data/m8_classification_input.csv;
# the RNG seed is consumed only by R. Class codes are 1, 2 (and 3), the order
# of the sorted labels. Two classes and odd k avoid kNN vote ties, whose
# tie-break in fda.usc indexes curves by class number (see the "Clustering
# and classification" docs page). Random-projection directions are fixed here.
# The discriminant curves lie exactly in the span of sin(k pi t), k = 1, 2, 3,
# so any three-component PCA spans the same space and discriminant
# probabilities, which are invariant to invertible linear maps of the
# features, agree whatever the eigenproblem weights. CI reads the committed
# CSVs and does not need R.
suppressMessages({library(fda.usc); library(MASS)})
stopifnot(packageVersion("fda.usc") == "2.2.0")
options(digits = 17)
header <- paste("#", R.version.string, "; fda.usc", packageVersion("fda.usc"),
                "; MASS", packageVersion("MASS"), "; classification_m8.R")

set.seed(8401)
m <- 21
grid <- seq(0, 1, length.out = m)
two <- function(n, cls) {
    a <- rnorm(n, c(0, 0.35)[cls], 0.3); b <- rnorm(n, c(0.5, 0.7)[cls], 0.25)
    c <- rnorm(n, 0, 0.2)
    outer(a, rep(1, m)) + outer(b, sin(2 * pi * grid)) + outer(c, cos(2 * pi * grid)) +
        matrix(rnorm(n * m, 0, 0.05), n, m)
}
g_train <- rep(1:2, each = 20); x_train <- rbind(two(20, 1), two(20, 2))
g_test <- rep(1:2, each = 5); x_test <- rbind(two(5, 1), two(5, 2))
proj <- matrix(rnorm(10 * m), 10, m)

phi <- rbind(sin(pi * grid), sin(2 * pi * grid), sin(3 * pi * grid))
three <- function(n, cls) {
    mu <- rbind(c(1, 0, 0), c(0, 1, 0.3), c(0.4, 0.4, 1))[cls, ]
    sd <- rbind(c(0.6, 0.4, 0.3), c(0.4, 0.6, 0.3), c(0.5, 0.3, 0.6))[cls, ]
    coefs <- sweep(matrix(rnorm(n * 3), n, 3) * rep(sd, each = n), 2, mu, "+")
    coefs %*% phi
}
g_lda <- rep(1:3, each = 15); x_lda <- rbind(three(15, 1), three(15, 2), three(15, 3))
g_lda_test <- rep(1:3, each = 3); x_lda_test <- rbind(three(3, 1), three(3, 2), three(3, 3))

inputs <- list(grid = matrix(grid, 1), g_train = matrix(g_train, 1), x_train = x_train,
    g_test = matrix(g_test, 1), x_test = x_test, proj = proj, g_lda = matrix(g_lda, 1),
    x_lda = x_lda, g_lda_test = matrix(g_lda_test, 1), x_lda_test = x_lda_test)
input_rows <- do.call(rbind, lapply(names(inputs), function(name) {
    x <- inputs[[name]]
    data.frame(dataset = name, row = as.vector(row(x)), col = as.vector(col(x)), value = as.vector(x))
}))
input_path <- "test/data/m8_classification_input.csv"
writeLines(c(header, "# dataset,row,col,value"), input_path)
input_rows$value <- sprintf("%.17g", input_rows$value)
write.table(input_rows, input_path, append = TRUE, sep = ",", row.names = FALSE,
            col.names = FALSE, quote = FALSE)

out <- NULL
add <- function(case, quantity, value) {
    value <- as.matrix(value)
    out <<- rbind(out, data.frame(case = case, quantity = quantity, row = as.vector(row(value)),
        col = as.vector(col(value)), value = as.vector(unname(value))))
}
code <- function(f) as.integer(f)
fd <- function(x, t = grid) fdata(x, t)
group <- factor(g_train)

# kNN with fda.usc's leave-one-out choice among k = 3, 5, ..., 15.
invisible(capture.output(knn <- classif.knn(group, fd(x_train))))
knn_new <- predict(knn, fd(x_test), type = "probs")
add("knn", "loo_error", knn$gcv)
add("knn", "k", knn$h.opt)
add("knn", "loo_probability", knn$prob.group)
add("knn", "loo_class", code(knn$group.est))
add("knn", "probability", knn_new$prob.group)
add("knn", "class", code(knn_new$group.pred))

# Maximum depth: depth of every curve with respect to each class.
for (d in c("FM", "mode", "RP")) {
    pd <- if (d == "RP") list(proj = fd(proj)) else list()
    fit <- classif.depth(group, fd(x_train), depth = d, par.depth = pd)
    new <- classif.depth(group, fd(x_train), fd(x_test), depth = d, par.depth = pd)
    add(paste0("maxdepth_", d), "depth", fit$dep)
    add(paste0("maxdepth_", d), "class", code(fit$group.est))
    add(paste0("maxdepth_", d), "new_depth", new$dep)
    add(paste0("maxdepth_", d), "new_class", code(new$group.pred))
}

# DD-classifier on Fraiman-Muniz depths with the glm, lda, qda and knn rules.
ctrl <- list(draw = FALSE)
# classif.DD names the depth columns after the deparsed argument.
xtr <- fd(x_train)
dd_glm <- classif.DD(group, xtr, depth = "FM", classif = "glm", control = ctrl)
dd_new <- predict(dd_glm, fd(x_test), type = "dep")
beta <- coef(dd_glm$fit$fit[[1]])
add("dd_glm", "depth", dd_glm$dep)
add("dd_glm", "new_depth", dd_new$dep)
# fda.usc's glm stops once the relative deviance change is below 1e-8, which
# leaves its coefficients about 1e-8 (relative) from the maximum-likelihood
# estimate; the same model refitted to convergence gives the coefficients.
yy <- as.integer(group == levels(group)[2])
converged <- glm(yy ~ dd_glm$dep, family = binomial,
                 control = glm.control(epsilon = 1e-15, maxit = 100))
stopifnot(max(abs(coef(converged) - beta)) < 1e-6)
add("dd_glm", "coef", coef(converged))
add("dd_glm", "fitted", dd_glm$fit$prob.group)
add("dd_glm", "class", code(dd_glm$group.est))
add("dd_glm", "new_fitted", plogis(cbind(1, dd_new$dep) %*% beta))
add("dd_glm", "new_class", code(dd_new$group.pred))
for (rule in c("lda", "qda")) {
    dd <- classif.DD(group, xtr, depth = "FM", classif = rule, control = ctrl)
    add(paste0("dd_", rule), "posterior", predict(dd$fit)$posterior)
    add(paste0("dd_", rule), "class", code(dd$group.est))
    add(paste0("dd_", rule), "new_posterior", predict(dd$fit, dd_new$dep)$posterior)
    add(paste0("dd_", rule), "new_class", code(predict(dd, fd(x_test))))
}
invisible(capture.output(dd_knn <- classif.DD(group, xtr, depth = "FM", classif = "knn",
    control = ctrl)))
add("dd_knn", "loo_error", dd_knn$fit$gcv)
add("dd_knn", "k", dd_knn$fit$h.opt)
add("dd_knn", "loo_probability", dd_knn$fit$prob.group)
add("dd_knn", "new_probability", predict(dd_knn$fit, dd_new$dep, type = "probs")$prob.group)
# Modal depth: classif.DD passes the pooled-sample bandwidth to every class.
dd_mode <- classif.DD(group, xtr, depth = "mode", classif = "lda", control = ctrl)
add("dd_mode", "depth", dd_mode$dep)
add("dd_mode", "bandwidth", dd_mode$par.depth[[1]]$h[1])
add("dd_mode", "new_depth", predict(dd_mode, fd(x_test), type = "dep")$dep)

# Functional LDA and QDA on three principal-component scores. fda.usc's
# default weights = "equal" give every class prior 1/3.
glda <- factor(g_lda)
ldata <- list(df = data.frame(y = glda), X = fd(x_lda))
basis <- list(X = create.pc.basis(fd(x_lda), l = 1:3))
for (rule in c("lda", "qda")) {
    fit <- do.call(paste0("classif.", rule), list(y ~ X, data = ldata, basis.x = basis))
    new <- predict(fit, list(X = fd(x_lda_test)), type = "probs")
    add(rule, "prior", fit$prior)
    add(rule, "posterior", fit$prob.group)
    add(rule, "class", code(fit$group.est))
    add(rule, "new_posterior", new$prob.group)
    add(rule, "new_class", code(new$group.pred))
}

path <- "test/data/m8_classification_golden.csv"
writeLines(c(header, "# case,quantity,row,col,value"), path)
out$value <- sprintf("%.17g", out$value)
write.table(out, path, append = TRUE, sep = ",", row.names = FALSE, col.names = FALSE, quote = FALSE)
