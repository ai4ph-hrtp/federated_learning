### LOAD PACKAGES ###

suppressPackageStartupMessages({
  library(rlang)
  library(ggplot2)
  library(dplyr)
  library(yardstick)
  library(randomForest)
  library(xgboost)
})

### DATA PROCESSING ###

split_sites <- function(df_train,
                        df_test,
                        site_col,
                        model,
                        formula,
                        target_col) {
  df_test$.row_id <- seq_len(nrow(df_test)) # for ordering predictions
  factor_cols <- setdiff(names(Filter(is.factor, df_train)), site_col)
  train_split <- split(df_train, df_train[[site_col]])
  test_split  <- split(df_test, df_test[[site_col]])
  sites <- intersect(names(train_split), names(test_split))
  canonical_levels <- lapply(factor_cols, function(col)
    levels(df_train[[col]]))
  names(canonical_levels) <- factor_cols
  missing_levels <- list()
  for (site in sites) {
    train_site <- train_split[[site]]
    for (col in factor_cols) {
      expected_levels <- canonical_levels[[col]]
      present_levels  <- levels(droplevels(train_site[[col]]))
      missing <- setdiff(expected_levels, present_levels)
      if (length(missing) > 0) {
        msg <- sprintf(
          "Train site '%s', column '%s' missing levels: %s",
          site,
          col,
          paste(missing, collapse = ", ")
        )
        missing_levels[[length(missing_levels) + 1]] <- msg
      }
    }
  }
  if (length(missing_levels) > 0)
    stop(paste(
      c(
        "Some sites are missing factor levels in training data:",
        missing_levels
      ),
      collapse = "\n"
    ))
  make_xgb_data <- function(df) {
    X <- model.matrix(terms(formula, data = df), data = df)[, -1, drop = FALSE]
    y <- as.numeric(as.character(df[[target_col]]))
    list(data = X, label = y)
  }
  site_data <- lapply(sites, function(site) {
    train <- train_split[[site]]
    test  <- test_split[[site]]
    res <- list(train = train,
                test = test,
                site_id = site)
    if (model == "xgboost") {
      train_xgb <- make_xgb_data(train)
      test_xgb  <- make_xgb_data(test)
      res$train_data  <- train_xgb$data
      res$train_label <- train_xgb$label
      res$test_data   <- test_xgb$data
      res$test_label  <- test_xgb$label
    }
    res
  })
  names(site_data) <- sites
  site_data
}

### MODEL FITTING ###

fit_model <- function(x, model, formula, target_col, ...) {
  model <- match.arg(model, choices = c("logistic", "randomForest", "xgboost"))
  if (model == "logistic") {
    glm(formula, data = x$train, family = binomial(), ...)
  } else if (model == "randomForest") {
    randomForest::randomForest(formula, data = x$train, ...)
  } else if (model == "xgboost") {
    args <- list(...)
    if (is.null(args$nrounds))
      args$nrounds <- 100
    if (is.null(args$verbose))
      args$verbose <- 0
    do.call(xgboost::xgboost, c(
      list(
        data = x$train_data,
        label = x$train_label,
        objective = "binary:logistic"
      ),
      args
    ))
  }
}

### FEDERATED LEARNING UTILITIES ###

compute_site_weights <- function(site_splits) {
  counts <- sapply(site_splits, \(site) nrow(site$train))
  weights <- counts / sum(counts)
  list(weights = weights, counts = counts)
}

print_site_weights <- function(site_names, weights, counts) {
  lines <- sprintf("  %s: %.3f (%d samples)", site_names, weights, counts)
  cat("Site weights:\n", paste(lines, collapse = "\n"), "\n")
}

aggregate_coefficients <- function(models, weights) {
  global_coef <- numeric(length(models[[1]]$coefficients))
  for (i in seq_along(models)) {
    global_coef <- global_coef + weights[i] * models[[i]]$coefficients
  }
  global_coef
}

evaluate_model <- function(model_fit, test_data, model_type, truth) {
  prob <- switch(
    model_type,
    logistic = predict(model_fit, test_data, type = "response"),
    randomForest = predict(model_fit, test_data, type = "prob")[, 2],
    xgboost = as.numeric(predict(model_fit, test_data))
  )
  pred <- factor(ifelse(prob >= 0.5, 1, 0), levels = levels(truth))
  acc  <- accuracy_vec(truth = truth, estimate = pred)
  list(
    accuracy = acc,
    predictions = pred,
    probabilities = prob
  )
}

report_performance_summary <- function(perf_table, weights) {
  cat("\nPerformance summary:\n")
  for (site_name in rownames(perf_table)) {
    cat(
      sprintf(
        "  %s: Initial = %.4f, Updated = %.4f, Δ = %+0.4f\n",
        site_name,
        perf_table[site_name, "initial"],
        perf_table[site_name, "updated"],
        perf_table[site_name, "delta"]
      )
    )
  }
  weighted_initial <- sum(weights * perf_table[, "initial"])
  weighted_updated <- sum(weights * perf_table[, "updated"])
  weighted_delta <- weighted_updated - weighted_initial
  cat(
    sprintf(
      "\nOverall weighted accuracy: Initial = %.4f, Updated = %.4f, Δ = %+0.4f\n",
      weighted_initial,
      weighted_updated,
      weighted_delta
    )
  )
  invisible(c(
    initial = weighted_initial,
    updated = weighted_updated,
    delta = weighted_delta
  ))
}

collect_ordered_results <- function(results, which) {
  df <- do.call(rbind, lapply(results, function(x)
    data.frame(
      row_id = x$row_id,
      truth = x$truth,
      prediction = x[[which]]$predictions,
      probability = x[[which]]$probabilities
    )))
  df[order(df$row_id), ]
}

### FEDERATED AVERAGING ###

fedavg <- function(site_splits, target_col, formula) {
  cat("FedAvg on", length(site_splits), "sites\n\n")
  site_weights <- compute_site_weights(site_splits)
  print_site_weights(names(site_splits),
                     site_weights$weights,
                     site_weights$counts)
  site_names <- names(site_splits)
  local_models <- list()
  local_models_initial <- list()
  results <- list()
  cat("\nTraining local models and evaluating initial performance...\n")
  for (site_name in site_names) {
    site <- site_splits[[site_name]]
    model <- fit_model(site, model = "logistic", formula = formula)
    test_data <- site$test
    test_truth <- test_data[[target_col]]
    eval_initial <- evaluate_model(model, test_data, "logistic", test_truth)
    local_models[[site_name]] <- model
    local_models_initial[[site_name]] <- model
    results[[site_name]] <- list(truth = test_truth,
                                 row_id = test_data$.row_id,
                                 initial = eval_initial)
    cat(sprintf(
      "  %s: initial accuracy = %.4f\n",
      site_name,
      eval_initial$accuracy
    ))
  }
  cat("\nAggregating global model coefficients...\n")
  global_coef <- aggregate_coefficients(local_models, site_weights$weights)
  cat(
    "Updating local models with global coefficients and evaluating updated performance...\n"
  )
  for (site_name in site_names) {
    site <- site_splits[[site_name]]
    test_data <- site$test
    test_truth <- test_data[[target_col]]
    local_models[[site_name]]$coefficients <- global_coef
    eval_updated <- evaluate_model(local_models[[site_name]], test_data, "logistic", test_truth)
    results[[site_name]]$updated <- eval_updated
    cat(
      sprintf(
        "  %s: updated accuracy = %.4f (Δ = %+0.4f)\n",
        site_name,
        eval_updated$accuracy,
        eval_updated$accuracy - results[[site_name]]$initial$accuracy
      )
    )
  }
  perf_table <- do.call(rbind, lapply(site_names, function(site_name) {
    initial_acc <- results[[site_name]]$initial$accuracy
    updated_acc <- results[[site_name]]$updated$accuracy
    delta <- updated_acc - initial_acc
    c(initial = initial_acc,
      updated = updated_acc,
      delta = delta)
  }))
  rownames(perf_table) <- site_names
  overall <- report_performance_summary(perf_table, site_weights$weights)
  df_initial <- collect_ordered_results(results, "initial")
  df_updated <- collect_ordered_results(results, "updated")
  preds_init <- lapply(results, \(x) x$initial$predictions)
  preds_updt <- lapply(results, \(x) x$updated$predictions)
  probs_init <- lapply(results, \(x) x$initial$probabilities)
  probs_updt <- lapply(results, \(x) x$updated$probabilities)
  invisible(
    list(
      algorithm = "FedAvg",
      model = "logistic",
      site_names = site_names,
      site_weights = site_weights$weights,
      models_initial = local_models_initial,
      models_updated = local_models,
      accuracy_initial = perf_table[, "initial"],
      accuracy_updated = perf_table[, "updated"],
      accuracy_overall_initial = overall["initial"],
      accuracy_overall_updated = overall["updated"],
      truth_overall = df_initial$truth,
      predictions_initial = preds_init,
      predictions_updated = preds_updt,
      probabilities_initial = probs_init,
      probabilities_updated = probs_updt,
      predictions_overall_initial = df_initial$prediction,
      predictions_overall_updated = df_updated$prediction,
      probabilities_overall_initial = df_initial$probability,
      probabilities_overall_updated = df_updated$probability
    )
  )
}

### DISTRIBUTED ENSEMBLE ###

distributed_ensemble <- function(site_splits,
                                 model,
                                 target_col,
                                 formula,
                                 ...) {
  cat(sprintf(
    "Distributed Ensemble: %s on %d sites\n\n",
    model,
    length(site_splits)
  ))
  weights_info <- compute_site_weights(site_splits)
  weights <- weights_info$weights
  counts <- weights_info$counts
  site_names <- names(site_splits)
  print_site_weights(site_names, weights, counts)
  site_models <- list()
  results <- list()
  cat("\nTraining individual site models and evaluating individual performance...\n")
  for (site_name in site_names) {
    site <- site_splits[[site_name]]
    trained_model <- fit_model(
      x = site,
      model = model,
      formula = formula,
      target_col = target_col,
      ...
    )
    site_models[[site_name]] <- trained_model
    test_data <- if (model == "xgboost")
      site$test_data
    else
      site$test
    test_label <- if (model == "xgboost")
      as.factor(site$test_label)
    else
      site$test[[target_col]]
    ind_eval <- evaluate_model(
      model_fit = trained_model,
      test_data = test_data,
      model_type = model,
      truth = test_label
    )
    results[[site_name]] <- list(
      truth = test_label,
      row_id = site$test$.row_id,
      initial = ind_eval
    )
    cat(sprintf(
      "  %s: initial accuracy = %.4f\n",
      site_name,
      ind_eval$accuracy
    ))
  }
  cat("\nEvaluating updated performance (weighted ensemble)...\n")
  for (site_name in site_names) {
    site <- site_splits[[site_name]]
    test_data <- if (model == "xgboost")
      site$test_data
    else
      site$test
    test_label <- if (model == "xgboost")
      as.factor(site$test_label)
    else
      site$test[[target_col]]
    n_test <- nrow(test_data)
    ensemble_probs <- vapply(site_models, function(m) {
      if (model == "xgboost")
        as.numeric(predict(m, test_data))
      else
        predict(m, test_data, type = "prob")[, 2]
    }, numeric(n_test))
    weighted_probs <- drop(ensemble_probs %*% weights)
    ensemble_pred <- factor(ifelse(weighted_probs >= 0.5, 1, 0), levels = levels(test_label))
    acc_ensemble <- mean(ensemble_pred == test_label)
    results[[site_name]]$updated <- list(
      accuracy = acc_ensemble,
      predictions = ensemble_pred,
      probabilities = weighted_probs
    )
    cat(
      sprintf(
        "  %s: updated accuracy = %.4f (Δ = %+0.4f)\n",
        site_name,
        acc_ensemble,
        acc_ensemble - results[[site_name]]$initial$accuracy
      )
    )
  }
  perf_table <- do.call(rbind, lapply(site_names, function(site_name) {
    initial_acc <- results[[site_name]]$initial$accuracy
    ensemble_acc <- results[[site_name]]$updated$accuracy
    delta <- ensemble_acc - results[[site_name]]$initial$accuracy
    c(initial = initial_acc,
      updated = ensemble_acc,
      delta = delta)
  }))
  rownames(perf_table) <- site_names
  overall <- report_performance_summary(perf_table, weights)
  df_initial <- collect_ordered_results(results, "initial")
  df_updated <- collect_ordered_results(results, "updated")
  preds_init <- lapply(results, \(x) x$initial$predictions)
  preds_updt <- lapply(results, \(x) x$updated$predictions)
  probs_init <- lapply(results, \(x) x$initial$probabilities)
  probs_updt <- lapply(results, \(x) x$updated$probabilities)
  invisible(
    list(
      algorithm = "Distributed Ensemble",
      model = model,
      site_names = site_names,
      site_weights = weights,
      models_initial = site_models,
      accuracy_initial = perf_table[, "initial"],
      accuracy_updated = perf_table[, "updated"],
      accuracy_overall_initial = overall["initial"],
      accuracy_overall_updated = overall["updated"],
      truth_overall = df_initial$truth,
      predictions_initial = preds_init,
      predictions_updated = preds_updt,
      probabilities_initial = probs_init,
      probabilities_updated = probs_updt,
      predictions_overall_initial = df_initial$prediction,
      predictions_overall_updated = df_updated$prediction,
      probabilities_overall_initial = df_initial$probability,
      probabilities_overall_updated = df_updated$probability
    )
  )
}

### MAIN FUNCTION ###

fed_learn <- function(df_train,
                      df_test,
                      target_col,
                      site_col = "site",
                      formula,
                      algorithm = c("FedAvg", "Distributed Ensemble"),
                      model = c("logistic", "randomForest", "xgboost"),
                      ...) {
  # Validate required arguments
  if (missing(df_train) ||
      missing(df_test))
    stop("Both training and testing datasets must be provided.")
  if (missing(target_col))
    stop("You must specify the target column name.")
  if (missing(formula))
    stop("You must specify a model formula.")
  algorithm <- match.arg(algorithm)
  model <- match.arg(model)
  if (algorithm == "FedAvg" && model != "logistic") {
    stop("FedAvg supports only logistic regression.")
  }
  if (algorithm == "Distributed Ensemble" &&
      !(model %in% c("randomForest", "xgboost"))) {
    stop("Distributed Ensemble supports only randomForest or xgboost.")
  }
  # Validate target and site columns
  for (col in c(target_col, site_col)) {
    if (!col %in% names(df_train) || !col %in% names(df_test)) {
      stop(sprintf(
        "Column '%s' must exist in both training and testing datasets.",
        col
      ))
    }
  }
  # Check target_col is factor(0/1) and consistent
  if (!is.factor(df_train[[target_col]]) ||
      !is.factor(df_test[[target_col]])) {
    stop(sprintf(
      "Target column '%s' must be a factor in both datasets.",
      target_col
    ))
  }
  train_levels <- levels(df_train[[target_col]])
  test_levels  <- levels(df_test[[target_col]])
  if (!identical(train_levels, c("0", "1")) ||
      !identical(test_levels, c("0", "1"))) {
    stop(
      sprintf(
        "Target column '%s' must have levels exactly c('0', '1') in both datasets.",
        target_col
      )
    )
  }
  # Check sites are consistent
  train_sites <- sort(unique(df_train[[site_col]]))
  test_sites  <- sort(unique(df_test[[site_col]]))
  if (!identical(train_sites, test_sites)) {
    stop("Training and testing datasets must contain the same sites.")
  }
  # Validate columns & types
  common_cols <- intersect(names(df_train), names(df_test))
  train_classes <- sapply(df_train[common_cols], class)
  test_classes  <- sapply(df_test[common_cols], class)
  if (!all(train_classes == test_classes)) {
    stop("Column types must match between training and testing datasets.")
  }
  # Non-site categorical variables must be factor, not character
  cols <- setdiff(common_cols, c(target_col, site_col))
  for (df in list(df_train, df_test)) {
    invalid <- cols[!sapply(df[cols], \(x) is.numeric(x) ||
                              is.factor(x))]
    if (length(invalid) > 0) {
      stop(sprintf(
        "Columns must be numeric or factor. Invalid columns: %s",
        paste(invalid, collapse = ", ")
      ))
    }
  }
  # Split sites
  site_splits <- split_sites(
    df_train,
    df_test,
    target_col = target_col,
    site_col = site_col,
    model = model,
    formula = formula
  )
  # Dispatch to algorithm
  if (algorithm == "FedAvg") {
    fedavg(site_splits,
           target_col = target_col,
           formula = formula,
           ...)
  } else {
    distributed_ensemble(
      site_splits,
      model = model,
      target_col = target_col,
      formula = formula,
      ...
    )
  }
}
