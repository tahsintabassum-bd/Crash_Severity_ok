install.packages(c("tidyverse", "fastDummies", "data.table", "GGally", "corrplot", "skimr", 
                   "caret", "xgboost", "e1071", "randomForest", "glmnet", 
                   "pROC", "MLmetrics", "janitor", "plotly", "vroom"))

library(tidyverse)
library(forcats)
library(caret)
library(fastDummies)
library(xgboost)
library(dplyr)
library(pROC)
library(ggplot2)

OkData <- read.csv("OkData.csv")

print("Dataset successfully imported.")
print(unique(OkData$CRASH_SEVERITY))
str(OkData)

print("Missing values per column:")
print(colSums(is.na(OkData)))

columns_to_drop <- c("INJ_TYPE_4", "INJ_TYPE_5","INJ_TYPE_4","INJ_TYPE_3","INJ_TYPE_2","INJ_TYPE","PERSON_CONDITION","PERSON_ID_INJURY","CRASH_DRUG_RELATED", "BAC",
                     "DOCID", "COUNTY", "CITY", "AGENCY", "STREET_HIGHWAY",
                     "INTERSECTING_ROAD", "ADDRESS", "TRANSPORTED_BY",
                     "MEDICAL_FACILITY", "OHP_NON","HOUR4", "HOUR8", 
                     "HOUR6", "LONGITUDE", "LATITUDE", "x", "y", "DATE", 
                     "ObjectId", "WORKZONE_TYPE", "WORKZONE_LOCATION","INJURY_SEVERITY")
OkData <- OkData[, !(names(OkData) %in% columns_to_drop)]

print("Structure after dropping columns:")
str(OkData)

numeric_columns <- sapply(OkData, is.numeric)

OkData[, numeric_columns] <- lapply(OkData[, numeric_columns], function(x) {
  ifelse(is.na(x), median(x, na.rm = TRUE), x)
})

categorical_columns <- names(OkData)[sapply(OkData, is.character) | sapply(OkData, is.factor)]

OkData[, categorical_columns] <- lapply(OkData[, categorical_columns], function(x) {
  ifelse(is.na(x), names(sort(table(x), decreasing = TRUE))[1], x)
})

print("Missing values after handling:")
print(colSums(is.na(OkData)))

OkData$CRASH_SEVERITY <- as.factor(OkData$CRASH_SEVERITY)
OkData$CRASH_SEVERITY <- factor(OkData$CRASH_SEVERITY, levels = c("None","Possible","Minor","Serious","Fatal"))
OkData$CRASH_SEVERITY <- as.integer(OkData$CRASH_SEVERITY)
print("CRASH_SEVERITY successfully encoded:")
print(table(OkData$CRASH_SEVERITY))

categorical_columns <- setdiff(categorical_columns, "CRASH_SEVERITY")
if (length(categorical_columns) > 0) {
  OkData <- dummy_cols(OkData, 
                       select_columns = categorical_columns, 
                       remove_first_dummy = TRUE, 
                       remove_selected_columns = TRUE)
}

print("Dataset structure after one-hot encoding:")
str(OkData)

print("Target variable distribution:")
print(table(OkData$CRASH_SEVERITY))
print("Proportions:")
print(prop.table(table(OkData$CRASH_SEVERITY)))

target_variable <- "CRASH_SEVERITY"
X <- OkData[, !(names(OkData) %in% target_variable)]
y <- as.numeric(as.factor(OkData$CRASH_SEVERITY)) - 1

set.seed(42)
train_index <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X[train_index, ]
y_train <- y[train_index]
X_test <- X[-train_index, ]
y_test <- y[-train_index]

dtrain <- xgb.DMatrix(data = as.matrix(X_train), label = y_train)
dtest <- xgb.DMatrix(data = as.matrix(X_test), label = y_test)

params <- list(
  objective = "multi:softprob",  # Changed to "softprob" for probability output
  num_class = length(unique(y)),
  eval_metric = "mlogloss",
  eta = 0.1,
  max_depth = 6,
  subsample = 0.8,
  colsample_bytree = 0.8,
  min_child_weight = 1
)

set.seed(42)
xgb_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 100,
  watchlist = list(train = dtrain, eval = dtest),
  verbose = 1,
  print_every_n = 10
)

# Obtain predicted probabilities
predicted_probabilities <- predict(xgb_model, newdata = dtest, reshape = TRUE)
colnames(predicted_probabilities) <- levels(OkData$CRASH_SEVERITY)

# Compute ROC for each class
roc_curves <- lapply(1:ncol(predicted_probabilities), function(class_index) {
  roc(
    response = as.numeric(y_test + 1) == class_index,
    predictor = predicted_probabilities[, class_index],
    quiet = TRUE  # Suppress messages
  )
})

# Prepare data for ggplot
roc_data <- do.call(rbind, lapply(seq_along(roc_curves), function(index) {
  roc_info <- roc_curves[[index]]
  data.frame(
    TPR = rev(roc_info$sensitivities),
    FPR = rev(1 - roc_info$specificities),
    Class = colnames(predicted_probabilities)[index]
  )
}))

# Plot the ROC Curves using ggplot
roc_curve <- ggplot(roc_data, aes(x = FPR, y = TPR, color = Class)) +
  geom_line() +
  labs(
    title = "AUC-ROC Curves for Crash Severity Classes",
    x = "1 - Specificity (False Positive Rate)",
    y = "Sensitivity (True Positive Rate)"
  ) +
  theme_minimal() +
  scale_color_brewer(palette = "Set1")

print(roc_curve)

# Confusion Matrix
predicted_classes <- apply(predicted_probabilities, 1, which.max)
conf_matrix <- confusionMatrix(as.factor(predicted_classes), as.factor(y_test + 1))
print("Model Accuracy:")
print(conf_matrix$overall['Accuracy'])

# Confusion Matrix as a heatmap
conf_matrix_df <- as.data.frame(as.table(conf_matrix$table))
conf_matrix_plot <- ggplot(conf_matrix_df, aes(Reference, Prediction)) +
  geom_tile(aes(fill = Freq), color = "white") +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  geom_text(aes(label = Freq), color = "black", size = 4) +
  labs(
    title = "Confusion Matrix Heatmap",
    x = "Actual Class",
    y = "Predicted Class",
    fill = "Frequency"
  ) +
  theme_minimal()

print(conf_matrix_plot)

# Feature Importance
importance_matrix <- xgb.importance(model = xgb_model)
print("Feature Importance:")
print(importance_matrix)

# Plot top N features
top_n_features <- 20
xgb.plot.importance(importance_matrix, top_n = top_n_features)

xgb.save(xgb_model, "xgboost_crash_severity_model.model")

library(pROC)
library(MLmetrics)

# Assuming `predicted_probabilities` and `y_test` from your model
# `predicted_probabilities` is a matrix of probabilities
# `y_test` is the true class labels (numeric format starting from 0 or 1)

# Convert `y_test` to one-hot encoding for multi-class metrics
num_classes <- ncol(predicted_probabilities)
y_test_onehot <- model.matrix(~ 0 + as.factor(y_test))

# 1. AUC for each class
auc_values <- sapply(1:num_classes, function(class_index) {
  roc_curve <- roc(
    response = y_test_onehot[, class_index],
    predictor = predicted_probabilities[, class_index],
    quiet = TRUE
  )
  auc(roc_curve)
})

# Print AUC for each class
print("AUC for each class:")
print(auc_values)

# 2. F1 Score (and F-measure, which is equivalent to F1 Score) for each class
predicted_classes <- apply(predicted_probabilities, 1, which.max)
f1_scores <- sapply(1:num_classes, function(class_index) {
  Precision <- Precision(y_pred = predicted_classes, y_true = y_test + 1, positive = class_index)
  Recall <- Recall(y_pred = predicted_classes, y_true = y_test + 1, positive = class_index)
  if (Precision + Recall > 0) {
    2 * Precision * Recall / (Precision + Recall)  # F1 Score formula
  } else {
    0
  }
})

# Print F1 Scores
print("F1 Score for each class:")
print(f1_scores)

# 3. G-mean
conf_matrix <- table(Predicted = predicted_classes, Actual = y_test + 1)
sensitivity <- diag(conf_matrix) / rowSums(conf_matrix)  # Recall
specificity <- diag(conf_matrix) / colSums(conf_matrix)  # Specificity
g_mean <- sqrt(sensitivity * specificity)

# Print G-mean
print("G-mean for each class:")
print(g_mean)

# Summary Metrics (average across all classes)
overall_auc <- mean(auc_values)
overall_f1 <- mean(f1_scores, na.rm = TRUE)
overall_gmean <- mean(g_mean, na.rm = TRUE)

print("Overall Metrics:")
print(paste("Average AUC:", round(overall_auc, 3)))
print(paste("Average F1 Score:", round(overall_f1, 3)))
print(paste("Average G-Mean:", round(overall_gmean, 3)))

