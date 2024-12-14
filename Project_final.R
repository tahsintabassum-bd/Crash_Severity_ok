install.packages(c("tidyverse", "fastDummies", "data.table", "GGally", "corrplot", "skimr", 
                   "caret", "xgboost", "e1071", "randomForest", "glmnet", 
                 "pROC", "MLmetrics", "janitor", "plotly", "vroom"))


whole_oklahoma_state_2015_to_2021_nonmotorist->OkData
#Load_Data
OkData <- read.csv("OKDATA.csv")

# Check for missing values

print("Missing values per column:")
colSums(is.na(OkData))  # Replace 'data' with your dataset name
install.packages("dplyr")  # Install if not already installed
library(dplyr)  
# Drop columns with excessive missing values
columns_to_drop <- c("INJ_TYPE_4", "INJ_TYPE_5","INJ_TYPE_3","INJ_TYPE_2","INJ_TYPE","PERSON_CONDITION","PERSON_ID_INJURY","CRASH_DRUG_RELATED", "BAC",
                     "DOCID", "COUNTY", "CITY", "AGENCY", "STREET_HIGHWAY",
                     "INTERSECTING_ROAD", "ADDRESS", "TRANSPORTED_BY", 
                     "MEDICAL_FACILITY", "OHP_NON","HOUR4", "HOUR8", 
                     "HOUR6", "LONGITUDE", "LATITUDE", "x", "y", "DATE", 
                     "ObjectId", "WORKZONE_TYPE", "WORKZONE_LOCATION","INJURY_SEVERITY", "AGE","TIME","HOUR","HOUR8","HOUR6", "TOTAL_NONMOTORISTS")
OkData <- OkData %>% select(-all_of(columns_to_drop))

# Load necessary library
library(forcats)

# Identify categorical columns in the data
categorical_columns <- names(OkData)[sapply(OkData, is.character) | sapply(OkData, is.factor)]


# Verify changes
str(OkData)  # Check the structure to see reduced levels
# Impute missing values for numerical columns with median
numerical_columns <- sapply(OkData, is.numeric)
OkData[, numerical_columns] <- lapply(OkData[, numerical_columns], function(x) {
  ifelse(is.na(x), median(x, na.rm = TRUE), x)
})

# Impute missing values for categorical columns with mode
categorical_columns <- sapply(OkData, is.character)
OkData[, categorical_columns] <- lapply(OkData[, categorical_columns], function(x) {
  ifelse(is.na(x), names(sort(table(x), decreasing = TRUE))[1], x)
})

# Verify that missing values are handled
print("Missing values after handling:")
colSums(is.na(OkData))
OkData$CRASH_SEVERITY <- as.factor(OkData$CRASH_SEVERITY)



# Check the levels for each categorical column
str(OkData)
categorical_columns <- names(OkData)[sapply(OkData, function(x) is.factor(x) || is.character(x))]
print(categorical_columns)
library(forcats)

# Exclude CRASH_SEVERITY from lumping
categorical_columns_to_lump <- setdiff(categorical_columns, "CRASH_SEVERITY")

# Lump each categorical column to the top 5 levels
for (col in categorical_columns_to_lump) {
  OkData[[col]] <- fct_lump_n(OkData[[col]], n = 5)  # Lump to top 5 levels
}

# Verify the levels after lumping
lapply(OkData[categorical_columns_to_lump], levels)


# Drop low-variance variables
library(caret)
low_variance <- nearZeroVar(OkData, saveMetrics = TRUE)  # Identify low-variance variables
low_variance_cols <- rownames(low_variance[low_variance$nzv, ])
OkData <- OkData[, !(names(OkData) %in% low_variance_cols)]

OkData$CRASH_SEVERITY <- factor(OkData$CRASH_SEVERITY, levels = c("None","Possible","Minor","Serious","Fatal"))
OkData$CRASH_SEVERITY <- as.integer(OkData$CRASH_SEVERITY)
print("CRASH_SEVERITY successfully encoded:")
print(table(OkData$CRASH_SEVERITY))

# Identify numeric columns (excluding the target)
numeric_columns <- setdiff(names(OkData)[sapply(OkData, is.numeric)], "CRASH_SEVERITY")

# Standardize numeric columns
OkData[, numeric_columns] <- scale(OkData[, numeric_columns])

# Verify scaling
summary(OkData[, numeric_columns])

categorical_columns <- names(OkData)[sapply(OkData, function(x) is.factor(x) || is.character(x))]

# Exclude 'CRASH_SEVERITY' from the list of columns to encode
categorical_columns <- setdiff(categorical_columns, "CRASH_SEVERITY")

library(fastDummies)

OkData <- dummy_cols(
  OkData,
  select_columns = categorical_columns,
  remove_first_dummy = TRUE,
  remove_selected_columns = TRUE
)
str(OkData$CRASH_SEVERITY)

# Count of each class
class_counts <- table(OkData$CRASH_SEVERITY)
print(class_counts)

# Proportion of each class
class_proportions <- prop.table(class_counts)
print(round(class_proportions, 3))



##################################
#splitting the dataset 
library(caret)

set.seed(123)  # For reproducibility
trainIndex <- createDataPartition(OkData$CRASH_SEVERITY, p = 0.8, list = FALSE)
trainData <- OkData[trainIndex, ]
testData <- OkData[-trainIndex, ]

install.packages("smotefamily")
library(smotefamily)
# Ensure the target variable is a factor
trainData$CRASH_SEVERITY <- as.factor(trainData$CRASH_SEVERITY)

# Apply SMOTE
smote_result <- SMOTE(X = trainData[, -which(names(trainData) == "CRASH_SEVERITY")],
                      target = trainData$CRASH_SEVERITY,
                      K = 5,
                      dup_size = 2)  # dup_size controls the amount of over-sampling

# Create a balanced dataset
trainData_balanced <- smote_result$data

# Rename the target column back to 'CRASH_SEVERITY'
names(trainData_balanced)[ncol(trainData_balanced)] <- "CRASH_SEVERITY"

# Ensure the target variable is a factor again
trainData_balanced$CRASH_SEVERITY <- as.factor(trainData_balanced$CRASH_SEVERITY)

# Check the new class distribution
table(trainData_balanced$CRASH_SEVERITY)

###################################
# Cross validation
library(caret)

set.seed(123)  # For reproducibility

train_control <- trainControl(method = "cv", 
                              number = 5, 
                              classProbs = TRUE, 
                              summaryFunction = multiClassSummary)
####
# Get the levels from trainData_balanced
valid_levels <- make.names(levels(trainData_balanced$CRASH_SEVERITY))

# Apply the valid levels to both datasets
levels(trainData_balanced$CRASH_SEVERITY) <- valid_levels
levels(testData$CRASH_SEVERITY) <- valid_levels

# Verify the levels
print(levels(trainData_balanced$CRASH_SEVERITY))
print(levels(testData$CRASH_SEVERITY))

####
#Modeling
# LR
model_lr <- train(CRASH_SEVERITY ~ ., 
                  data = trainData_balanced, 
                  method = "multinom", 
                  trControl = train_control,
                  trace = FALSE)

#SVM

model_svm <- train(CRASH_SEVERITY ~ ., 
                   data = trainData_balanced, 
                   method = "svmLinear", 
                   trControl = train_control)

#ANN
model_ann <- train(CRASH_SEVERITY ~ ., 
                   data = trainData_balanced, 
                   method = "nnet", 
                   trControl = train_control, 
                   trace = FALSE)
#CF
model_cf <- train(CRASH_SEVERITY ~ ., 
                  data = trainData_balanced, 
                  method = "ranger", 
                  trControl = train_control)

#rf
model_rf <- train(CRASH_SEVERITY ~ ., 
data = trainData_balanced, 
method = "rf", 
trControl = train_control)



# Load necessary libraries
library(caret)
library(randomForest)

# Set seed for reproducibility
set.seed(123)

# Define a range for the number of trees to test
ntree_values <- seq(50, 500, by = 50)  # From 50 to 500 trees in increments of 50

# Create an empty data frame to store results
accuracy_results <- data.frame(ntree = integer(), Accuracy = numeric())

# Loop through different ntree values
for (ntree in ntree_values) {
  # Train Random Forest model with the specified number of trees
  rf_model <- train(
    CRASH_SEVERITY ~ .,
    data = trainData_balanced,
    method = "rf",
    trControl = trainControl(method = "cv", number = 5),
    tuneGrid = expand.grid(mtry = 5),  # Use the optimal mtry value found earlier
    ntree = ntree
  )
  
  # Store the accuracy result
  accuracy <- max(rf_model$results$Accuracy)
  accuracy_results <- rbind(accuracy_results, data.frame(ntree = ntree, Accuracy = accuracy))
}

# Plot the accuracy against the number of trees
library(ggplot2)

ggplot(accuracy_results, aes(x = ntree, y = Accuracy)) +
  geom_line(color = "blue") +
  geom_point(size = 2) +
  labs(
    title = "Random Forest Hyperparameter Tuning: Number of Trees",
    x = "Number of Trees (ntree)",
    y = "Accuracy (Cross-Validation)"
  ) +
  theme_minimal()



# Ensure necessary libraries are loaded
library(ggplot2)

# Create a data frame for predicted and actual values
residual_data <- data.frame(
  Predicted = as.numeric(pred_rf),
  Actual = as.numeric(testData$CRASH_SEVERITY)
)

# Calculate residuals (0 = Correct, 1 = Incorrect)
residual_data$Residuals <- ifelse(residual_data$Predicted == residual_data$Actual, 0, 1)

# Create a scatter plot for residuals
ggplot(residual_data, aes(x = Predicted, y = Residuals)) +
  geom_jitter(color = "blue", width = 0.2, height = 0.1) +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
  labs(
    title = "Residual Plot for Random Forest Model",
    x = "Predicted Crash Severity",
    y = "Residuals (0 = Correct, 1 = Incorrect)"
  ) +
  theme_minimal()

# Ensure necessary libraries are loaded
library(ggplot2)

# Get predicted values
pred_rf <- predict(model_rf, testData, type = "raw")

# Calculate residuals
residuals_rf <- as.numeric(testData$CRASH_SEVERITY) - as.numeric(pred_rf)

# Create a scatter plot of residuals
residual_plot <- data.frame(Predicted = as.numeric(pred_rf), Residuals = residuals_rf)

ggplot(residual_plot, aes(x = Predicted, y = Residuals)) +
  geom_point(color = "blue", alpha = 0.5) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  labs(title = "Residual Plot for Random Forest Model",
       x = "Predicted Values",
       y = "Residuals") +
  theme_minimal()

# Create a data frame for predicted and actual values
misclassification_data <- data.frame(
  Actual = testData$CRASH_SEVERITY,
  Predicted = pred_rf
)

# Plot misclassifications
library(ggplot2)

ggplot(misclassification_data, aes(x = Actual, fill = Predicted)) +
  geom_bar(position = "dodge") +
  labs(title = "Misclassification Plot for Random Forest Model",
       x = "Actual Crash Severity",
       y = "Count",
       fill = "Predicted") +
  theme_minimal()


























# Set up cross-validation control
train_control <- trainControl(method = "cv", number = 5)

# Grid for hyperparameter tuning
tune_grid <- expand.grid(mtry = c(2, 4, 6, 8, 10))

# Tuning Random Forest with different numbers of trees
set.seed(123)
rf_tune <- train(
  CRASH_SEVERITY ~ .,
  data = trainData_balanced,
  method = "rf",
  trControl = trainControl(method = "cv", number = 5),
  tuneGrid = expand.grid(mtry = 10),  # Replace 5 with the optimal mtry found previously
  ntree = 50  # Specify the number of trees (e.g., 50, 100, 200, etc.)
)

# View the best hyperparameters
print(model_rf$bestTune)

plot(model_rf)


# Calculate feature importance
rf_importance <- varImp(model_rf, scale = TRUE)

# Plot feature importance
plot(rf_importance, top = 10, main = "Top 10 Important Features for RF Model")



# Compute residuals
residuals_rf <- ifelse(pred_rf == testData$CRASH_SEVERITY, 0, 1)

# Plot residuals
library(ggplot2)
ggplot(data.frame(Residuals = residuals_rf), aes(x = Residuals)) +
  geom_bar() +
  labs(title = "Residual Plot for RF Model", x = "Residual (0 = Correct, 1 = Incorrect)", y = "Count")



# Make predictions on the test dataset
pred_rf <- predict(model_rf, testData)

# Generate confusion matrix
conf_rf <- confusionMatrix(pred_rf, testData$CRASH_SEVERITY)
print(conf_rf)
library(caret)
library(ggplot2)

# Original confusion matrix
conf_rf <- confusionMatrix(pred_rf, testData$CRASH_SEVERITY)

# Create a data frame from the confusion matrix
conf_matrix_data <- as.data.frame(conf_rf$table)

# Rename X1-X5 to descriptive labels
new_labels <- c("None","Minor", "Possible", "Serious", "Fatal")
conf_matrix_data$Prediction <- factor(conf_matrix_data$Prediction, levels = levels(conf_matrix_data$Prediction), labels = new_labels)
conf_matrix_data$Reference <- factor(conf_matrix_data$Reference, levels = levels(conf_matrix_data$Reference), labels = new_labels)

# Plot the confusion matrix
ggplot(data = conf_matrix_data, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), color = "white", size = 5) +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  labs(title = "Confusion Matrix for Random Forest Model",
       x = "Actual Crash Severity",
       y = "Predicted Crash Severity") +
  theme_minimal()


# Compute residuals
residuals_rf <- ifelse(pred_rf == testData$CRASH_SEVERITY, 0, 1)

# Plot residuals
library(ggplot2)
ggplot(data.frame(Residuals = residuals_rf), aes(x = Residuals)) +
  geom_bar() +
  labs(title = "Residual Plot for RF Model", x = "Residual (0 = Correct, 1 = Incorrect)", y = "Count")


#XGBoost
model_xgb <- train(CRASH_SEVERITY ~ ., 
                   data = trainData_balanced, 
                   method = "xgbTree", 
                   trControl = train_control)



#Predict on the test dataset for each model
pred_lr <- predict(model_lr, testData)
pred_svm <- predict(model_svm, testData)
pred_ann <- predict(model_ann, testData)
pred_cf <- predict(model_cf, testData)
pred_rf <- predict(model_rf, testData)
pred_xgb <- predict(model_xgb, testData)

###
# Convert numeric values to factor with the same levels as pred_lr
levels(pred_lr)
levels(testData$CRASH_SEVERITY)
confusionMatrix(pred_lr, testData$CRASH_SEVERITY)

print(unique(pred_lr))
print(unique(testData$CRASH_SEVERITY))

# Convert numeric values to factor with the same levels as pred_lr
testData$CRASH_SEVERITY <- factor(paste0("X", testData$CRASH_SEVERITY), levels = levels(pred_lr))

# Verify the conversion
print(unique(testData$CRASH_SEVERITY))

####
#performance measure 
confusionMatrix(pred_lr, testData$CRASH_SEVERITY)
confusionMatrix(pred_svm, testData$CRASH_SEVERITY)
confusionMatrix(pred_ann, testData$CRASH_SEVERITY)
confusionMatrix(pred_cf, testData$CRASH_SEVERITY)
confusionMatrix(pred_rf, testData$CRASH_SEVERITY)
confusionMatrix(pred_xgb, testData$CRASH_SEVERITY)

# Resample results for comparison
results <- resamples(list(
  Logistic_Regression = model_lr,
  SVM = model_svm,
  ANN = model_ann,
  Classification_Forest = model_cf,
  Random_Forest = model_rf,
  XGBoost = model_xgb
))

# Summary of performance metrics
summary(results)

# Boxplot of model performance
bwplot(results)



# Function to calculate average metrics across all classes
average_metrics <- function(metrics) {
  metrics %>%
    summarise(
      Avg_Precision = mean(Precision, na.rm = TRUE),
      Avg_Recall = mean(Recall, na.rm = TRUE),
      Avg_F1_Score = mean(F1_Score, na.rm = TRUE),
      Avg_Specificity = mean(Specificity, na.rm = TRUE),
      Avg_G_Mean = mean(G_Mean, na.rm = TRUE)
    )
}

# Calculate average metrics for each model
avg_metrics_lr <- average_metrics(metrics_lr)
avg_metrics_svm <- average_metrics(metrics_svm)
avg_metrics_ann <- average_metrics(metrics_ann)
avg_metrics_rf <- average_metrics(metrics_rf)
avg_metrics_xgb <- average_metrics(metrics_xgb)

# Combine the results into a single data frame
avg_metrics_combined <- data.frame(
  Model = c("Logistic Regression", "SVM", "ANN", "Random Forest", "XGBoost"),
  rbind(avg_metrics_lr, avg_metrics_svm, avg_metrics_ann, avg_metrics_rf, avg_metrics_xgb)
)

# Display the averaged performance metrics
print(avg_metrics_combined)
# Function to calculate average accuracy from a confusion matrix
calculate_avg_accuracy <- function(conf_matrix) {
  overall_accuracy <- conf_matrix$overall["Accuracy"]
  return(overall_accuracy)
}


##########
#more performance measures
library(caret)
library(dplyr)

# Function to calculate Precision, Recall, F1-Score, and G-Mean
calculate_metrics <- function(conf_matrix) {
  results <- data.frame(
    Class = rownames(conf_matrix$table),
    Precision = conf_matrix$byClass[, "Pos Pred Value"],
    Recall = conf_matrix$byClass[, "Sensitivity"],
    F1_Score = 2 * (conf_matrix$byClass[, "Pos Pred Value"] * conf_matrix$byClass[, "Sensitivity"]) /
      (conf_matrix$byClass[, "Pos Pred Value"] + conf_matrix$byClass[, "Sensitivity"]),
    Specificity = conf_matrix$byClass[, "Specificity"],
    G_Mean = sqrt(conf_matrix$byClass[, "Sensitivity"] * conf_matrix$byClass[, "Specificity"])
  )
  return(results)
}

# Confusion matrices for each model
conf_lr <- confusionMatrix(pred_lr, testData$CRASH_SEVERITY)
conf_svm <- confusionMatrix(pred_svm, testData$CRASH_SEVERITY)
conf_ann <- confusionMatrix(pred_ann, testData$CRASH_SEVERITY)
conf_rf <- confusionMatrix(pred_rf, testData$CRASH_SEVERITY)
conf_xgb <- confusionMatrix(pred_xgb, testData$CRASH_SEVERITY)

# Calculate metrics for each model
metrics_lr <- calculate_metrics(conf_lr)
metrics_svm <- calculate_metrics(conf_svm)
metrics_ann <- calculate_metrics(conf_ann)
metrics_rf <- calculate_metrics(conf_rf)
metrics_xgb <- calculate_metrics(conf_xgb)

# Display results
list(
  Logistic_Regression = metrics_lr,
  SVM = metrics_svm,
  ANN = metrics_ann,
  Random_Forest = metrics_rf,
  XGBoost = metrics_xgb
)

# For visualizing feature importance
install.packages("vip")
library(vip)
# For a Random Forest model
vip(model_rf)
############
install.packages("gridExtra")
library(gridExtra)
library(pROC)
library(ggplot2)
library(dplyr)

# Create a list of models and their predictions
model_predictions <- list(
  Logistic_Regression = predict(model_lr, testData, type = "prob"),
  SVM = predict(model_svm, testData, type = "prob"),
  ANN = predict(model_ann, testData, type = "prob"),
  Random_Forest = predict(model_rf, testData, type = "prob"),
  XGBoost = predict(model_xgb, testData, type = "prob")
)

# Actual classes
actual_classes <- testData$CRASH_SEVERITY

# Function to compute ROC curves for each class within a specific model
compute_roc_per_model <- function(predictions, actual, model_name) {
  do.call(rbind, lapply(levels(actual), function(class) {
    roc_curve <- roc(actual == class, predictions[, class])
    data.frame(
      fpr = 1 - roc_curve$specificities,
      tpr = roc_curve$sensitivities,
      class = class,
      model = model_name
    )
  }))
}

# Compute ROC curves for all models
roc_data_list <- mapply(compute_roc_per_model, model_predictions, MoreArgs = list(actual = actual_classes), names(model_predictions), SIMPLIFY = FALSE)

# Combine all results
roc_data <- do.call(rbind, roc_data_list)

# Create a separate plot for each model
plot_list <- lapply(unique(roc_data$model), function(model_name) {
  ggplot(filter(roc_data, model == model_name), aes(x = fpr, y = tpr, color = class)) +
    geom_line(size = 1) +
    labs(title = paste("ROC Curve for", model_name),
         x = "1 - Specificity (False Positive Rate)",
         y = "Sensitivity (True Positive Rate)",
         color = "Class") +
    theme_minimal() +
    theme(legend.position = "right")
})

# Display the plots using grid.arrange
library(gridExtra)
do.call(grid.arrange, c(plot_list, ncol = 2))



#####
#class weights
class_weights <- c(X1 = 2, X2 = 1, X3 = 1.5, X4 = 2.5, X5 = 2)

model_rf_weighted <- train(
  CRASH_SEVERITY ~ .,
  data = trainData_balanced,
  method = "ranger",
  trControl = train_control,
  weights = class_weights[trainData_balanced$CRASH_SEVERITY]
)

#####

# Ensure necessary libraries are loaded
library(caret)
library(nnet)  # For Logistic Regression as the meta-model

# Step 1: Combine Predictions into a Data Frame -------------------------------

# Filter out NAs in testData$CRASH_SEVERITY
valid_indices <- !is.na(testData$CRASH_SEVERITY)
testData_valid <- testData[valid_indices, ]

# Align predictions with valid test data
pred_lr_valid <- pred_lr[valid_indices]
pred_svm_valid <- pred_svm[valid_indices]
pred_ann_valid <- pred_ann[valid_indices]
pred_cf_valid <- pred_cf[valid_indices]
pred_rf_valid <- pred_rf[valid_indices]
pred_xgb_valid <- pred_xgb[valid_indices]

# Create a data frame with predictions from all models
stacked_data <- data.frame(
  LR = pred_lr_valid,
  SVM = pred_svm_valid,
  ANN = pred_ann_valid,
  CF = pred_cf_valid,
  RF = pred_rf_valid,
  XGB = pred_xgb_valid,
  CRASH_SEVERITY = testData_valid$CRASH_SEVERITY
)

# Verify the structure of the stacked data
str(stacked_data)

# Step 2: Train the Meta-Model ------------------------------------------------

set.seed(123)  # For reproducibility

# Train a Logistic Regression model as the meta-model
meta_model <- train(
  CRASH_SEVERITY ~ .,
  data = stacked_data,
  method = "multinom",    # Logistic Regression for multi-class classification
  trControl = trainControl(method = "cv", number = 5),
  trace = FALSE
)

# Step 3: Predict and Evaluate the Meta-Model ---------------------------------

# Make predictions using the meta-model
meta_predictions <- predict(meta_model, stacked_data)

# Evaluate the performance with a confusion matrix
confusionMatrix(meta_predictions, stacked_data$CRASH_SEVERITY)


# Ensure necessary libraries are loaded
library(caret)
library(dplyr)

# Function to calculate Precision, Recall, F1-Score, and G-Mean
calculate_metrics <- function(conf_matrix) {
  results <- data.frame(
    Class = rownames(conf_matrix$table),
    Precision = conf_matrix$byClass[, "Pos Pred Value"],
    Recall = conf_matrix$byClass[, "Sensitivity"],
    F1_Score = 2 * (conf_matrix$byClass[, "Pos Pred Value"] * conf_matrix$byClass[, "Sensitivity"]) /
      (conf_matrix$byClass[, "Pos Pred Value"] + conf_matrix$byClass[, "Sensitivity"]),
    Specificity = conf_matrix$byClass[, "Specificity"],
    G_Mean = sqrt(conf_matrix$byClass[, "Sensitivity"] * conf_matrix$byClass[, "Specificity"])
  )
  return(results)
}

# Generate confusion matrix for the stacked model
conf_stacked <- confusionMatrix(meta_predictions, stacked_data$CRASH_SEVERITY)

# Calculate metrics for the stacked model
metrics_stacked <- calculate_metrics(conf_stacked)

# Display the performance metrics for the stacked model
print(metrics_stacked)



library(ggplot2)

# Identify misclassified instances
stacked_data$Predicted <- meta_predictions
stacked_data$Correct <- stacked_data$CRASH_SEVERITY == stacked_data$Predicted

# Plot of correct vs. incorrect predictions
ggplot(stacked_data, aes(x = CRASH_SEVERITY, fill = Correct)) +
  geom_bar(position = "dodge") +
  labs(title = "Distribution of Correct vs. Incorrect Predictions",
       x = "Actual Crash Severity",
       y = "Count",
       fill = "Prediction Correct") +
  theme_minimal()



#clustering 


# Table showing actual vs. predicted
table(stacked_data$CRASH_SEVERITY, stacked_data$Predicted)


# Extract coefficients from the meta-model
coef_meta <- coef(meta_model$finalModel)
print(coef_meta)

library(reshape2)

# Melt the coefficients for visualization
coef_df <- melt(coef_meta)
colnames(coef_df) <- c("Class", "Model", "Coefficient")

# Plot the coefficients
ggplot(coef_df, aes(x = Model, y = Coefficient, fill = Class)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Feature Importance (Coefficients) for Meta-Model",
       x = "Base Models",
       y = "Coefficient Value",
       fill = "Class") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Plot cross-validation results
plot(meta_model)


# Print the confusion matrix
print(conf_stacked)

# Print the detailed metrics
print(metrics_stacked)


library(ggplot2)

# Convert confusion matrix to data frame for visualization
confusion_data <- as.data.frame(conf_stacked$table)

# Plot the confusion matrix heatmap
ggplot(confusion_data, aes(x = Prediction, y = Reference, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), color = "white", size = 4) +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  labs(title = "Confusion Matrix Heatmap for Stacked Model",
       x = "Predicted Class",
       y = "Actual Class") +
  theme_minimal()


# Extract coefficients from the meta-model
meta_coef <- coef(meta_model$finalModel)

# Convert coefficients to a data frame
coef_df <- as.data.frame(meta_coef)
coef_df$Feature <- rownames(coef_df)

# Reshape the data frame for visualization
library(reshape2)
coef_long <- melt(coef_df, id.vars = "Feature", variable.name = "Class", value.name = "Coefficient")

# View the reshaped data frame
head(coef_long)

library(ggplot2)

# Plot the coefficients for each class
ggplot(coef_long, aes(x = Feature, y = Coefficient, fill = Class)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Feature Importance (Coefficients) for Stacked Model",
       x = "Base Models (Features)",
       y = "Coefficient Value",
       fill = "Class") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


##3
# Load necessary libraries
library(caret)
library(vip)        # For visualizing feature importance
library(ggplot2)
library(dplyr)
library(NeuralNetTools)  # For ANN feature importance

# Logistic Regression (LR) Feature Importance
lr_importance <- varImp(model_lr, scale = TRUE)$importance
lr_importance$model <- "LR"

# SVM Feature Importance (for linear SVM)
svm_importance <- tryCatch({
  importance <- as.data.frame(coef(model_svm$finalModel))
  importance$Feature <- rownames(importance)
  importance <- importance %>% rename(Overall = V1)
  importance$model <- "SVM"
  importance
}, error = function(e) NULL)

# ANN Feature Importance using Garson's Algorithm
ann_importance <- tryCatch({
  importance <- garson(model_ann$finalModel)
  importance <- as.data.frame(importance)
  importance$Feature <- rownames(importance)
  importance <- importance %>% rename(Overall = rel_imp)
  importance$model <- "ANN"
  importance
}, error = function(e) NULL)

# Classification Forest (CF) Feature Importance
cf_importance <- tryCatch({
  importance <- varImp(model_cf, scale = TRUE)$importance
  importance$model <- "CF"
  importance
}, error = function(e) NULL)

# Random Forest (RF) Feature Importance
rf_importance <- varImp(model_rf, scale = TRUE)$importance
rf_importance$model <- "RF"

# XGBoost Feature Importance
xgb_importance <- varImp(model_xgb, scale = TRUE)$importance
xgb_importance$model <- "XGB"

# Combine all available feature importance data frames
combined_importance <- bind_rows(
  lr_importance %>% mutate(Feature = rownames(lr_importance)),
  svm_importance,
  ann_importance,
  cf_importance,
  rf_importance %>% mutate(Feature = rownames(rf_importance)),
  xgb_importance %>% mutate(Feature = rownames(xgb_importance))
)

# Step 2: Aggregate Importance Across All Models -----------------------------
aggregated_importance <- combined_importance %>%
  group_by(Feature) %>%
  summarise(Average_Importance = mean(Overall, na.rm = TRUE)) %>%
  arrange(desc(Average_Importance))

# Step 3: Plot the Aggregated Feature Importance -----------------------------
# Visualize the top 20 features
ggplot(aggregated_importance %>% top_n(20, Average_Importance), 
       aes(x = reorder(Feature, Average_Importance), y = Average_Importance)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  coord_flip() +
  labs(title = "Aggregated Feature Importance from Base Models",
       x = "Features",
       y = "Average Importance Score") +
  theme_minimal()
