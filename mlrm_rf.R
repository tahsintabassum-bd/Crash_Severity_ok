install.packages(c("tidyverse", "fastDummies", "data.table", "GGally", "corrplot", "skimr", 
                   "caret", "xgboost", "e1071", "randomForest", "glmnet", 
                   "pROC", "MLmetrics", "janitor", "plotly", "vroom"))
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
                     "MEDICAL_FACILITY", "OHP_NON","HOUR", "HOUR8", 
                     "HOUR6", "LONGITUDE", "LATITUDE", "x", "y", "DATE", 
                     "ObjectId", "WORKZONE_TYPE", "WORKZONE_LOCATION","INJURY_SEVERITY")
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

# Lumping categorical variables
library(forcats)
categorical_columns <- names(OkData)[sapply(OkData, is.character) | sapply(OkData, is.factor)]
for (col in categorical_columns) {
  OkData[[col]] <- as.factor(OkData[[col]])  # Ensure it's a factor
  OkData[[col]] <- fct_lump(OkData[[col]], n = 10)  # Reduce to 10 levels
}



# Drop low-variance variables
library(caret)
low_variance <- nearZeroVar(OkData, saveMetrics = TRUE)  # Identify low-variance variables
low_variance_cols <- rownames(low_variance[low_variance$nzv, ])
OkData <- OkData[, !(names(OkData) %in% low_variance_cols)]



# Identify categorical columns
categorical_columns <- sapply(OkData, is.character)
print(categorical_columns)


OkData$CRASH_SEVERITY <- factor(OkData$CRASH_SEVERITY, levels = c("None","Possible","Minor","Serious","Fatal"))
OkData$CRASH_SEVERITY <- as.integer(OkData$CRASH_SEVERITY)
print("CRASH_SEVERITY successfully encoded:")
print(table(OkData$CRASH_SEVERITY))


# One-Hot Encoding for other categorical features, excluding CRASH_SEVERITY
library(fastDummies)

# Identify categorical columns
categorical_columns <- names(OkData)[sapply(OkData, is.character)]

# Exclude 'CRASH_SEVERITY' from encoding
categorical_columns <- setdiff(categorical_columns, "CRASH_SEVERITY")

# Apply one-hot encoding to the remaining categorical columns
if (length(categorical_columns) > 0) {
  OkData <- dummy_cols(OkData, 
                       select_columns = categorical_columns, 
                       remove_first_dummy = TRUE, 
                       remove_selected_columns = TRUE)
}

# Verify the structure of the updated dataset
str(OkData)


# Check target variable distribution
table(OkData$CRASH_SEVERITY)

# Proportions
prop.table(table(OkData$CRASH_SEVERITY))

# Identify numeric columns
numeric_columns <- sapply(OkData, is.numeric)

# Scale numeric columns (excluding CRASH_SEVERITY)
scaled_columns <- setdiff(names(OkData)[numeric_columns], "CRASH_SEVERITY")
OkData[, scaled_columns] <- scale(OkData[, scaled_columns])

# Verify scaling
summary(OkData[, scaled_columns])

# Identify numeric columns
numeric_columns <- sapply(OkData, is.numeric)

# Scale numeric columns (excluding CRASH_SEVERITY)
scaled_columns <- setdiff(names(OkData)[numeric_columns], "CRASH_SEVERITY")
OkData[, scaled_columns] <- scale(OkData[, scaled_columns])

# Verify scaling
summary(OkData[, scaled_columns])


# Check for missing values in CRASH_SEVERITY
sum(is.na(OkData$CRASH_SEVERITY))  # Count NA values
sum(is.nan(OkData$CRASH_SEVERITY)) # Count NaN values

# Impute missing values in CRASH_SEVERITY with the mode (most frequent value)
mode_value <- as.integer(names(sort(table(OkData$CRASH_SEVERITY), decreasing = TRUE))[1])
OkData$CRASH_SEVERITY[is.na(OkData$CRASH_SEVERITY) | is.nan(OkData$CRASH_SEVERITY)] <- mode_value

#Multinomial logistic regression
install.packages("nnet")
library(nnet)
OkData$CRASH_SEVERITY <- as.factor(OkData$CRASH_SEVERITY)
set.seed(123)  # For reproducibility
train_indices <- sample(seq_len(nrow(OkData)), size = 0.7 * nrow(OkData))

train_data <- OkData[train_indices, ]
test_data <- OkData[-train_indices, ]
model <- multinom(CRASH_SEVERITY ~ ., data = train_data)
summary(model)
# Assess Variable Relevance

# Remove Predictors with High p-Values
# Extract p-values
coeff_summary <- summary(model)
z_scores <- coeff_summary$coefficients / coeff_summary$standard.errors
p_values <- (1 - pnorm(abs(z_scores), 0, 1)) * 2  # Two-tailed test
print(p_values)

# Identify predictors with p > 0.05
irrelevant_vars <- colnames(p_values)[apply(p_values, 2, function(x) all(x > 0.05))]

# Remove irrelevant variables from training and test data
train_data <- train_data[, !(names(train_data) %in% irrelevant_vars)]
test_data <- test_data[, !(names(test_data) %in% irrelevant_vars)]

# Rebuild the model
model_refined <- multinom(CRASH_SEVERITY ~ ., data = train_data)
summary(model_refined)

# Load required package
if (!requireNamespace("broom", quietly = TRUE)) {
  install.packages("broom")
}
library(broom)

# Extract tidy table from the multinomial logistic regression model
tidy_table <- tidy(model_refined)

# View the tidy table
head(tidy_table)

# Save to CSV if needed
write.csv(tidy_table, "model_summary_table.csv", row.names = FALSE)
# Ensure broom is loaded
library(broom)

# Extract full tidy table from the multinomial logistic regression model
full_tidy_table <- tidy(model_refined)

# View the full table in the R console (optional)
View(full_tidy_table)

# Save the full table to a CSV file
write.csv(full_tidy_table, "multinomial_model_summary.csv", row.names = FALSE)

library(dplyr)
# Ensure tidyr is loaded
library(tidyr)

# Address NaNs in sqrt(diag(vc)) by refitting the model if necessary
# Option 1: Check VIF for multicollinearity
library(car)
vif_values <- vif(model_refined)
print(vif_values)
coeff_magnitude <- abs(summary(model_refined)$coefficients)

sorted_coeff <- coeff_magnitude %>%
  as.data.frame() %>%
  mutate(Predictor = rownames(.)) %>%  # Create a new column from rownames
  pivot_longer(cols = -Predictor, names_to = "Outcome_Level", values_to = "Magnitude") %>%
  arrange(desc(Magnitude))
print(head(sorted_coeff, 10))

#Stepwise Selection
model_stepwise <- step(model_refined, direction = "both")
summary(model_stepwise)

coeff_magnitude <- abs(summary(model_stepwise)$coefficients)

sorted_coeff <- coeff_magnitude %>%
  as.data.frame() %>%
  mutate(Predictor = rownames(.)) %>%  # Create a new column from rownames
  pivot_longer(cols = -Predictor, names_to = "Outcome_Level", values_to = "Magnitude") %>%
  arrange(desc(Magnitude))
print(head(sorted_coeff, 20))
# Extract the formula of the final model after stepwise selection
final_formula <- formula(model_stepwise)
print(final_formula)

# List of relevant variables
relevant_vars <- all.vars(final_formula)
print("Relevant Variables from Stepwise Selection:")
print(relevant_vars)

coefficients <- summary(model_stepwise)$coefficients
print(coefficients)

library(dplyr)
coeff_magnitude <- abs(summary(model_stepwise)$coefficients)
ranked_vars <- apply(coeff_magnitude, 2, function(x) sum(x))  # Sum of absolute coefficients
ranked_vars <- sort(ranked_vars, decreasing = TRUE)
print(ranked_vars)

directionality <- apply(summary(model_stepwise)$coefficients, 2, function(x) mean(x))
print(directionality)

coeff_df <- as.data.frame(summary(model_stepwise)$coefficients)
coeff_df$Predictor <- rownames(coeff_df)
coeff_df <- coeff_df %>%
  pivot_longer(-Predictor, names_to = "Level", values_to = "Coefficient")

library(ggplot2)

var_importance <- data.frame(
  Predictor = names(ranked_vars),
  Importance = ranked_vars
)

ggplot(var_importance, aes(x = reorder(Predictor, -Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  coord_flip() +
  labs(title = "Predictor Importance After Stepwise Selection",
       x = "Predictors", y = "Coefficient Magnitude") +
  theme_minimal()

ggplot(OkData, aes(x = INTERSECTION, fill = CRASH_SEVERITY)) +
  geom_bar(position = "fill") +
  labs(title = "Crash Severity by Number of Vehicles Involved",
       x = "Type of Intersection", y = "Proportion") +
  theme_minimal()
ggplot(OkData, aes(x = INTERSECTION, fill = CRASH_SEVERITY)) +
  geom_bar(position = "fill") +
  labs(
    title = "Crash Severity by Type of Intersection",
    x = "Type of Intersection",
    y = "Proportion",
    fill = "Crash Severity"
  ) +
  theme_bw(base_size = 14) +  # Black and white theme with larger text size
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", color = "darkblue", size = 16), # Center-align title
    axis.title.x = element_text(color = "darkred", size = 14, face = "bold"),
    axis.title.y = element_text(color = "darkred", size = 14, face = "bold"),
    axis.text.x = element_text(angle = 45, hjust = 1, size = 12), # Tilt x-axis labels for readability
    axis.text.y = element_text(size = 12),
    legend.position = "right",
    legend.background = element_rect(fill = "gray90", color = "black"),
    legend.title = element_text(face = "bold")
  )
ggplot(OkData, aes(x = INTERSECTION, fill = CRASH_SEVERITY)) +
  geom_bar(position = "fill") +
  labs(
    title = "Crash Severity by Type of Intersection",
    x = "Type of Intersection",
    y = "Proportion",
    fill = "Crash Severity"
  ) +
  scale_fill_brewer(palette = "Set2")
library(dplyr)
OkData %>%
  group_by(HIGHWAY_CLASS, LIGHTING) %>%
  summarize(Avg_Severity = mean(as.numeric(CRASH_SEVERITY))) %>%
  ggplot(aes(x = HIGHWAY_CLASS, y = Avg_Severity, fill = LIGHTING)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Interaction Between Highway Class and Lighting",
       x = "Highway Class", y = "Average Severity") +
  theme_minimal()
install.packages("ggmosaic")

library(ggmosaic)

ggplot(data = OkData) +
  geom_mosaic(aes(weight = 1, x = product(CRASH_SEVERITY), fill = CRASH_SEVERITY, conds = product(INTERSECTION))) +
  labs(
    title = "Mosaic Plot of Crash Severity by Intersection Type",
    x = "Intersection Type",
    y = "Crash Severity",
    fill = "Crash Severity"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 16),
    axis.text.x = element_text(angle = 45, hjust = 1, size = 12),
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank()
  )
# Prepare data for bubble chart
bubble_data <- OkData %>%
  group_by(INTERSECTION, CRASH_SEVERITY) %>%
  summarise(Count = n()) %>%
  ungroup()

# Plot bubble chart
ggplot(bubble_data, aes(x = INTERSECTION, y = CRASH_SEVERITY, size = Count, color = CRASH_SEVERITY)) +
  geom_point(alpha = 0.7) +
  scale_size(range = c(2, 12)) +
  labs(
    title = "Crash Severity by Intersection Type",
    x = "Intersection Type",
    y = "Crash Severity",
    size = "Count"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 16),
    axis.text.x = element_text(angle = 45, hjust = 1, size = 12)
  )
library(ggplot2)
library(dplyr)
library(tidyr)

# Prepare data for heatmap
heatmap_data <- OkData %>%
  group_by(PEDESTRIAN_LOCATION, CRASH_SEVERITY) %>%
  summarise(Count = n()) %>%
  mutate(Proportion = Count / sum(Count)) %>%
  ungroup()

# Plot heatmap
ggplot(heatmap_data, aes(x = CRASH_SEVERITY, y = INTERSECTION, fill = Proportion)) +
  geom_tile(color = "white") +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  labs(
    title = "Heatmap of Crash Severity by Intersection Type",
    x = "Crash Severity",
    y = "Intersection Type",
    fill = "Proportion"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 16),
    axis.text.x = element_text(angle = 45, hjust = 1, size = 12),
    axis.text.y = element_text(size = 12)
  )

# Model Evaluation

# Predict on test data
predicted_probs <- predict(model_stepwise, newdata = test_data, type = "probs")
predicted_classes <- predict(model_stepwise, newdata = test_data)

# Evaluate performance
confusion_matrix <- confusionMatrix(predicted_classes, test_data$CRASH_SEVERITY)
print(confusion_matrix)
library(ggplot2)
library(reshape2)

# Convert confusion matrix into a data frame for visualization
cm_table <- as.table(confusion_matrix$table)
cm_df <- as.data.frame(cm_table)

# Rename columns for clarity
colnames(cm_df) <- c("Actual", "Predicted", "Frequency")

# Create a heatmap
ggplot(cm_df, aes(x = Predicted, y = Actual, fill = Frequency)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Frequency), color = "black", size = 5) +
  scale_fill_gradient(low = "lightblue", high = "blue") +
  labs(
    title = "Confusion Matrix Heatmap",
    x = "Predicted Class",
    y = "Actual Class",
    fill = "Frequency"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    text = element_text(size = 12)
  )




library(ggplot2)

# Aggregate data
line_data <- OkData %>%
  group_by(YEAR, INTERSECTION, CRASH_SEVERITY) %>%
  summarise(Count = n()) %>%
  mutate(Proportion = Count / sum(Count)) %>%
  ungroup()

# Plot line chart
ggplot(line_data, aes(x = YEAR, y = Proportion, color = CRASH_SEVERITY, group = CRASH_SEVERITY)) +
  geom_line(size = 1.2) +
  facet_wrap(~INTERSECTION, scales = "free_y") +
  labs(
    title = "Crash Severity by Pedestrian Action",
    x = "Year",
    y = "Proportion",
    color = "Crash Severity"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 16),
    axis.text.x = element_text(angle = 45, hjust = 1, size = 12),
    legend.position = "bottom"
  )

# Accuracy
accuracy <- mean(predicted_classes == test_data$CRASH_SEVERITY)
print(paste("Accuracy:", round(accuracy * 100, 2), "%"))
# Install MLmetrics if not already installed
install.packages("MLmetrics")

# Load the package
library(MLmetrics)

# F1-Score
f1_scores <- sapply(levels(test_data$CRASH_SEVERITY), function(class) {
  F1_Score(y_pred = predicted_classes == class, y_true = test_data$CRASH_SEVERITY == class)
})
macro_f1 <- mean(f1_scores)
print(f1_scores)
print(paste("Macro-Averaged F1-Score:", round(macro_f1, 2)))

# AUC-ROC
# Install pROC if not already installed
install.packages("pROC")

# Load the pROC package
library(pROC)

roc_results <- list()
for (class in colnames(predicted_probs)) {
  binary_response <- as.numeric(test_data$CRASH_SEVERITY == class)
  roc_results[[class]] <- roc(binary_response, predicted_probs[, class])
}
auc_values <- sapply(roc_results, auc)
print(auc_values)
macro_auc <- mean(auc_values)
print(paste("Macro-Averaged AUC:", round(macro_auc, 2)))
library(pROC)
library(ggplot2)

# Prepare a list to store ROC objects for each class
roc_results <- list()

# Compute ROC for each class
for (class in colnames(predicted_probs)) {
  binary_response <- as.numeric(test_data$CRASH_SEVERITY == class)  # One-vs-all binary labels
  roc_results[[class]] <- roc(binary_response, predicted_probs[, class])
}

# Create a data frame for ROC curves
roc_data <- do.call(rbind, lapply(names(roc_results), function(class) {
  roc_obj <- roc_results[[class]]
  data.frame(
    Sensitivity = roc_obj$sensitivities,
    Specificity = 1 - roc_obj$specificities,  # False positive rate
    Class = class
  )
}))

# Plot ROC curves using ggplot2
ggplot(roc_data, aes(x = Specificity, y = Sensitivity, color = Class)) +
  geom_line(size = 1) +
  scale_color_manual(values = c("blue", "green", "orange", "purple", "red")) +
  labs(
    title = "AUC-ROC Curves for MLRM",
    x = "1 - Specificity (False Positive Rate)",
    y = "Sensitivity (True Positive Rate)",
    color = "Class"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),
    axis.text = element_text(size = 12),
    axis.title = element_text(size = 14),
    legend.text = element_text(size = 12)
  )
# 3. G-mean
conf_matrix <- table(Predicted = predicted_classes, Actual = y_test + 1)
sensitivity <- diag(conf_matrix) / rowSums(conf_matrix)  # Recall
specificity <- diag(conf_matrix) / colSums(conf_matrix)  # Specificity
g_mean <- sqrt(sensitivity * specificity)

# Print G-mean
print("G-mean for each class:")
print(g_mean)

overall_gmean <- mean(g_mean, na.rm = TRUE)
print(paste("Average G-Mean:", round(overall_gmean, 3)))

# Compare Model Fit
print(AIC(model_stepwise))
print(BIC(model_stepwise))


#random forest

install.packages("randomForest")
library(randomForest)
set.seed(123)  # For reproducibility

# Train the Random Forest model
rf_model <- randomForest(
  CRASH_SEVERITY ~ .,         # Formula
  data = train_data,          # Training data
  ntree = 500,                # Number of trees
  mtry = floor(sqrt(ncol(train_data) - 1)),  # Number of variables considered at each split
  importance = TRUE           # Calculate variable importance
)
# Predict class labels
rf_predicted_classes <- predict(rf_model, newdata = test_data)

# Predict probabilities
rf_predicted_probs <- predict(rf_model, newdata = test_data, type = "prob")
library(caret)
# Check levels of predicted classes
levels(rf_predicted_classes)

# Check levels of actual classes
levels(test_data$CRASH_SEVERITY)
# Set levels for predicted and actual classes
levels(rf_predicted_classes) <- levels(test_data$CRASH_SEVERITY)

# Convert to factors if needed
rf_predicted_classes <- factor(rf_predicted_classes, levels = levels(test_data$CRASH_SEVERITY))
test_data$CRASH_SEVERITY <- factor(test_data$CRASH_SEVERITY, levels = levels(rf_predicted_classes))

# Create the confusion matrix
rf_confusion_matrix <- confusionMatrix(rf_predicted_classes, test_data$CRASH_SEVERITY)
print(rf_confusion_matrix)
#accuracy
rf_accuracy <- mean(rf_predicted_classes == test_data$CRASH_SEVERITY)
print(paste("Accuracy:", round(rf_accuracy * 100, 2), "%"))

library(MLmetrics)

# Calculate F1-Score for each class
rf_f1_scores <- sapply(levels(test_data$CRASH_SEVERITY), function(class) {
  F1_Score(
    y_pred = rf_predicted_classes == class,
    y_true = test_data$CRASH_SEVERITY == class
  )
})
rf_macro_f1 <- mean(rf_f1_scores)
print(rf_f1_scores)
print(paste("Macro-Averaged F1-Score:", round(rf_macro_f1, 2)))
library(pROC)

# Compute AUC-ROC for each class (one-vs-all)
rf_roc_results <- list()
for (class in colnames(rf_predicted_probs)) {
  binary_response <- as.numeric(test_data$CRASH_SEVERITY == class)
  rf_roc_results[[class]] <- roc(binary_response, rf_predicted_probs[, class])
}
unique(binary_response)  # Should return 0 and 1
binary_response <- factor(as.numeric(test_data$CRASH_SEVERITY == class), levels = c(0, 1))
rf_roc_results <- list()  # Initialize list to store ROC results

for (class in colnames(rf_predicted_probs)) {
  # Create binary response for one-vs-all comparison
  binary_response <- as.numeric(test_data$CRASH_SEVERITY == class)
  
  # Skip the class if the binary response has only one level
  if (length(unique(binary_response)) < 2) {
    next  # Skip to the next class in the loop
  }
  
  # Compute ROC curve
  rf_roc_results[[class]] <- roc(factor(binary_response, levels = c(0, 1)), rf_predicted_probs[, class])
}


# AUC values
rf_auc_values <- sapply(rf_roc_results, auc)
rf_macro_auc <- mean(rf_auc_values)
# Extract AUC values, excluding any NULL or NA elements
rf_auc_values <- sapply(rf_roc_results, function(x) if (!is.null(x)) auc(x) else NA)

# Filter out NA values
rf_auc_values <- rf_auc_values[!is.na(rf_auc_values)]

# Print valid AUC values
print(rf_auc_values)

print(rf_auc_values)
print(paste("Macro-Averaged AUC:", round(rf_macro_auc, 2)))

library(ggplot2)
library(reshape2)

# Convert confusion matrix to a data frame
rf_cm_table <- as.table(rf_confusion_matrix$table)  # Assumes rf_confusion_matrix is already created
rf_cm_df <- as.data.frame(rf_cm_table)

# Rename columns for clarity
colnames(rf_cm_df) <- c("Actual", "Predicted", "Frequency")

# Create a heatmap
ggplot(rf_cm_df, aes(x = Predicted, y = Actual, fill = Frequency)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Frequency), color = "black", size = 5) +
  scale_fill_gradient(low = "lightblue", high = "blue") +
  labs(
    title = "Random Forest Confusion Matrix Heatmap",
    x = "Predicted Class",
    y = "Actual Class",
    fill = "Frequency"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    text = element_text(size = 12)
  )
# Plot AUC-ROC curves
ggplot(roc_data, aes(x = FPR, y = TPR, color = Class)) +
  geom_line(size = 1.2) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray") +  # Diagonal reference line
  labs(
    title = "AUC-ROC Curves for RF",
    x = "False Positive Rate (1 - Specificity)",
    y = "True Positive Rate (Sensitivity)",
    color = "Class"
  ) +
  theme_minimal() +
  scale_color_brewer(palette = "Set1") +
  theme(
    text = element_text(size = 14),
    legend.position = "right"
  )
# 1. Extract confusion matrix table
rf_cm_table <- rf_confusion_matrix$table

# 2. Initialize lists to store metrics
sensitivity_list <- numeric()
specificity_list <- numeric()
g_mean_list <- numeric()

# 3. Calculate Sensitivity, Specificity, and G-Mean for each class
for (class in levels(test_data$CRASH_SEVERITY)) {
  TP <- rf_cm_table[class, class]  # True Positives
  FN <- sum(rf_cm_table[class, ]) - TP  # False Negatives
  FP <- sum(rf_cm_table[, class]) - TP  # False Positives
  TN <- sum(rf_cm_table) - (TP + FN + FP)  # True Negatives
  
  sensitivity <- TP / (TP + FN)  # Sensitivity (Recall)
  specificity <- TN / (TN + FP)  # Specificity
  
  # Store values
  sensitivity_list <- c(sensitivity_list, sensitivity)
  specificity_list <- c(specificity_list, specificity)
  
  # G-Mean
  if (!is.na(sensitivity) && !is.na(specificity)) {
    g_mean <- sqrt(sensitivity * specificity)
  } else {
    g_mean <- NA
  }
  g_mean_list <- c(g_mean_list, g_mean)
}

# 4. Macro-Averaged G-Mean
macro_g_mean <- mean(g_mean_list, na.rm = TRUE)

# Print the results
results <- data.frame(
  Class = levels(test_data$CRASH_SEVERITY),
  Sensitivity = sensitivity_list,
  Specificity = specificity_list,
  G_Mean = g_mean_list
)

print(results)
print(paste("Macro-Averaged G-Mean:", round(macro_g_mean, 4)))
# Extract variable importance
var_importance <- importance(rf_model)

# Convert to a data frame for easier handling
var_importance_df <- data.frame(
  Predictor = rownames(var_importance),
  Importance = var_importance[, "MeanDecreaseGini"]
)

# Sort by importance
var_importance_df <- var_importance_df[order(-var_importance_df$Importance), ]
print(var_importance_df)

library(ggplot2)

ggplot(var_importance_df, aes(x = reorder(Predictor, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Variable Importance in Random Forest Model",
       x = "Predictor", y = "Importance (Mean Decrease Gini)") +
  theme_minimal()
library(pdp)

# Example for the top variable
# Install if not already installed
install.packages("pdp")

# Load the package
library(pdp)

partial_plot <- partial(rf_model, pred.var = var_importance_df$Predictor[1], plot = TRUE, rug = TRUE)

# Save plot
ggsave("partial_dependence_plot.png", plot = partial_plot, width = 8, height = 6)
# Generate partial dependence with ggplot2
partial_plot_gg <- partial(
  object = rf_model, 
  pred.var = var_importance_df$Predictor[1], 
  train = train_data, 
  plot.engine = "ggplot2"
)

# Save using ggsave
ggsave("partial_dependence_plot.png", plot = partial_plot_gg, width = 8, height = 6)
library(ggplot2)

# Generate the partial dependence data
partial_plot_gg <- partial(
  object = rf_model, 
  pred.var = var_importance_df$Predictor[1], 
  train = train_data
)

# Create a ggplot2 plot
p <- ggplot(partial_plot_gg, aes(x = eval(var_importance_df$Predictor[1]), y = yhat)) +
  geom_line(size = 1.2, color = "blue") +
  labs(
    title = paste("Partial Dependence Plot for", var_importance_df$Predictor[1]),
    x = var_importance_df$Predictor[1],
    y = "Predicted Value"
  ) +
  theme_minimal()

# Save the plot
ggsave("partial_dependence_plot.png", plot = p, width = 8, height = 6)
# Interaction of top two predictors
interaction_plot <- partial(rf_model, pred.var = c(var_importance_df$Predictor[1], var_importance_df$Predictor[2]), plot = TRUE)

# Save plot
ggsave("interaction_plot.png", plot = interaction_plot, width = 8, height = 6)
interaction_pdp <- partial(
  object = rf_model,
  pred.var = c(var_importance_df$Predictor[1], var_importance_df$Predictor[2]),
  train = train_data
)

# Visualize the interaction
library(ggplot2)
ggplot(interaction_pdp, aes(x = eval(var_importance_df$Predictor[1]), y = yhat, color = eval(var_importance_df$Predictor[2]))) +
  geom_line() +
  labs(
    title = "Interaction Plot for Top Predictors",
    x = var_importance_df$Predictor[1],
    y = "Predicted Value"
  ) +
  theme_minimal()

ggplot(misclassified, aes(x = CRASH_SEVERITY, y = AGE_GRP)) +
  geom_boxplot() +
  labs(title = "Age Distribution of Misclassified Cases",
       x = "Actual Severity",
       y = "Age Group")


library(lubridate)

# Aggregate crashes by month/year
crash_trends <- train_data %>%
  mutate(Date = ymd(paste(YEAR, MONTH, DAY, sep = "-"))) %>%
  group_by(Date) %>%
  summarize(Avg_Severity = mean(CRASH_SEVERITY))

# Plot
ggplot(crash_trends, aes(x = Date, y = Avg_Severity)) +
  geom_line() +
  labs(title = "Average Crash Severity Over Time", x = "Date", y = "Severity")
# Load necessary library
library(dplyr)

# Create a Date column and calculate average severity by month
filtered_data <- filtered_data %>%
  mutate(Date = as.Date(paste(YEAR, MONTH, "01", sep = "-"))) %>% # First day of each month
  group_by(Date) %>%
  summarize(Avg_Severity = mean(CRASH_SEVERITY, na.rm = TRUE))
# Load necessary library
library(dplyr)
# Filter data for years 2015 to 2021
filtered_data <- train_data %>%
  filter(YEAR >= 2015 & YEAR <= 2021)
# Load ggplot2 for plotting
library(ggplot2)

# Create the trend plot
ggplot(filtered_data, aes(x = Date, y = CRASH_SEVERITY)) +
  geom_line(color = "blue", size = 1) +
  geom_point(color = "red", size = 2) +
  labs(
    title = "Trend of Crash Severity (2015-2021)",
    x = "Year",
    y = "Average Crash Severity"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),
    axis.text = element_text(size = 12),
    axis.title = element_text(size = 14)
  )

# Create a Date column and calculate average severity by month
filtered_data <- filtered_data %>%
  mutate(Date = as.Date(paste(YEAR, MONTH, "01", sep = "-"))) %>% # First day of each month
  group_by(Date) %>%
  summarize(Avg_Severity = mean(CRASH_SEVERITY, na.rm = TRUE))
# Filter data for years 2015 to 2021
filtered_data <- train_data %>%
  filter(YEAR >= 2015 & YEAR <= 2021)

# Create a Date column and calculate counts for each severity level
severity_trend <- filtered_data %>%
  mutate(Date = as.Date(paste(YEAR, MONTH, "01", sep = "-"))) %>% # First day of each month
  group_by(Date, CRASH_SEVERITY) %>%
  summarize(Count = n(), .groups = "drop") # Count occurrences for each severity level
# Aggregate the data
pedestrian_severity <- train_data %>%
  group_by(PEDESTRIAN_ACTION, CRASH_SEVERITY) %>%
  summarize(Count = n(), .groups = "drop")

# Plot
library(ggplot2)
# Aggregate the data
pedestrian_severity <- train_data %>%
  group_by(PEDESTRIAN_ACTION, CRASH_SEVERITY) %>%
  summarize(Count = n(), .groups = "drop")
ggplot(pedestrian_severity, aes(x = PEDESTRIAN_ACTION, y = Count, fill = as.factor(CRASH_SEVERITY))) +
  geom_bar(stat = "identity", position = "fill") + # Use "fill" to show proportions
  labs(
    title = "Crash Severity Levels by Pedestrian Action",
    x = "Pedestrian Action",
    y = "Proportion",
    fill = "Crash Severity"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    plot.title = element_text(hjust = 0.5)
  )

# Aggregate crash severity counts by day of the week
day_severity <- train_data %>%
  group_by(DAY, CRASH_SEVERITY) %>%
  summarize(Count = n(), .groups = "drop")

ggplot(day_severity, aes(x = DAY, y = Count, group = CRASH_SEVERITY, color = as.factor(CRASH_SEVERITY))) +
  geom_line(size = 1.2) +
  geom_point(size = 3) +
  labs(
    title = "Crash Severity Trends by Day of the Week",
    x = "Day of the Week",
    y = "Count",
    color = "Crash Severity"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5),
    axis.text.x = element_text(angle = 45, hjust = 1)
  )
# Assuming `train_data` has columns "ID" (Pedestrian/Bicycle) and "CRASH_SEVERITY"
# Install the package if not already installed
install.packages("tidyverse")

# Load the package
library(tidyverse)

# Summarize crash severity by ID
severity_table <- train_data %>%
  group_by(ID, CRASH_SEVERITY) %>%
  summarize(Count = n(), .groups = "drop") %>%
  pivot_wider(names_from = CRASH_SEVERITY, values_from = Count, values_fill = 0)

# Rename columns for better clarity
colnames(severity_table) <- c("ID", "None", "Possible", "Minor", "Serious", "Fatal")

# Add a Total column
severity_table <- severity_table %>%
  mutate(Total = rowSums(across(where(is.numeric))))

# View the table
print(severity_table)
severity_table_prop <- severity_table %>%
  mutate(across(None:Fatal, ~ . / Total))
library(ggplot2)

ggplot(train_data, aes(x = CRASH_SEVERITY, fill = ID)) +
  geom_bar(position = "dodge") +
  labs(title = "Crash Severity by Non-Motorist Type",
       x = "Crash Severity Level", y = "Count") +
  theme_minimal()
# Load required packages
library(ggplot2)
library(dplyr)

# Create a data frame from the table
data <- data.frame(
  Predictor = c(
    "NUM_VEH_INVOLVED: 4 or More", "LIGHTING: Other", "INTERSECTION: Roundabout",
    "HIGHWAY_CLASS: Interstate Turnpike", "SEX: Not Stated", "NUM_VEH_INVOLVED: Not Stated",
    "SEX: Unknown", "LIGHTING: Unknown", "WEATHER: Severe Crosswind", "WEATHER: Snow",
    "INTERSECTION: Traffic Circle", "INTERSECTION: Y-Intersection", "INTERSECTION: Not an Intersection",
    "WEATHER: Unknown", "NUM_VEH_INVOLVED: 3 Vehicles"
  ),
  Fatal = c(9.64, 10.1, 0, 0, 4.05, 0, 6.45, 7.23, 4.03, 0, 0, 4.74, 0, 4.18, 0),
  Serious = c(7.43, 0, 0, 0, 7.08, 4.61, 0, 0, 0, 8.57, 0, 0, 4.43, 0, 0),
  Minor = c(7.32, 0, 6.42, 0, 0, 0, 0, 0, 0, 0, 6.54, 0, 0, 0, 0),
  Possible = c(6.78, 0, 6.78, 4.57, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5.89)
)

# Melt data into long format for ggplot2
library(tidyr)
data_long <- pivot_longer(train_data, cols = Fatal:Possible, names_to = "Severity", values_to = "Magnitude")

# Plot
ggplot(data_long, aes(x = reorder(Predictor, -Magnitude), y = Magnitude, fill = Severity)) +
  geom_bar(stat = "identity", position = "stack") +
  coord_flip() +
  labs(
    title = "Impact of Predictors on Crash Severity",
    x = "Predictors",
    y = "Magnitude",
    fill = "Severity Level"
  ) +
  theme_minimal()

# Crash Severity by Pedestrian Location
ggplot(OkData, aes(x = PEDESTRIAN_LOCATION, fill = CRASH_SEVERITY)) +
  geom_bar(position = "fill") +
  labs(title = "Crash Severity by Pedestrian Location", x = "Pedestrian Location", y = "Proportion") +
  theme_minimal()

# Weather and Pedestrian Action Interaction
ggplot(OkData, aes(x = PEDESTRIAN_ACTION, fill = WEATHER)) +
  geom_bar(position = "fill") +
  facet_wrap(~CRASH_SEVERITY) +
  labs(title = "Weather and Pedestrian Action Interaction", x = "Pedestrian Action", y = "Proportion") +
  theme_minimal()

ggplot(OkData, aes(x = PEDESTRIAN_ACTION, fill = LIGHTING)) +
  geom_bar(position = "fill") +
  facet_wrap(~CRASH_SEVERITY, nrow = 5) +  # Split facets into two rows
  labs(
    title = " and Pedestrian Action Interaction",
    x = "Pedestrian Action",
    y = "Proportion",
    fill = "Lighting"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 16),
    axis.text.x = element_text(angle = 45, hjust = 1, size = 10),
    legend.position = "right"
  )
severe <- filter(OkData, CRASH_SEVERITY %in% c(4, 5))
non_severe <- filter(OkData, CRASH_SEVERITY %in% c(1, 2, 3))

# Severe plot
ggplot(severe, aes(x = PEDESTRIAN_ACTION, fill = WEATHER)) +
  geom_bar(position = "fill") +
  labs(
    title = "Weather and Pedestrian Action (Severe Crashes)",
    x = "Pedestrian Action",
    y = "Proportion",
    fill = "Weather"
  ) +
  theme_minimal()

# Non-severe plot
ggplot(non_severe, aes(x = PEDESTRIAN_ACTION, fill = WEATHER)) +
  geom_bar(position = "fill") +
  labs(
    title = "Weather and Pedestrian Action (Non-Severe Crashes)",
    x = "Pedestrian Action",
    y = "Proportion",
    fill = "Weather"
  ) +
  theme_minimal()
