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
                     "MEDICAL_FACILITY", "OHP_NON","HOUR4", "HOUR8", 
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


OkData$CRASH_SEVERITY <- factor(OkData$CRASH_SEVERITY, levels = c("None","Minor","Possible","Serious","Fatal"))
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

ggplot(OkData, aes(x = NUM_VEH_INVOLVED, fill = CRASH_SEVERITY)) +
  geom_bar(position = "fill") +
  labs(title = "Crash Severity by Number of Vehicles Involved",
       x = "Number of Vehicles Involved", y = "Proportion") +
  theme_minimal()

library(dplyr)
OkData %>%
  group_by(HIGHWAY_CLASS, LIGHTING) %>%
  summarize(Avg_Severity = mean(as.numeric(CRASH_SEVERITY))) %>%
  ggplot(aes(x = HIGHWAY_CLASS, y = Avg_Severity, fill = LIGHTING)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Interaction Between Highway Class and Lighting",
       x = "Highway Class", y = "Average Severity") +
  theme_minimal()

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
    title = "AUC-ROC Curves for Crash Severity Classes",
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
# Install and load necessary library
if (!requireNamespace("keras", quietly = TRUE)) install.packages("keras")
library(keras)
install.packages("reticulate")
library(reticulate)

reticulate::py_config()
Yes
reticulate::install_miniconda()
install.packages("keras")
library(keras)
install_keras()  # Installs TensorFlow and dependencies

# Step 1: Prepare Data
# Scale numeric features
numeric_features <- names(train_data)[sapply(train_data, is.numeric)]
numeric_features <- setdiff(numeric_features, "CRASH_SEVERITY")  # Exclude target variable
train_data[numeric_features] <- scale(train_data[numeric_features])
test_data[numeric_features] <- scale(test_data[numeric_features])

# Convert target variable to categorical
train_labels <- to_categorical(train_data$CRASH_SEVERITY - 1)  # Keras uses 0-based indexing
test_labels <- to_categorical(test_data$CRASH_SEVERITY - 1)

# Remove target variable from features
train_features <- as.matrix(train_data[, -which(names(train_data) == "CRASH_SEVERITY")])
test_features <- as.matrix(test_data[, -which(names(test_data) == "CRASH_SEVERITY")])

# Step 2: Define the Neural Network
model <- keras_model_sequential() %>%
  layer_dense(units = 128, activation = "relu", input_shape = ncol(train_features)) %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = ncol(train_labels), activation = "softmax")  # Output layer

# Compile the model
model %>% compile(
  optimizer = "adam",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

# Step 3: Train the Neural Network
history <- model %>% fit(
  x = train_features,
  y = train_labels,
  epochs = 50,
  batch_size = 32,
  validation_split = 0.2,
  verbose = 1
)

# Step 4: Evaluate the Model
model %>% evaluate(test_features, test_labels)

# Step 5: Predict and Analyze Results
predicted_probs <- model %>% predict(test_features)
predicted_classes <- apply(predicted_probs, 1, which.max) - 1  # Convert back to 0-based

# Confusion Matrix
library(caret)
conf_matrix <- confusionMatrix(as.factor(predicted_classes), as.factor(test_data$CRASH_SEVERITY - 1))
print(conf_matrix)

# Plot Training History
plot(history)

# Step 6: Visualize ROC Curves for Each Class
library(pROC)
roc_results <- list()
for (class in 0:(ncol(test_labels) - 1)) {
  binary_response <- test_labels[, class]
  roc_results[[class + 1]] <- roc(binary_response, predicted_probs[, class + 1])
}

# Plot ROC Curves
roc_data <- do.call(rbind, lapply(1:length(roc_results), function(i) {
  data.frame(
    Sensitivity = roc_results[[i]]$sensitivities,
    Specificity = 1 - roc_results[[i]]$specificities,
    Class = paste("Class", i - 1)
  )
}))

library(ggplot2)
ggplot(roc_data, aes(x = Specificity, y = Sensitivity, color = Class)) +
  geom_line(size = 1) +
  labs(
    title = "Neural Network AUC-ROC Curves",
    x = "1 - Specificity (False Positive Rate)",
    y = "Sensitivity (True Positive Rate)",
    color = "Class"
  ) +
  theme_minimal()

