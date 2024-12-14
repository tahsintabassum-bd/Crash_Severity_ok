install.packages(c("tidyverse", "fastDummies", "data.table", "GGally", "corrplot", "skimr", 
                   "caret", "xgboost", "e1071", "randomForest", "glmnet", 
                   "pROC", "MLmetrics", "janitor", "plotly", "vroom"))
install.packages("dplyr")  # Install if not already installed
library(dplyr) 
library(forcats)
library(fastDummies)
library(caret)
install.packages("smotefamily")
library(smotefamily)

#whole_oklahoma_state_2015_to_2021_nonmotorist->OkData
#Load_Data
OkData <- read.csv("OKDATA.csv")

..................................................................#Step 1: Data Preparartion..................................................................................................

# Check for missing values
print("Missing values per column:")
colSums(is.na(OkData))  # Replace 'data' with your dataset name

# Drop columns with excessive missing values
columns_to_drop <- c("INJ_TYPE_4", "INJ_TYPE_5","INJ_TYPE_3","INJ_TYPE_2","INJ_TYPE","PERSON_CONDITION","PERSON_ID_INJURY","CRASH_DRUG_RELATED", "BAC",
                     "DOCID", "COUNTY", "CITY", "AGENCY", "STREET_HIGHWAY",
                     "INTERSECTING_ROAD", "ADDRESS", "TRANSPORTED_BY", 
                     "MEDICAL_FACILITY", "OHP_NON", 
                     "LONGITUDE", "LATITUDE", "x", "y", "DATE", 
                     "ObjectId", "WORKZONE_TYPE", "WORKZONE_LOCATION","INJURY_SEVERITY", "AGE","TIME","HOUR","HOUR8","HOUR6", "TOTAL_NONMOTORISTS")
OkData <- OkData %>% select(-all_of(columns_to_drop))
library(dplyr)

# Categorize 'Day' into 'Weekday' and 'Weekend'
OkData <- OkData %>%
  mutate(DAY = case_when(
    DAY %in% c("B MON", "C TUE", "D WED", "E THU", "F FRI") ~ "Weekday",
    DAY %in% c("A SUN", "G SAT") ~ "Weekend",
    TRUE ~ "Other"  # Handles unexpected or undefined values
  ))

# Verify the result
table(OkData$DAY)


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


# Exclude CRASH_SEVERITY from lumping
categorical_columns_to_lump <- setdiff(categorical_columns, "CRASH_SEVERITY")

# Lump each categorical column to the top 5 levels
for (col in categorical_columns_to_lump) {
  OkData[[col]] <- fct_lump_n(OkData[[col]], n = 5)  # Lump to top 5 levels
}

# Verify the levels after lumping
lapply(OkData[categorical_columns_to_lump], levels)


# Drop low-variance variables
low_variance <- nearZeroVar(OkData, saveMetrics = TRUE)  # Identify low-variance variables
low_variance_cols <- rownames(low_variance[low_variance$nzv, ])
OkData <- OkData[, !(names(OkData) %in% low_variance_cols)]


#Encoding data
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



OkData <- dummy_cols(
  OkData,
  select_columns = categorical_columns,
  remove_first_dummy = TRUE,
  remove_selected_columns = TRUE
)
str(OkData$CRASH_SEVERITY)
OkData$CRASH_SEVERITY <- as.factor(OkData$CRASH_SEVERITY)

# Count of each class
class_counts <- table(OkData$CRASH_SEVERITY)
print(class_counts)

# Proportion of each class
class_proportions <- prop.table(class_counts)
print(round(class_proportions, 3))


##################################
#splitting the dataset 

set.seed(123)  # For reproducibility
trainIndex <- createDataPartition(OkData$CRASH_SEVERITY, p = 0.8, list = FALSE)
trainData <- OkData[trainIndex, ]
testData <- OkData[-trainIndex, ]
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


.......................................................#Step 2 : Model Development..............................................................................................
...........................................#Model 1: Multinomial Logistic Regression Model.....................
install.packages("nnet")
library(nnet)
#initial model
LR_model <- multinom(CRASH_SEVERITY ~ ., data = trainData)
summary(LR_model)

# Assess Variable Relevance
# Remove Predictors with High p-Values
# Extract p-values
coeff_summary <- summary(LR_model)
z_scores <- coeff_summary$coefficients / coeff_summary$standard.errors
p_values <- (1 - pnorm(abs(z_scores), 0, 1)) * 2  # Two-tailed test
print(p_values)

# Identify predictors with p > 0.05
irrelevant_vars <- colnames(p_values)[apply(p_values, 2, function(x) all(x > 0.05))]

# Remove irrelevant variables from training and test data
trainData <- trainData[, !(names(trainData) %in% irrelevant_vars)]
testData <- testData[, !(names(testData) %in% irrelevant_vars)]

# Rebuild the model
LR_model_refined <- multinom(CRASH_SEVERITY ~ ., data = trainData)
summary(LR_model_refined)

# Extract full tidy table from the multinomial logistic regression model
#full_tidy_table <- tidy(LR_model_refined)

# View the full table in the R console (optional)
#View(full_tidy_table)

# Save the full table to a CSV file
#write.csv(full_tidy_table, "multinomial_model_summary.csv", row.names = FALSE)

library(dplyr)
library(tidyr)
library(car)
library(car)


coeff_magnitude <- abs(summary(LR_model_refined)$coefficients)
#ranked variable
sorted_coeff <- coeff_magnitude %>%
  as.data.frame() %>%
  mutate(Predictor = rownames(.)) %>%  # Create a new column from rownames
  pivot_longer(cols = -Predictor, names_to = "Outcome_Level", values_to = "Magnitude") %>%
  arrange(desc(Magnitude))
print(head(sorted_coeff, 10))

#Stepwise Selection
LR_model_stepwise <- step(LR_model_refined, direction = "both")
summary(LR_model_stepwise)

# Extract the formula of the final model after stepwise selection
final_formula <- formula(LR_model_stepwise)
print(final_formula)

# List of relevant variables
relevant_vars <- all.vars(final_formula)
print("Relevant Variables from Stepwise Selection:")
print(relevant_vars)

#Identifying direstionality
directionality <- apply(summary(LR_model_stepwise)$coefficients, 2, function(x) mean(x))
print(directionality)

#ranked variable
coeff_df <- as.data.frame(summary(LR_model_stepwise)$coefficients)
coeff_df$Predictor <- rownames(coeff_df)
coeff_df <- coeff_df %>%
  pivot_longer(-Predictor, names_to = "Level", values_to = "Coefficient")

#important var
var_importance <- data.frame(
  Predictor = names(ranked_vars),
  Importance = ranked_vars
)
library(ggplot2)
#visualization of ranked data
ggplot(var_importance, aes(x = reorder(Predictor, -Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  coord_flip() +
  labs(title = "Predictor Importance After Stepwise Selection",
       x = "Predictors", y = "Coefficient Magnitude") +
  theme_minimal()

.........................................................................#Model Evaluation................................

# Predict on test data
#predicted_probs <- predict(LR_model_stepwise, newdata = testData, type = "probs")
#predicted_classes <- predict(LR_model_stepwise, newdata = testData)

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

# Accuracy(we can remove this)
accuracy <- mean(predicted_classes == test_data$CRASH_SEVERITY)
print(paste("Accuracy:", round(accuracy * 100, 2), "%"))

# F1-Score
library(MLmetrics)

f1_scores <- sapply(levels(test_data$CRASH_SEVERITY), function(class) {
  F1_Score(y_pred = predicted_classes == class, y_true = test_data$CRASH_SEVERITY == class)
})
macro_f1 <- mean(f1_scores)
print(f1_scores)
print(paste("Macro-Averaged F1-Score:", round(macro_f1, 2)))

# AUC-ROC
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

