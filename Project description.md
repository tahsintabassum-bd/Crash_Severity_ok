This project implemented a comprehensive data preprocessing pipeline and advanced modeling techniques to predict crash severity in Oklahoma. 
Feature selection involved removing irrelevant, redundant, and low-variance features, as well as those with over 30% missing data, ensuring a clean dataset. 
Temporal variables were simplified (e.g., categorizing days into Weekday/Weekend), and high-cardinality categorical variables were consolidated into the top five levels.
Missing values were imputed using the median for numerical features and the mode for categorical features. 
Categorical variables were encoded using one-hot encoding to ensure compatibility with machine learning algorithms, 
while numerical features were standardized using z-score normalization to equalize their contribution. 
Outliers were identified and addressed through statistical methods and visualizations, 
and SMOTE was applied to address class imbalance by oversampling underrepresented crash severity categories. 

A variety of supervised machine learning models were employed to predict crash severity, including
Logistic Regression (LR), Support Vector Machine (SVM), Artificial Neural Network (ANN), Random
Forest (RF), and XGBoost. Additionally, unsupervised learning techniques such as K-means
clustering with t-SNE visualization were used to identify patterns in the data.
Finally, a stacking ensemble approach was implemented, combining base models like Random Forest and XGBoost with a meta-model to improve overall prediction performance.
The performance of each supervised learning model based on cross-
validation accuracy and key metrics (Precision, Recall, F1-Score, and G-Mean) were calculated.
This robust workflow ensured accurate crash severity predictions and valuable insights for targeted safety interventions.
