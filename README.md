# Loan Approval Prediction Project

## Project Objective
This project aims to build and evaluate various machine learning models to predict loan approval status based on a given dataset. The goal is to identify the most effective model for this classification task.

## Data Source
The analysis uses the `loan_approval_dataset.csv` file, containing various features related to applicants and their financial status.

## Data Preprocessing and Exploration
The initial steps involved thorough data preprocessing and exploratory data analysis:
- **Column Cleaning**: Removed leading/trailing spaces from column names and string values.
- **Categorical Encoding**: Converted 'education' (ordinal) and 'self_employed', 'loan_status' (nominal) categorical features into numerical representations using OrdinalEncoder and mapping.
- **Missing Values**: Checked for and confirmed no null values in the dataset.
- **Outlier Detection & Removal**: Visualized outliers using box plots and iteratively removed them from 'commercial_assets_value', 'residential_assets_value', and 'bank_asset_value' using the IQR method.
- **Duplicate Removal**: Identified and removed duplicate rows from the dataset.
- **QQ-Plots**: Generated QQ-plots for numerical features to assess their distribution normality.
- **Correlation Analysis**: Performed correlation analysis and visualized it with a heatmap to identify highly correlated features. Redundant features ('income_annum', 'loan_amount', 'luxury_assets_value') were dropped to mitigate multicollinearity.

## Feature Engineering: Principal Component Analysis (PCA)
Principal Component Analysis (PCA) was applied for dimensionality reduction and to explore its impact on model performance:
- An elbow curve was plotted to determine the optimal number of principal components, suggesting 7 components for 90% explained variance.
- Models were trained and evaluated with PCA using 2 components for visualization and 6 components for performance comparison.

## Clustering
The dataset was also analyzed using K-Means clustering after PCA, with evaluation metrics like Davies-Bouldin Index, Silhouette Score, and Dunn Index to determine the optimal number of clusters.

## Model Training and Evaluation
Several classification and regression models were trained and evaluated based on accuracy, confusion matrix, classification report, MSE, MAE, and RMSE:

### Classification Models:
-   **Logistic Regression**
-   **Decision Tree Classifier**
-   **Random Forest Classifier**
-   **Gradient Boosting Classifier**
-   **XGBoost Classifier**

### Regression Models (for `loan_status` as a continuous target):
-   **Linear Regression** (with and without PCA)
    -   Residual plots and density graphs were used for analysis.
-   **Polynomial Regression** (with and without PCA, for varying degrees)
    -   Evaluation of MSE, MAE, and RMSE to find the best polynomial degree.
    -   Visualizations of regression boundaries in 3D for PCA=2.
-   **Lasso Regression** (Polynomial degree 2, with and without PCA, for different alpha values)
-   **Ridge Regression** (Polynomial degree 2, with and without PCA, for different alpha values)
    -   Visualizations of regression boundaries in 3D for PCA=2.
-   **Elastic Net Regression** (Polynomial degree 2, with and without PCA, for different alpha and l1_ratio values)
    -   Visualizations of regression boundaries in 3D for PCA=2.

## Key Findings :
- The classification models (Decision Tree, Random Forest, XGBoost) generally achieved high accuracy in predicting loan status.
- Ridge Regression showed promising results compared to Lasso and Elastic Net in certain configurations, indicating its effectiveness in handling multicollinearity and preventing overfitting.
- PCA influenced the performance of regression models, with different numbers of components yielding varied results.

This notebook provides a comprehensive approach to tackling the loan approval prediction problem, from data preparation to model deployment and evaluation.
