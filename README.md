# Customer Churn Prediction

This project aims to predict customer churn for a telecommunications company. Customer churn is a critical problem for businesses, as retaining existing customers is often more cost-effective than acquiring new ones. By identifying customers at high risk of churning, the company can proactively implement retention strategies.

## Dataset

The dataset used for this project is `WA_Fn-UseC_-Telco-Customer-Churn.csv`. It contains information about a fictional telecommunications company's customers, including their services, account information, and whether they churned.

Key features in the dataset include:
-   `customerID`: Customer ID
-   `gender`: Whether the customer is a male or a female
-   `SeniorCitizen`: Whether the customer is a senior citizen or not (1, 0)
-   `Partner`: Whether the customer has a partner or not (Yes, No)
-   `Dependents`: Whether the customer has dependents or not (Yes, No)
-   `tenure`: Number of months the customer has stayed with the company
-   `PhoneService`: Whether the customer has phone service or not (Yes, No)
-   `MultipleLines`: Whether the customer has multiple lines or not (Yes, No, No phone service)
-   `InternetService`: Customer’s internet service provider (DSL, Fiber optic, No)
-   `OnlineSecurity`: Whether the customer has online security or not (Yes, No, No internet service)
-   `OnlineBackup`: Whether the customer has online backup or not (Yes, No, No internet service)
-   `DeviceProtection`: Whether the customer has device protection or not (Yes, No, No internet service)
-   `TechSupport`: Whether the customer has tech support or not (Yes, No, No internet service)
-   `StreamingTV`: Whether the customer has streaming TV or not (Yes, No, No internet service)
-   `StreamingMovies`: Whether the customer has streaming movies or not (Yes, No, No internet service)
-   `Contract`: The contract term of the customer (Month-to-month, One year, Two year)
-   `PaperlessBilling`: Whether the customer has paperless billing or not (Yes, No)
-   `PaymentMethod`: The customer’s payment method (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic))
-   `MonthlyCharges`: The amount charged to the customer monthly
-   `TotalCharges`: The total amount charged to the customer
-   `Churn`: Whether the customer churned or not (Yes or No) - **Target Variable**

## Methodology

The project follows a standard machine learning pipeline:

### 1. Data Collection and Pre-processing
-   The `customerID` column was dropped as it's not relevant for modeling.
-   Missing values in `TotalCharges` (represented as spaces) were replaced with '0.0' and the column was converted to a float data type.
-   The target variable `Churn` was label encoded (Yes=1, No=0).

### 2. Exploratory Data Analysis (EDA)
-   **Numerical Features**: Histograms and box plots were used to understand the distribution and identify outliers for `tenure`, `MonthlyCharges`, and `TotalCharges`. A correlation heatmap was generated to show relationships between these features.
-   **Categorical Features**: Count plots were used to visualize the distribution of unique values within each categorical feature.

### 3. Data Preprocessing
-   All categorical features (including the target `Churn`) were converted into numerical representations using `LabelEncoder` from `sklearn.preprocessing`. The encoders for each column were saved using `pickle` for later use in the predictive system.

### 4. Training and Testing the Dataset
-   The dataset was split into training and testing sets (80% training, 20% testing).
-   To address class imbalance in the target variable (`Churn`), the Synthetic Minority Oversampling Technique (SMOTE) was applied to the training data. This balances the number of 'Churn' (minority class) and 'No Churn' (majority class) instances.

### 5. Model Training
-   Three classification models were trained and evaluated using 5-fold cross-validation on the SMOTE-resampled training data:
    -   Decision Tree Classifier
    -   Random Forest Classifier
    -   XGBoost Classifier
-   Random Forest Classifier showed the highest average cross-validation accuracy.

### 6. Model Evaluation
-   The best-performing model, Random Forest, was evaluated on the unseen test set (`X_test`, `y_test`).
-   Evaluation metrics included:
    -   Accuracy Score
    -   Confusion Matrix
    -   Classification Report (Precision, Recall, F1-score)

### 7. Predictive System
-   The trained Random Forest model and the fitted `LabelEncoder` objects (saved in `customer_churn_model.pkl` and `encoders.pkl` respectively) are loaded.
-   A function or script is provided to take new, unseen customer data as input, apply the same preprocessing steps using the saved encoders, and then use the loaded model to predict churn likelihood.

## Results

The Random Forest model achieved an accuracy of approximately 77.86% on the test set. The classification report provides further details on precision, recall, and f1-score for both churn and non-churn classes.

## Usage

To use this project:
1.  **Clone the repository.**
2.  **Ensure you have the required libraries installed:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `sklearn`, `imblearn`, `xgboost`.
3.  **Run the Jupyter Notebook/Colab notebook** step-by-step to understand the data processing and model training.
4.  **To make new predictions:**
    -   Load the `customer_churn_model.pkl` and `encoders.pkl` files.
    -   Prepare your input data in a dictionary format, matching the original column names.
    -   Use the loaded encoders to transform the categorical features of your input data.
    -   Use the loaded model's `predict()` or `predict_proba()` method to get predictions.



```
