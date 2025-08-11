# Objective
The primary aim of this project is to build a predictive model that identifies telecom customers who are likely to churn and uncovers the key factors contributing to the churn. This model processes data using a machine learning pipeline that includes feature engineering, feature selection and training data using multiple classification algorithms to ensure robust performance.
<br>

# Dataset Overview:
The dataset has been sourced from **[Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)**. It represents customer behaviors in a fictional telecommunications company that includes **7,043 records** across **21 features**.
1. **Demographic attributes**: describe basic customer profiles including gender, senior citizen, partner, dependents.
2. **Service Usage & Subscriptions**: types of services and subscriptions used by the customer, for example, phone service, tech support, online security, device protection etc.
3. **Account & Billing Details**: capture the types of contracts, tenure and billing information of the customers.
4. **Target Varibale - Churn**: Churn indicates whether a customer ended their service *(Yes/ No)*.
<br>

# Steps Performed:
- *Exploratory Data Analysis (EDA)* to derive key insights from the dataset and understand the distribution of independent and dependent variables.
- Initial *data inspection and cleaning* (data types conversion, missing values etc.)
- *Feature engineering* to derive additional features from existing ones to enhance the predictive power.
- Selecting most relevant features using *feature selection* based on exploratory insights, domain knowledge and potential predictive power.
- *Data transformation and encoding* to make the data model ready.
- *Splitting dataset* into training and test sets.
- *Model building* and evaluating the best classification model on test set.
- *Model fine-tuning* to achieve the highest cross-validated accuracy.
- Derving *feature importance* to identify most important features for targeting high-risk customer segments.
<br>

# Models Used:
1. **Logistic Regression**
   
   | Evaluation Metric | Value   |
   | ----------------- | --------|
   | Accuracy          | 0.79    |
   | ROC AUC Score     | 0.828   |
   | Precision         | 0.63    |
   | Recall            | 0.52    |
   | F1 Score          | 0.57    |
   
2. **Random Forest**

   | Evaluation Metric | Value   |
   | ----------------- | --------|
   | Accuracy          | 0.767   |
   | ROC AUC Score     | 0.808   |
   | Precision         | 0.57    |
   | Recall            | 0.48    |
   | F1 Score          | 0.52    |
   
3. **XG Boost**
   
   | Evaluation Metric | Value   |
   | ----------------- | --------|
   | Accuracy          | 0.769   |
   | ROC AUC Score     | 0.805   |
   | Precision         | 0.58    |
   | Recall            | 0.49    |
   | F1 Score          | 0.53    |

**Logistic Regression** identified as the best-performing model, achieving the highest ROC AUC score of **0.828**.
<br>

# Hyperparameter Tuning:

Fine tuning was performed using **Grid Search CV** and the final tuned model was achieved with the following configuration:
- *C*: 10
- *Penalty*: l2
- *Solver*: liblinear
- *Best cross-validation ROC AUC*: 0.8435

Predictions on test set using the fine-tuned model:
| Evaluation Metric | Value   |
| ----------------- | --------|
| Accuracy          | 0.79    |
| ROC AUC Score     | 0.827   |
| Precision         | 0.84    |
| Recall            | 0.89    |
| F1 Score          | 0.86    |

## Usage:
To run the ipynb locally:
- Clone the repository using git clone.
- Install the dependencies
  `!pip install -q pandas numpy seaborn matplotlib scikit-learn xgboost`
- Download the dataset from Kaggle and place it in the same directory as ipynb.
- Run the code cells using Jupyter or any other compatible tool.
