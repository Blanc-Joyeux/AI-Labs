import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Load the dataset from an Excel file
df = pd.read_excel('synthetic_telecom_churn_dataset.xlsx')

# Exploratory Data Analysis
print(df.head())
print(df.info())

# Preprocess the data
# Convert columns to numeric, coercing errors to NaN
numeric_cols = df.select_dtypes(include=['number']).columns
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Fill missing values for numeric columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# Encode categorical variables
df = pd.get_dummies(df, columns=['region'], drop_first=True)

# Normalize numerical features
scaler = StandardScaler()
df[['age', 'income', 'monthly_minutes', 'monthly_data_gb', 'support_tickets', 'monthly_bill', 'outstanding_bal']] = scaler.fit_transform(
    df[['age', 'income', 'monthly_minutes', 'monthly_data_gb', 'support_tickets', 'monthly_bill', 'outstanding_balance']])

# Split the data into training and testing sets
X = df.drop(columns=['customer_id', 'churn'])
y = df['churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("AUC-ROC:", roc_auc_score(y_test, y_pred))

# Optional: Monitor model performance over time manually
# This can be done by retraining the model periodically and evaluating performance