
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import pandas as pd

# Load the Titanic dataset
data = sns.load_dataset('titanic')
#print(data.head())
#print(data.columns)

# Display missing values
print(data.isnull().sum())

data['age'].fillna(data['age'].median(), inplace=True)
data['embarked'].fillna(data['embarked'].mode()[0], inplace=True)
data.drop(columns=['deck'], inplace=True)
data['embark_town'].fillna(data['embark_town'].mode()[0], inplace=True)

print(data.isnull().sum())

# Box plot for 'age' column
sns.boxplot(data['age'])
plt.title('Age Outliers')
plt.show()

# Box plot for 'fare' column
sns.boxplot(data['fare'])
plt.title('Fare Outliers')
plt.show()

# Calculate Q1 and Q3 for 'age'
Q1 = data['age'].quantile(0.25)
Q3 = data['age'].quantile(0.75)

# Calculate IQR for 'age'
IQR = Q3 - Q1

# Print Q1, Q3, and IQR for 'age'
print("Q1 (25th percentile for age):", Q1)
print("Q3 (75th percentile for age):", Q3)
print("IQR for age:", IQR)

# Calculate Q1 and Q3 for 'fare'
q1 = data['fare'].quantile(0.25)
q3 = data['fare'].quantile(0.75)

# Calculate IQR for 'fare'
iqr = q3 - q1

# Print Q1, Q3, and IQR for 'fare'
print("Q1 (25th percentile for fare):", q1)
print("Q3 (75th percentile for fare):", q3)
print("IQR for fare:", iqr)



# Function to handle outliers by capping them to the upper/lower bounds
def handle_outliers(column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Cap values outside the bounds
    data[column] = np.where(data[column] < lower_bound, lower_bound, data[column])
    data[column] = np.where(data[column] > upper_bound, upper_bound, data[column])


# Apply to 'age' and 'fare'
handle_outliers('age')
handle_outliers('fare')

# Re-plot box plots
sns.boxplot(data['age'])
plt.title('Age after Handling Outliers')
plt.show()

sns.boxplot(data['fare'])
plt.title('Fare after Handling Outliers')
plt.show()


# Initialize the scaler
scaler = MinMaxScaler()

# Select the columns you want to normalize
columns_to_normalize = ['age', 'fare']  # Example columns

# Apply the scaler to the selected columns and create a DataFrame
data[columns_to_normalize] = scaler.fit_transform(data[columns_to_normalize])

# Check the result
print(data[columns_to_normalize].head())

# Creating new features
data['family_size'] = data['sibsp'] + data['parch'] + 1
data['is_alone'] = (data['family_size'] == 1).astype(int)
#data['title'] = data['who'].str.extract(' ([A-Za-z]+)\.', expand=False)

# Binning or discretizing features
bins = [0, 12, 18, 35, 60, np.inf]
labels = ['Child', 'Teen', 'Young Adult', 'Adult', 'Senior']
data['age_group'] = pd.cut(data['age'], bins=bins, labels=labels)

data['fare_group'] = pd.qcut(data['fare'], 4, labels=['Low', 'Medium', 'High', 'Very High'])

# Encoding categorical variables
data = pd.get_dummies(data, columns=['sex', 'embarked', 'age_group'], drop_first=True)

# Interaction features
data['pclass_sex'] = data['pclass'].astype(str) + '_' + data['sex_male'].astype(str)

# Scaling new features if necessary
scaler = MinMaxScaler()
data[['family_size', 'fare']] = scaler.fit_transform(data[['family_size', 'fare']])

print(data.head())
#print(data.columns)

# Step 1: Split data into features (X) and target (y)
X = data.drop(columns=['survived'])  # Replace 'survived' with your target column
y = data['survived']

# Step 2: Handle categorical features if needed (convert them into numerical)
X = pd.get_dummies(X, drop_first=True)

# Step 3: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Train Logistic Regression
log_reg = LogisticRegression(random_state=42, max_iter=1000)  # max_iter for convergence
log_reg.fit(X_train, y_train)

# Step 5: Train a Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
rf_model.fit(X_train, y_train)

# Step 6: Get feature importance scores
importances = rf_model.feature_importances_
feature_names = X.columns

# Step 7: Create a DataFrame for better visualization
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Step 8: Display the most important features
print(feature_importance_df)

# Step 9: Select features with importance above a certain threshold (optional)
threshold = 0.01  # Adjust threshold as needed
selected_features = feature_importance_df[feature_importance_df['Importance'] > threshold]['Feature']
print(f"Selected Features: {selected_features.tolist()}")

# Step 10: Make Predictions
log_reg_preds = log_reg.predict(X_test)
rf_preds = rf_model.predict(X_test)

# Step 11: Evaluate Logistic Regression
print("Logistic Regression Results:")
print("Accuracy:", accuracy_score(y_test, log_reg_preds))
print("Classification Report:\n", classification_report(y_test, log_reg_preds))
print("Confusion Matrix:\n", confusion_matrix(y_test, log_reg_preds))

# Step 12o: Evaluate Random Forest
print("\nRandom Forest Results:")
print("Accuracy:", accuracy_score(y_test, rf_preds))
print("Classification Report:\n", classification_report(y_test, rf_preds))
print("Confusion Matrix:\n", confusion_matrix(y_test, rf_preds))