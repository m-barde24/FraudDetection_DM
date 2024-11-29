# Fraud Detection Using Classification and Frequent Pattern Mining

# Importing Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from mlxtend.frequent_patterns import apriori
import os
os.environ['MPLCONFIGDIR'] = os.getcwd()

# Step 1: Load the Dataset (First 1000 Rows)
# Provide the correct path to the dataset
dataset_path = 'creditcard_2023.csv'  # Change this to the full path if needed
try:
    data = pd.read_csv(dataset_path, nrows=2000)  # Load only the first 1000 rows
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print(f"Error: The file '{dataset_path}' was not found. Please check the file path.")
    exit()  # Stop execution if the dataset is missing

# Step 2: Data Exploration
print("\nDataset Info:\n", data.info())
print("\nMissing Values:\n", data.isnull().sum())
print("\nClass Distribution:\n", data['Class'].value_counts())

# Visualizing Class Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='Class', data=data, hue='Class', palette='coolwarm', legend=False)

plt.title('Class Distribution')
plt.xticks([0, 1], ['Non-Fraud (0)', 'Fraud (1)'])
plt.show()

# Step 3: Data Preprocessing
# Scale the 'Amount' feature
scaler = MinMaxScaler()
data['Normalized_Amount'] = scaler.fit_transform(data[['Amount']])
data.drop('Amount', axis=1, inplace=True)  # Drop the original 'Amount' column

# Step 4: Frequent Pattern Mining
# Separate non-fraudulent transactions for pattern mining
non_fraudulent = data[data['Class'] == 0]

# Create a simple binary matrix for frequent pattern mining
basket = (non_fraudulent[['Time', 'Normalized_Amount']]
          .astype(str)  # Convert to string for pattern grouping
          .apply(lambda x: '_'.join(x), axis=1)
          .value_counts()
          .reset_index())
basket.columns = ['Itemset', 'Count']

# Convert itemsets to a DataFrame for Apriori algorithm
basket_df = basket.set_index('Itemset')['Count'].to_frame().T
basket_df[basket_df > 0] = 1  # Binary matrix for Apriori

# Apply Apriori algorithm
frequent_patterns = apriori(basket_df, min_support=0.1, use_colnames=True)
print("\nFrequent Patterns:\n", frequent_patterns)

# Step 5: Classification
# Splitting the data into features (X) and target (y)
X = data.drop(['Class', 'Time'], axis=1)  # Drop 'Time' for simplicity
y = data['Class']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Random Forest Classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

# Step 6: Evaluation
# Make predictions
y_pred = classifier.predict(X_test)

# Evaluation Metrics
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Fraud', 'Fraud'], yticklabels=['Non-Fraud', 'Fraud'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Step 7: Visualizations
# Feature Distributions
plt.figure(figsize=(8, 6))
sns.histplot(data['Normalized_Amount'], kde=True, bins=20, color='purple')
plt.title('Distribution of Normalized Transaction Amount')
plt.xlabel('Normalized Amount')
plt.show()

# Feature Importance (Random Forest)
importances = classifier.feature_importances_
feature_names = X.columns
plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=feature_names, palette='viridis')
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()
plt.close()
print("\nScript completed successfully.")