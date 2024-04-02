# -DS_Industrial-Copper-Modeling
 DS_Industrial Copper Modeling

import pandas as pd

# Load dataset
data = pd.read_csv('industrial_copper_dataset.csv')

# Explore the dataset
print(data.head())

# Preprocess the data (handle missing values, encode categorical variables, etc.)
# Example:
# data.dropna(inplace=True)
# data = pd.get_dummies(data)

# Split data into features (X) and target variable (y)
X = data.drop('target_variable_column_name', axis=1)
y = data['target_variable_column_name']

# Split data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a machine learning model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate the model
from sklearn.metrics import accuracy_score
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
