# Import libraries

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# Load dataset

df = pd.read_csv("heart-disease ITP (1).csv")

# Separating independent and dependent columns

X = df.drop("target", axis=1)
y = df["target"]

# Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
X,
y,
test_size=0.2,
random_state=4
)

# Preprocessing

preprocessor = ColumnTransformer([
(
"scale",
MinMaxScaler(),
["age", "trestbps", "chol", "thalach", "oldpeak"]
)
], remainder="passthrough")

# Pipeline using Random Forest

pipeline = Pipeline([
(
"preprocessor",
preprocessor
),
(
"model",
RandomForestClassifier(
n_estimators=300,
max_depth=6,
min_samples_split=4,
min_samples_leaf=2,
max_features="sqrt",
random_state=42
)
)
])

# Train the model

pipeline.fit(X_train, y_train)

# Evaluate the model
# Training Accuracy
train_accuracy = pipeline.score(X_train, y_train)

# Test Accuracy
test_accuracy = pipeline.score(X_test, y_test)

print("Training Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)# Save the trained pipeline

joblib.dump(
pipeline,
"heart_disease_pipeline.pkl"
)

print("Pipeline trained and saved successfully!")

