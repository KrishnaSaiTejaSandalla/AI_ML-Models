import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Loading the dataset from student_data.csv file
data = pd.read_csv("data/student_data.csv")

# Separating features and target
X = data.drop("pass", axis=1)
y = data["pass"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

# Prediction phase
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Model Accuracy:", accuracy)
print("Confusion Matrix:")
print(cm)

with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved successfully!")