import pickle
import numpy as np

# Load saved model
with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)

# Collecting input from user
study_hours = float(input("Enter study hours: "))
attendance = float(input("Enter attendance percentage: "))
previous_score = float(input("Enter previous score: "))

input_data = np.array([[study_hours, attendance, previous_score]])

# Predict the output
prediction = model.predict(input_data)

if prediction[0] == 1:
    print("The student is likely to PASS")
else:
    print("The student is likely to FAIL")