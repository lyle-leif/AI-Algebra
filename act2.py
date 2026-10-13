import numpy as np
import pandas as pd
from numpy.linalg import lstsq, inv
from numpy.linalg import solve, inv

np.set_printoptions(precision=4, suppress=True)

data = {
    "GPA": [1.25, 1.50, 2.75, 2.00, 1.30],
    "Exam": [95, 90, 70, 80, 96],
    "Income": [25000, 28000, 90000, 60000, 20000],
    "Leadership": [8, 9, 4, 6, 10],
    "Scholarship": ["Yes", "Yes", "No", "No", "Yes"]
}

df = pd.DataFrame(data)

print("SCHOLARSHIP APPLICANT RECORDS")
print(df)

A = np.array([
    [1.25, 95, 25000, 8],
    [1.50, 90, 28000, 9],
    [2.75, 70, 90000, 4],
    [2.00, 80, 60000, 6],
    [1.30, 96, 20000, 10]
], dtype=float)

print("A = Matrix A")
print("Matrix A:")
print(A)

Scholarship_binary=[1.0 if val== "Yes" else 0.0 for val in data["Scholarship"]]

B = np.array(Scholarship_binary, dtype=float).reshape(-1,1)
b = np.array(Scholarship_binary, dtype=float)

print("B = Matrix B")
print("Matrix B:")
print(B)

weights, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
weights = weights.flatten()

criteria = ["GPA", "Exam", "Income", "Leadership"]

print("\nMODEL WEIGHTS")
for criterion, weight in zip(criteria, weights):
    print(f"{criterion:12s}: {weight:.6f}")

print("\nVerification")
predictions = A @ weights
print("Raw scores:", predictions)
print("Original:",b)

names = ["Charmaine Argona",
         "Lyle Esguerra",
         "Zairus Opinion",
         "Loraine Relativo",
         "Emmanuel Seda"]

data = {
    "GPA": [1.00, 2.00, 2.50, 1.25, 3.00],
    "Exam": [99, 67, 85, 90, 60],
    "Income": [13000, 50000, 30000, 10000, 150000],
    "Leadership": [8, 7, 10, 8, 9]
}

df = pd.DataFrame(data)

G = np.array([
    [1.00, 99, 13000, 8],
    [2.00, 67, 50000, 7],
    [2.50, 85, 30000, 10],
    [1.25, 90, 10000, 8],
    [3.00, 60, 150000, 9]
], dtype=float)

print("\nNEW APPLICANT INFORMATION")
print(df)

predictions = G @ weights
result = []

print("\nSCHOLARSHIP PREDICTION")

for prediction, name in zip(predictions, names):
    print(f"\nName: {name}")
    print(f"Prediction Score: {prediction:.4f}")

    if prediction >= 0.5:
        print("Scholarship Awarded: YES")
        result.append("YES")
    else:
        print("Scholarship Awarded: NO")
        result.append("NO")

new_score = []

for x in predictions:
    new_score.append(round(x, 2))

new_data = {
    "Name" : names,
    "GPA": data["GPA"],
    "Exam": data["Exam"],
    "Income": data["Income"],
    "Leadership": data["Leadership"],
    "Prediction Score": new_score,
    "Scholarship Awarded": result
}

dt = pd.DataFrame(new_data)

print("\nFINAL SCHOLARSHIP APPLICANT RECORDS")
print(dt)