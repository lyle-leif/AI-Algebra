import numpy as np
import pandas as pd
from numpy.linalg import lstsq, inv
from numpy.linalg import solve, inv

A = np.array([
    [3,2,5,1],
    [4,1,4,2],
    [2,3,4,3],
    [5,2,3,1]
],dtype=float)

b = np.array([90,88,85,92],dtype=float)

weights = solve(A,b)

criteria=["GPA","Exam","Income","Leadership"]

print("Recovered Weights")
for c,w in zip(criteria,weights):
    print(f"{c:12s}: {w:.4f}")

print("\nVerification")
print(A@weights)
print("Original:",b)

new_student=np.array([
            [5, 5, 5, 5],
            [4, 4, 4, 4],
            [3, 3, 3, 3],
            [2, 2, 2, 2],
            [1, 1, 1, 1]],dtype=float)

student_list = ["Argona, Charmaine", "Esguerra, Lyle Daniel", "Opinion, Zairus", "Seda, Emmanuel" ]

for d, y in zip(student_list, new_student):
    score=y@weights    
    print(f"\n{d}: {y}")
    print(f"Predicted Scholarship Score: {score:.2f}")

