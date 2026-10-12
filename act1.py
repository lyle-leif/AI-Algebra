"""Recover scholarship criteria weights and predict student scores.

Each row of ``A`` holds one applicant's ratings on four criteria (GPA, Exam,
Income, Leadership), and ``b`` holds the scholarship score each applicant
received. Solving the linear system ``A @ weights = b`` recovers how much each
criterion contributes to the final score. The recovered weights are then used
to predict scores for a new set of students.
"""

import numpy as np
import pandas as pd
from numpy.linalg import lstsq, inv
from numpy.linalg import solve, inv

# Criteria ratings for four past applicants (rows) across four criteria (columns).
A = np.array([
    [3,2,5,1],
    [4,1,4,2],
    [2,3,4,3],
    [5,2,3,1]
],dtype=float)

# Scholarship score each past applicant received.
b = np.array([90,88,85,92],dtype=float)

# --- Input validation for the system A @ weights = b ---
if A.ndim != 2 or A.shape[0] != A.shape[1]:
    raise ValueError(f"A must be a square 2D matrix, got shape {A.shape}")
if b.ndim != 1 or b.shape[0] != A.shape[0]:
    raise ValueError(f"b must be a 1D array with {A.shape[0]} entries, got shape {b.shape}")
if not (np.all(np.isfinite(A)) and np.all(np.isfinite(b))):
    raise ValueError("A and b must contain only finite numbers (no NaN or inf)")
if np.linalg.matrix_rank(A) < A.shape[0]:
    raise ValueError("A is singular, so the weights cannot be uniquely recovered")

# Solve for the weight of each criterion.
weights = solve(A,b)

criteria=["GPA","Exam","Income","Leadership"]

# Each weight needs a matching criterion name, or zip() below would drop some.
if len(criteria) != len(weights):
    raise ValueError(f"Expected {len(weights)} criteria names, got {len(criteria)}")

print("Recovered Weights")
for c,w in zip(criteria,weights):
    print(f"{c:12s}: {w:.4f}")

# Multiplying A by the recovered weights should reproduce the original scores.
print("\nVerification")
print(A@weights)
print("Original:",b)

# Criteria ratings for the students whose scores we want to predict.
new_student=np.array([
            [5, 5, 5, 5],
            [4, 4, 4, 4],
            [3, 3, 3, 3],
            [2, 2, 2, 2],
            [1, 1, 1, 1]],dtype=float)

student_list = ["Argona, Charmaine", "Esguerra, Lyle Daniel", "Opinion, Zairus", "Seda, Emmanuel" ]

# --- Input validation for the new students ---
if new_student.ndim != 2 or new_student.shape[1] != len(weights):
    raise ValueError(
        f"new_student must have {len(weights)} columns (one per criterion), "
        f"got shape {new_student.shape}"
    )
if not np.all(np.isfinite(new_student)):
    raise ValueError("new_student must contain only finite numbers (no NaN or inf)")

# Predict each student's score as the weighted sum of their ratings.
# Note: zip() stops at the shorter input, so rows without a name are skipped.
for d, y in zip(student_list, new_student):
    score=y@weights
    print(f"\n{d}: {y}")
    print(f"Predicted Scholarship Score: {score:.2f}")

