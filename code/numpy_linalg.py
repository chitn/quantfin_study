"""
Using numpy to solve a linear equation Ax = b
"""

# %% load library
import numpy as np


# %% let's try
A = np.array([[ 4,  3, 2],
              [-2,  2, 3],
              [ 3, -5, 2]])
invA = np.linalg.inv(A)

B = np.array([25, -10, -4])

X = np.linalg.inv(A).dot(B)

print(A)
print(invA)
print(X)

X = np.linalg.solve(A, B)
print(X)

upper = np.diag([3,3],1)
mid   = np.diag([4,2,2],0)
lower = np.diag([1,1],-1)
A = upper + mid + lower
print(A)

size  = 5
upper = np.diag(list(range(size  )), 1)
mid   = np.diag(list(range(size+1)), 0)
lower = np.diag(list(range(size  )),-1)
A = upper + mid + lower
print(A)



