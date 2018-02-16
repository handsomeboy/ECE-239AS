import numpy as np
from scipy import linalg
a = np.array([[0.8, 0.6], [0.6, -0.8]])
print(np.matmul(a, a.T))
eigenvals, eigenvecs = linalg.eig(a)
print(eigenvals)
print(eigenvecs)