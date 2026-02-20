import numpy as np
import scipy.linalg

Q = np.zeros((4, 0), dtype=float)
R = np.zeros((0, 0), dtype=float)
v1 = np.array([1.0, 2.0, 3.0, 4.0])
v2 = np.array([1.0, 1.0, 1.0, 1.0])
Q, R = scipy.linalg.qr_insert(Q, R, v1, 0, which="col")
print("After 1 insert:", Q.shape, R.shape)
Q, R = scipy.linalg.qr_insert(Q, R, v2, 1, which="col")
print("After 2 inserts:", Q.shape, R.shape)
