import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.sparse.linalg

n = 20
h = 1


def initialize_rho(N):
    rho = np.zeros((N, N, N))
    k = int(N/2) - 1
    rho[k:k+2, k:k+2, k:k+2] = 1
    return rho


def initialize_phi(N):
    phi = np.zeros((N, N, N))
    return phi


def create_laplacian(N):
    A = np.zeros((N, N, N, N, N, N))
    for i in range(0, N):
        for j in range(0, N):
            for k in range(0, N):
                for a in range(0, N):
                    for b in range(0, N):
                        for c in range(0, N):
                            A[i, j, k, a, b, c] = operator_entry(i, j, k, a, b, c, N)

    return A


def kronecker(i, j):
    if i == j:
        return 1
    else:
        return 0


def operator_entry(i, j, k, a, b, c, N):
    if boundary(i, j, k, N):
        entry = kronecker(i, a) * kronecker(j, b) * kronecker(k, c)
    else:
        entry = -6 * kronecker(i, a) * kronecker(j, b) * kronecker(k, c) \
                + kronecker(i + 1, a) * kronecker(j, b) * kronecker(k, c) \
                + kronecker(i, a) * kronecker(j + 1, b) * kronecker(k, c) \
                + kronecker(i, a) * kronecker(j, b) * kronecker(k + 1, c) \
                + kronecker(i - 1, a) * kronecker(j, b) * kronecker(k, c) \
                + kronecker(i, a) * kronecker(j - 1, b) * kronecker(k, c) \
                + kronecker(i, a) * kronecker(j, b) * kronecker(k - 1, c)
    return entry


def boundary(i, j, k, N):
    if i == 0 or i == N-1:
        return True
    elif j == 0 or j == N-1:
        return True
    elif k == 0 or k == N-1:
        return True
    else:
        return False


b = 4 * np.pi * h**2 * np.reshape(initialize_rho(n), n**3)
print('Creating operator matrix...')
a = create_laplacian(n)
A = np.reshape(a, (n**3, n**3))
np.set_printoptions(threshold=np.inf)
print('Running BiCGstab...')
phi = scipy.sparse.linalg.bicgstab(A, b)
P = np.reshape(phi[0], (n, n, n))

fig = plt.figure()
ax = plt.axes(projection='3d')
x = np.linspace(0, n, n)
y = np.linspace(0, n, n)
X, Y = np.meshgrid(x, y)
print(P[5, :, :])
ax.plot_wireframe(X, Y, P[5, :, :])
plt.show()

