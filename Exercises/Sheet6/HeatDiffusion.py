import numpy as np
import matplotlib.pyplot as plt

e = 1
D = 1
T_0 = 1
L = 1
N = 100


# Cast differential operator into trigonal matrix form
def initialize_matrix(N):
    diagonal = np.ones((N, 1))
    upper = np.ones((N, 1))
    lower = np.ones((N, 1))

    diagonal = np.multiply(diagonal, -2)
    diagonal[0] = 1
    diagonal[N-1] = 1
    lower[N-1] = 0
    upper[0] = 0
    return lower, diagonal, upper


# Initialization of boundary condition etc.
def initialize_b(T):
    h = 2 * L / N
    result = (-1*e*h**2) / D * np.ones((N, 1))
    result[0] = T
    result[N-1] = T
    return result


# Function to multiply a trigonal matrix and a vector
def trigonal_multiplication(lower, diagonal, upper, vector):
    result = np.zeros((N, 1))
    result[0] = diagonal[0] * vector[0]
    result[N-1] = diagonal[N-1] * vector[N-1]
    for i in range(1, N-1):
        result[i] = lower[i] * vector[i-1] + diagonal[i] * vector[i] + upper[i] * vector[i+1]
    return result


# Function that implements the forward-elimination, backward-substitution algorithm
def forward_elimination(lower, diagonal, upper, b):
    for i in range(0, N-2):
        if diagonal[i] > 0:
            m = (-1) * np.abs(lower[i+1]/diagonal[i])
        else:
            m = np.abs(lower[i + 1] / diagonal[i])
        lower[i+1] = lower[i+1] + m * diagonal[i]
        diagonal[i+1] = diagonal[i+1] + m * upper[i]
        b[i+1] = b[i+1] + m * b[i]
    return diagonal, b


# Function that implements the backward-substitution step
# The output vector contains the values for T_i
def backward_substitution(diagonal, b):
    result = np.zeros((N, 1))
    result[N-1] = b[N-1]
    result[0] = b[0]
    for i in range(2, N):
        result[N-i] = (b[N-i] - result[N-i+1])/diagonal[N-i]
    return result


# Function that implements a Jacobi iteration
def jacobi_step(lower, diagonal, upper, x, b):
    zero = np.zeros((N, 1))
    p = trigonal_multiplication(lower, zero, upper, x)
    x = trigonal_multiplication(zero, 1./diagonal, zero, b + p)
    return x


# Create array with spacing h and N points
def create_x_array():
    h = 2*L/N
    result = np.zeros((N, 1))
    for i in range(-int(np.floor(N/2)), int(np.floor(N/2))):
        result[int(np.floor(N/2)+i)] = h*i
    return result


# Function to run the full matrix algorithm
def run_matrix_algorithm():
    l, d, u = initialize_matrix(N)
    b = initialize_b(T_0)
    a, b = forward_elimination(l, d, u, b)
    y = backward_substitution(a, b)

    # Verify solution and calculate residual
    l, d, u = initialize_matrix(N)
    b = initialize_b(T_0)
    print('Residual:', trigonal_multiplication(l, d, u, y) - b)

    x = create_x_array()
    plt.plot(x, y, 'k--')


# Function to run the full jacobi algorithm
def run_jacobi_algorithm(steps):
    x = create_x_array()
    l, d, u = initialize_matrix(N)
    b = initialize_b(T_0)
    y = np.zeros((N, 1))
    for i in range(0, steps):
        y = jacobi_step((-1) * l, d, (-1) * u, y, b)
        plt.plot(x, y)


# Initializes the restriction matrices
def initialize_restriction_matrices():
    R1 = np.multiply((1/4), [[2.,1.,0,0,0,0,0,0,0],
                  [0,1.,2.,1.,0,0,0,0,0],
                  [0,0,0,1.,2.,1.,0,0,0],
                  [0,0,0,0,0,1.,2.,1.,0],
                  [0,0,0,0,0,0,0,1.,2.]])
    R2 = np.multiply((1/4), [[2.,1.,0,0,0],
                  [0,1.,2.,1.,0],
                  [0,0,0,1.,2.]])
    R3 = np.multiply((1/4), [[2.,1.,0],
                  [0,1.,2.]])
    return R1, R2, R3


# Initializes the prolongation matrices
def initialize_prolongation_matrices():
    P0 = np.multiply((1/2), [[2.,0,0,0,0],
                  [1.,1.,0,0,0],
                  [0,2.,0,0,0],
                  [0,1.,1.,0,0],
                  [0,0,2.,0,0],
                  [0,0,1.,1.,0],
                  [0,0,0,2.,0],
                  [0,0,0,1.,1.],
                  [0,0,0,0,2.]])
    P1 = np.multiply((1/2), [[2.,0,0],
                  [1.,1.,0],
                  [0,2.,0],
                  [0,1.,1.],
                  [0,0,2.]])
    P2 = np.multiply((1/2), [[2.,0],
                  [1.,1.],
                  [0,2.]])
    return P0, P1, P2


def calculate_transpose_and_restrictions():
    A = [[1.,0,0,0,0,0,0,0,0],
         [1.,-2.,1.,0,0,0,0,0,0],
         [0,1.,-2.,1.,0,0,0,0,0],
         [0,0,1.,-2.,1.,0,0,0,0],
         [0,0,0,1.,-2.,1.,0,0,0],
         [0,0,0,0,1.,-2.,1.,0,0],
         [0,0,0,0,0,1.,-2.,1.,0],
         [0,0,0,0,0,0,1.,-2.,1.],
         [0,0,0,0,0,0,0,0,1.]]
    R1, R2, R3 = initialize_restriction_matrices()
    P0, P1, P2 = initialize_prolongation_matrices()

    print(np.transpose(P0) - 2 * R1)
    print(np.transpose(P1) - 2 * R2)
    print(np.transpose(P2) - 2 * R3)


def V_shape(R1, A, y, b):
    l, d, u = initialize_matrix(N)

    y = jacobi_step(l, d, u, y, b)
    residual = b - np.matmul(A, y)

    residual = np.matmul(R1, residual)
    return y


run_matrix_algorithm()
run_jacobi_algorithm(30)

plt.show()

