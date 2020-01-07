import numpy as np
import matplotlib.pyplot as plt

N = 100
L = 10
dx = L/N
timesteps = 10
dt = 3/timesteps


# Function to create a vector that contains the discrete values of the field
def create_field(dimension, left_boundary, right_boundary, length):
    u = np.zeros((dimension + 2, 1))
    for i in range(1, dimension + 1):
        dist = i * dx - length / 2
        if dist < 0:
            u[i] = 1
    u[0] = left_boundary
    u[dimension + 1] = right_boundary
    return u, dimension


# Function to create a top-hat shaped initial condition
def create_top_hat_field(dimension, length):
    u = np.zeros((dimension + 2, 1))
    for i in range(0, dimension + 2):
        dist = i*dx - length/2
        if dist >= np.ceil(-L/4) and dist <= np.floor(L/4):
            u[i] = 1
    return u, dimension


# Function to create the velocity field for Ex 1.2 with given initial conditions
def velocity(shape, dimension):
    v = np.zeros((dimension + 1, 1))
    if shape == 'constant':
        v = np.ones((dimension + 1, 1))
    if shape == 'linear_convergence':
        for i in range(0, dimension + 1):
            v[i] = -2*(i*dx - L/2) / L
    return v


# Implementing the symmetric numerical scheme given in Ex 1.1
def symmetric_scheme(left_boundary, right_boundary):
    u, dim = create_field(100, left_boundary, right_boundary, L)
    v = velocity(shape='constant', dimension=dim)
    for i in range(0, timesteps):
        for j in range(1, dim):
            u[j] = u[j] - v[j] * (u[j+1] - u[j-1])/(2*dx) * dt
    return u


# Implementing the upwind scheme with a asymmetrical realization of the derivative
def upwind_scheme(left_boundary, right_boundary):
    u, dim = create_field(100, left_boundary, right_boundary, L)
    v = velocity(shape='constant', dimension=dim)
    for i in range(0, timesteps):
        for j in range(1, dim):
            u[j] = u[j] - v[j] * (u[j] - u[j - 1]) / dx * dt
    return u


# Implementing the upwind scheme with the index shift
def upwind_scheme_reversed(left_boundary, right_boundary):
    u, dim = create_field(100, left_boundary, right_boundary, L)
    v = velocity(shape='constant', dimension=dim)
    for i in range(0, timesteps):
        for j in range(2, dim):
            u[j] = u[j] - v[j] * (u[j + 1] - u[j]) / dx * dt
    return u


# Function to perform an advection step in Ex1.2
def advect(u, dim, shape):
    v = velocity(shape=shape, dimension=dim)
    for j in range(1, dim):
        u[j] = u[j] - v[j] * (u[j] - u[j - 1]) / dx * dt
    return u


# Function to run the advection subroutine
def run_advection(shape, initial_condition=None):
    u, dim = create_field(100, 1, 0, L)
    if initial_condition == 'top_hat':
        u, dim = create_top_hat_field(100, L)
    for i in range(0, timesteps):
        u = advect(u, dim, shape)
    return u


def plot_01():
    x = np.linspace(-L/2, L/2, 102)
    y1 = symmetric_scheme(1, 0)
    y2 = upwind_scheme(1, 0)
    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.legend(['Symmetric Scheme', 'Upwind Scheme'], loc=3)


def plot_02():
    x = np.linspace(-L/2, L/2, 102)
    y1 = upwind_scheme_reversed(0, 1)
    plt.plot(x, y1)
    plt.legend(['Shifted Upwind Scheme'], loc=3)


def plot_03():
    x = np.linspace(-L/2, L/2, 102)
    #y1 = upwind_scheme(0.5, 0)
    y2 = upwind_scheme(1, 0.5)
    #plt.plot(x, y1)
    plt.plot(x, y2)
    plt.legend(['Upwind Scheme with right boundary = 0.5'], loc=3)


def plot_04():
    x = np.linspace(-L / 2, L / 2, 102)
    y1 = run_advection(shape='linear_convergence', initial_condition='top_hat')
    plt.plot(x, y1)
    plt.legend(['Upwind Scheme with new algorithm'], loc=3)

plot_03()

plt.ylabel("Solution u(x,t)")
plt.xlabel("x")
plt.show()

