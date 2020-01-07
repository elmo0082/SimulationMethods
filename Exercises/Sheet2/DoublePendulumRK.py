import numpy as np
import matplotlib.pyplot as plt

# Define constants of the system:
m1 = .5
m2 = 1.0
l1 = 2.0
l2 = 1.0
###


# Defining the functions f1,f2,f3,f4 of the four-component-vector
def f1(p1, p2, q1, q2):
    return (m2*np.square(l2)*q1 - m2*l1*l2*np.cos(p1-p2)*q2) / (m2*(m1+m2)*np.square(l1*l2) - np.square(m2*l1*l2*np.cos(p1-p2)))


def f2(p1, p2, q1, q2):
    return ((m1+m2)*np.square(l1)*q2 - m2*l1*l2*np.cos(p1-p2)*q1) / (m2*(m1+m2)*np.square(l1*l2) - np.square(m2*l1*l2*np.cos(p1-p2)))


def f3(p1, p2, q1, q2):
    return -m2*l1*l2*f1(p1, p2, q1, q2)*f2(p1, p2, q1, q2)*np.sin(p1-p2) - (m1 + m2)*l1*np.sin(p1)


def f4(p1, p2, q1, q2):
    return m2*l1*l2*f1(p1, p2, q1, q2)*f2(p1, p2, q1, q2)*np.sin(p1-p2) - m2*l2*np.sin(p2)


# Lagrangian function of the system:
def L(p1, p2, q1, q2):
    return .5*m1*np.square(l1*f1(p1, p2, q1, q2)) + \
           .5*m2*(np.square(l1*f1(p1, p2, q1, q2)) + np.square(l2*f2(p1, p2, q1, q2)) + 2*l1*l2*f1(p1, p2, q1, q2)*f2(p1, p2, q1, q2)*np.cos(p1-p2)) - \
           m1*l1*(1 - np.cos(p1)) - m2*(l1*(1-np.cos(p1)) + l2*(1-np.cos(p2)))


# Function to calculate the total energy
def Etot(p1, p2, q1, q2):
    return f1(p1, p2, q1, q2)*q1 + f2(p1, p2, q1, q2)*q2 - L(p1, p2, q1, q2)

# Algorithm with step size h
h = .05

# Initial conditions in rad:
p1 = 50 * np.pi/180
p2 = -120 * np.pi/180
q1 = 0  # Because the time derivatives of p1 and p2 are set to zero
q2 = 0
E0 = Etot(p1, p2, q1, q2)
t = range(0, 2000)
# Array with data points
ArrayE = np.empty((2000, 2))
ArrayP = np.empty((2000, 4))

# Second order Runge Kutta predictor scheme
for i in range(0, 150):
    ArrayE[i][0] = np.abs((Etot(p1, p2, q1, q2) - E0) / E0)
    # k's for p1
    k11 = f1(p1, p2, q1, q2)
    k12 = f1(p1 + k11*h, p2 + k11*h, q1 + k11*h, q2 + k11*h)
    # k'2 for p2
    k21 = f2(p1, p2, q1, q2)
    k22 = f2(p1 + k21*h, p2 + k21*h, q1 + k21*h, q2 + k21*h)
    # k's for q1
    k31 = f3(p1, p2, q1, q2)
    k32 = f2(p1 + k31*h, p2 + k31*h, q1 + k31*h, q2 + k31*h)
    # k's for q2
    k41 = f4(p1, p2, q1, q2)
    k42 = f2(p1 + k41*h, p2 + k41*h, q1 + k41*h, q2 + k41*h)
    # Calculating next step
    p1 = p1 + .5*(k11 + k12) * h
    p2 = p2 + .5*(k21 + k22) * h
    q1 = q1 + .5*(k31 + k32) * h
    q2 = q2 + .5*(k41 + k42) * h
    ArrayP[i][0] = p1
    ArrayP[i][1] = p2

# Plot the trajectory
plt.subplot(2, 2, 1)
plt.title('Trajectory for Heuns method with 150 steps')
plt.plot(l1*np.cos(ArrayP[:,0]), l1*np.sin(ArrayP[:,0]), 'b-')
plt.plot(l1*np.cos(ArrayP[:,0]) + l2*np.cos(ArrayP[:,1]), l1*np.sin(ArrayP[:,0]) + l2*np.sin(ArrayP[:,1]), 'r-')
# Plot the energy
plt.subplot(2, 2, 2)
plt.title('Relative energy error for Heuns method with 150 steps')
plt.plot(t, ArrayE[:, 0], 'k-')

# Again the initial conditions
p1 = 50 * np.pi/180
p2 = -120 * np.pi/180
q1 = 0  # Because the time derivatives of p1 and p2 are set to zero
q2 = 0

# RK4 scheme
for i in range(0, 500):
    ArrayE[i][1] = np.abs((Etot(p1, p2, q1, q2) - E0) / E0)
    # k's for p1
    k11 = f1(p1, p2, q1, q2)
    k12 = f1(p1 + k11 * h / 2, p2 + k11 * h / 2, q1 + k11 * h / 2, q2 + k11 * h / 2)
    k13 = f1(p1 + k12 * h / 2, p2 + k12 * h / 2, q1 + k12 * h / 2, q2 + k12 * h / 2)
    k14 = f1(p1 + k13 * h, p2 + k13 * h, q1 + k13 * h, q2 + k13 * h)
    # k'2 for p2
    k21 = f2(p1, p2, q1, q2)
    k22 = f1(p1 + k21 * h / 2, p2 + k21 * h / 2, q1 + k21 * h / 2, q2 + k21 * h / 2)
    k23 = f1(p1 + k22 * h / 2, p2 + k22 * h / 2, q1 + k22 * h / 2, q2 + k22 * h / 2)
    k24 = f1(p1 + k23 * h, p2 + k23 * h, q1 + k23 * h, q2 + k23 * h)
    # k's for q1
    k31 = f3(p1, p2, q1, q2)
    k32 = f1(p1 + k31 * h / 2, p2 + k31 * h / 2, q1 + k31 * h / 2, q2 + k31 * h / 2)
    k33 = f1(p1 + k32 * h / 2, p2 + k32 * h / 2, q1 + k32 * h / 2, q2 + k32 * h / 2)
    k34 = f1(p1 + k33 * h, p2 + k33 * h, q1 + k33 * h, q2 + k33 * h)
    # k's for q2
    k41 = f4(p1, p2, q1, q2)
    k42 = f1(p1 + k41 * h / 2, p2 + k41 * h / 2, q1 + k41 * h / 2, q2 + k41 * h / 2)
    k43 = f1(p1 + k42 * h / 2, p2 + k42 * h / 2, q1 + k42 * h / 2, q2 + k42 * h / 2)
    k44 = f1(p1 + k43 * h, p2 + k43 * h, q1 + k43 * h, q2 + k43 * h)
    # Calculating next step
    p1 = p1 + .5*(k11 + k12) * h
    p2 = p2 + .5*(k21 + k22) * h
    q1 = q1 + .5*(k31 + k32) * h
    q2 = q2 + .5*(k41 + k42) * h
    ArrayP[i][2] = p1
    ArrayP[i][3] = p2

# Plot the trajectory
plt.subplot(2, 2, 3)
plt.title('Trajectory for RK4 with 500 steps')
plt.plot(l1*np.cos(ArrayP[:,2]), l1*np.sin(ArrayP[:,2]), 'b-')
plt.plot(l1*np.cos(ArrayP[:,2]) + l2*np.cos(ArrayP[:,3]), l1*np.sin(ArrayP[:,2]) + l2*np.sin(ArrayP[:,3]), 'r-')
# Plot the energy
plt.subplot(2, 2, 4)
plt.title('Relative energy error for RK4 with 500 steps')
plt.plot(t, ArrayE[:,1], 'k-')
plt.show()

