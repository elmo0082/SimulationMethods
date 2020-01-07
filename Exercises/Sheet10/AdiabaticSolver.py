import numpy as np
import matplotlib.pyplot as plt
from codes_hydro.hydro_adi_classic import hydro_adi_classic_one_timestep
from codes_hydro.hydro_adi_riemann import hydro_adi_riemann_one_timestep

plt.rc('font', **{'family':'sans-serif', 'sans-serif': ['Helvetica']})
plt.rc('text', usetex=True)

t = 5000
N = 2000
L = 200

dx = L/N


def initialize_initial_conditions(N):
    rho = np.zeros((N + 2))
    p = np.zeros((N + 2))
    u = np.zeros((N + 2))

    rho[0:int(np.floor(N/2))] = 1e5
    rho[int(np.floor(N/2)):] = 1.24e4

    p[0:int(np.floor(N / 2))] = 1
    p[int(np.floor(N / 2)):] = 0.1

    return rho, u, p


# Recalculate dt at every step so that CFL is always satisfied
def calculate_dt(q):
    if np.amax(np.abs(q[1, :])) != 0:
        cfl = dx / (np.amax(np.abs(q[1, :])))
        dt = 0.6 * cfl
    else:
        dt = 0.1
    return dt


# Function to plot results
def plot_results(q, N):
    x = np.linspace(-100+dx, 100+dx, N+2)

    plt.subplot(3, 1, 1)
    plt.ylabel('Density Field $\\rho$')
    plt.xlabel('Position $x$')
    plt.plot(x, q[0, :])

    plt.subplot(3, 1, 2)
    plt.ylabel('Velocity Field $u$')
    plt.xlabel('Position $x$')
    plt.plot(x, q[1, :])

    plt.subplot(3, 1, 3)
    plt.ylabel('Pressure Field $P$')
    plt.xlabel('Position $x$')
    plt.plot(x, q[2, :])


def run_classical_solver(rho, u, p, time, N):
    q = np.zeros((3, N+2))
    q[0, :] = rho
    q[1, :] = u
    q[2, :] = p
    dt = 0.1
    t = 0
    while t < time:
        q = hydro_adi_classic_one_timestep(q, dx, dt)
        dt = calculate_dt(q)
        t += dt
        print('Time:', t)

    plot_results(q, N)


def run_riemann_solver(rho, u, p, time, N):
    q = np.zeros((3, N+2))
    q[0, :] = rho
    q[1, :] = u
    q[2, :] = p
    dt = 0.1
    t = 0
    while t < time:
        q = hydro_adi_riemann_one_timestep(q, dx, dt)
        dt = calculate_dt(q)
        t += dt
        print('Time:', t)

    plot_results(q, N)


rho, u, p = initialize_initial_conditions(N)
run_classical_solver(rho, u, p, t, N)
run_riemann_solver(rho, u, p, t, N)
plt.show()

