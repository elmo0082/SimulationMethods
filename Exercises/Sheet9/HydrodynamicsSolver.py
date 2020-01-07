import numpy as np
import matplotlib.pyplot as plt
from codes_hydro.hydro_iso_riemann import *

N = 20000
L = 200
dx = L/N
timesteps = 10000


# Create initial density distribution
def density_distribution(dimension, length):
    rho = np.zeros((dimension + 2, 1))
    for i in range(0, dimension + 1):
        x = i * dx - length / 2
        rho[i] = 1 + np.exp(-np.power(x, 2)/200)
    return rho


# Create initial velocity field
def initial_flux(dimension):
    flux = np.zeros((dimension + 2, 1))
    return flux


# Function to perform an advection step in Ex1.2 sheet 8
def advect(q, v, dx, dt):
    flux = np.zeros_like(v)
    ipos = np.where(v >= 0.)[0]
    ineg = np.where(v < 0.)[0]
    flux[ipos] = q[ipos]*v[ipos]
    flux[ineg] = q[ineg+1]*v[ineg]
    qnew = q.copy()
    qnew[1:-1] -= dt * (flux[1:] - flux[:-1]) / dx
    qnew[0] = qnew[-2]
    qnew[-1] = qnew[1]
    return qnew


# Function to perform one time step from provided from .zip-file
def hydro_iso_classic_one_timestep(q, cs, dx, dt):
    rho = q[0, :]
    print(rho)
    u = q[1, :]/rho
    uint = 0.5 * (u[1:] + u[:-1])
    q[0, :] = advect(q[0, :], uint, dx, dt)
    q[0, 0] = q[0, -1]
    q[1, :] = advect(q[1, :], uint, dx, dt)
    p = q[0, :] * cs**2
    q[1, 1:-1] += - dt * (p[2:] - p[:-2]) / (2*dx)
    return q


# Function to run the hydro-dynamics solver for fixed dt
def run_solver_fixed_dt():
    cs = 1  # Speed of sound is set equal to 1
    x = np.linspace(-L/2, L/2, N + 2)
    q = np.zeros((2, N + 2))
    q[0, :] = density_distribution(N, L)[:, 0]
    q[1, :] = initial_flux(N)[:, 0]

    snaps = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    time = 0
    dt = 0.05
    configure_pyplot()
    for i in range(0, timesteps):
        q = hydro_iso_classic_one_timestep(q, cs, dx, dt)

        snap, new_dt = snapshot(time, snaps, dt)
        if snap:
            plt.plot(x, q[0, :], label='Plot of the density at ' + str(time + new_dt) + ' s')
        time += dt
    plt.legend(loc="upper right")


# Function to run the hydro-dynamics solver for variable dt
def run_solver_variable_dt():
    cs = 1  # Speed of sound is set equal to 1
    x = np.linspace(-L/2, L/2, N + 2)
    q = np.zeros((2, N + 2))
    q[0, :] = density_distribution(N, L)[:, 0]
    q[1, :] = initial_flux(N)[:, 0]
    snaps = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    # snaps = np.linspace(0, 250, 400)

    time = 0
    dt = 0.001
    for i in range(0, timesteps):
        print("Time:", time, dt)

        snap, custom_dt = snapshot(time, snaps, dt)
        if snap:
            q = hydro_iso_classic_one_timestep(q, cs, dx, custom_dt)
            time += custom_dt
            # plt.cla()

            configure_pyplot()

            plt.plot(x, q[0, :], 'k-', label='Plot of the density at ' + str(time)[0:5] + ' s')
            # plt.legend(loc="upper right")
            # plt.savefig('./movie/hydro_' + frame_index(i) + '.png')
            time += calculate_dt(q)
        else:
            q = hydro_iso_classic_one_timestep(q, cs, dx, dt)
            time += dt
        dt = calculate_dt(q)


# Function to run the hll solver
def run_hll_solver():
    # Speed of sound is set equal to 1
    x = np.linspace(-L / 2, L / 2, N + 2)
    q = np.zeros((2, N + 2))
    q[0, :] = density_distribution(N, L)[:, 0]
    q[1, :] = initial_flux(N)[:, 0]
    cs = np.ones_like(q[1, :])

    snaps = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    time = 0
    dt = 0.005
    configure_pyplot()
    for i in range(0, timesteps):
        print(time)
        q = hydro_iso_classic_one_timestep(q, cs, dx, dt)

        snap, new_dt = snapshot(time, snaps, dt)
        if snap:
            plt.plot(x, q[0, :], label='Plot of the density at ' + str(time + new_dt) + ' s')
        time += dt

    plt.legend(loc="upper right")


# Function that calculates whether we want to have to do a snapshot
def snapshot(time, snaptimes, current_dt):
    snaptimes = np.array(snaptimes)
    index = np.where((snaptimes <= (time + current_dt)) & (time <= snaptimes))[0]
    if snaptimes[index].size == 1:
        return True, np.abs(snaptimes[index] - time)[0]
    else:
        return False, 0


# Function to calculate the CFL value
def calculate_dt(q):
    if np.amax(np.abs(q[1, :])) != 0:
        cfl = dx / (np.amax(np.abs(q[1, :])))
        dt = 0.4 * cfl
    else:
        dt = 0.1
    return dt


# Reconfigure canvas for every plot
def configure_pyplot():
    # Use Latex
    font = {'family': 'serif',
            'size': 22}
    plt.rc('font', **font)
    plt.rc('text', usetex=True)

    plt.axis([-L / 2 - 1, L / 2 + 1, 0.75, 3.5])
    plt.grid(1)
    plt.xlabel('Position $x$')
    plt.ylabel('Density $\\rho$')


def frame_index(i):
    if i < 10:
        return "0000" + str(i)
    elif i < 100:
        return "000" + str(i)
    elif i < 1000:
        return "00" + str(i)
    elif i < 10000:
        return "0" + str(i)
    else:
        return str(i)

#run_solver_fixed_dt()
run_solver_variable_dt()
run_hll_solver()
plt.show()
