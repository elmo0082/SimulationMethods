import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

font = {'family' : 'serif',
        'weight' : 'bold',
        'size'   : 22}

plt.rc('font', **font)
plt.rc('text', usetex=True)
plt.xlabel('Number of Particles [1]')
plt.ylabel('Calculation Time [s]')

n = [5000, 10000, 20000, 40000]
tree = [79.62, 187.43, 434.21, 1001.59]
exact = [131.78, 526.40, 2097.46, 8297.63]

plt.loglog(n, tree, '.', label="Calculation time for the Barnes-Hut tree algorithm.")
plt.loglog(n, exact, '.', label="Calculation time for brute-force algorithm.")


def myComplexFunc(x, a, b):
    return a * np.power(x, b)


x = np.logspace(0, 4, base=10)
popt, pcov = curve_fit(myComplexFunc, n, tree)
plt.plot(n, myComplexFunc(n, *popt), 'g-', label="Exponential fit: $0.0027 \cdot x^{1.209}$")
plt.legend(loc='upper left')
print(popt)
print("Approximation for $10^{10}$ particles.", (0.0027 * (10**10)**1.209) / (365*24*60*60))

plt.show()
