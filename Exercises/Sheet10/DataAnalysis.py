import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.rc('font', **{'family':'sans-serif', 'sans-serif': ['Helvetica'], 'size': 22})
plt.rc('text', usetex=True)


data = pd.read_csv('KH.csv')
array = data.to_numpy()
t = array.T[0, :]

rho1 = 1.0
rho2 = 2.0
u1 = 0.3
u2 = -0.3

k = 4 * np.pi
w = k * np.abs(u1 - u2) * np.sqrt(rho1 * rho2) / (rho1 + rho2) * t - 7.5

plt.plot(t, np.log(array.T[1, :]))
plt.plot(t, w)


plt.legend(['Measurement data', 'Growth rate'])
plt.ylabel('Logarithm of the Mean Energy Density $\\log(\\langle \\rho  \\rangle _y)$')
plt.xlabel('Time $t$')
plt.show()

