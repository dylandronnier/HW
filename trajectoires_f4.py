#-*- utf8 -*-

import matplotlib.pyplot as plt
import numpy as np

"""
SIMULATION DE 3 TRAJECTOIRES SOUS PROBABILITE HISTORIQUE

"""

""" Parameters """
T = 0.272 ### horizon ###
dt = 0.0001
rho = 0.1 ### correlation between the variables ###
mean = [0., 0.]
cov = [[1., rho], [rho, 1.]]
sigma0 = 0.145
a = 800
chi = 2.0
mu = 0.075

""" Initial conditions """
V0 = 0.145*0.145
S0 = 1.0


if __name__ == '__main__' :

    N = round(T/dt)
    t = np.linspace(0, T, N)
    V = V0 * np.ones((N,3))
    S = S0 * np.ones((N,3))

    ### Main loop ###
    for j in range(3):
        W = np.random.multivariate_normal(mean, cov, N-1)

        for i in range(1, N):
            sigma = np.sqrt(V[i-1,j])
            S[i,j] = S[i-1,j] * np.exp((mu - V[i-1,j]/2)*dt + W[i-1,1] * np.sqrt(dt * V[i-1,j]))
            V[i,j] = V[i-1,j] * np.exp((a*(sigma0 - sigma) - chi**2/2)*dt + W[i-1,0] * chi * np.sqrt(dt))
   
    ### Plot the final curve ###
    f, axarr = plt.subplots(2, sharex=True)
    axarr[0].plot(365 * t, S[:,0], color='b')
    axarr[1].plot(365 * t, np.sqrt(V[:,0]), color='b')
    axarr[0].plot(365 * t, S[:,1], color='r')
    axarr[1].plot(365 * t, np.sqrt(V[:,1]), color='r')
    axarr[0].plot(365 * t, S[:,2], color='g')
    axarr[1].plot(365 * t, np.sqrt(V[:,2]), color='g')
    axarr[0].set_title('Security price')
    axarr[1].set_title('Volatility')
    axarr[1].set_xlabel('Time (days)')
    plt.show()
