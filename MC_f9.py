#-*- utf8 -*-

"""
INFLUENCE DE RHO

(Resultats curieux)

"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

""" Parameters """
T = 0.5 ### horizon ###
dt = 0.0005 ### time step ###
a = 10 ### volatility drift ###
chi = 1.0 ### variance of variance ###
sigma0 = 0.15 ### stabilization volatility ###
r = 0.00 ### interest rate ###
mean = [0., 0.] ### Mean value ###


""" Initial condition """
V0 = 0.15*0.15 ### initial variance ###

n = norm.pdf
N = norm.cdf

def bs_price(S, K, T, r, sigma):
    d1 = (np.log(S/K)+(r+sigma*sigma/2.)*T)/(sigma*np.sqrt(T))
    d2 = d1-sigma*np.sqrt(T)
    return S*N(d1)-K*np.exp(-r*T)*N(d2)


def hw_price(S, K, T, r, sigma, beta):
    k = beta*beta * T
    racineT = np.sqrt(T)
    d1 = (np.log(S/K)+(r+sigma*sigma/2.)*T)/(sigma*racineT)
    d2 = d1-sigma*np.sqrt(T)
    c1 = sigma * S * racineT * n(d1)
    t1 = k *(1 + k/4 + k*k/20 + k*k*k/120 + k*k*k*k/840)* (d1 * d2 - 1) / 24
    t2 = k*k * (1 + 17*k/4/6 + 57 * k*k/28/6 + 89 * k * k * k / 112 / 6) *((d1 * d2 -3)*(d1*d2 - 1) - d1*d1 - d2 * d2) / 120
    return S*N(d1)-K*np.exp(-r*T)*N(d2) + c1 * (t1 + t2)


def bs_vega(S, K, T, r, sigma):
    d1 = (np.log(S/K)+(r+sigma*sigma/2.)*T)/(sigma*np.sqrt(T))
    return S * np.sqrt(T)*n(d1)


def find_vol(target_value, S, K, T, r):
    MAX_ITERATIONS = 100
    PRECISION = 1.0e-5

    sigma = 0.5
    for i in range(0, MAX_ITERATIONS):
        price = bs_price(S, K, T, r, sigma)
        vega = bs_vega(S, K, T, r, sigma)
        diff = target_value - price  # our root

        if (abs(diff) < PRECISION):
            return sigma
        if vega==0:
            print('attention')
            return sigma
        sigma = sigma + diff/vega # f(x) / f'(x)

    # value wasn't found, return best guess so far
    return sigma


if __name__ == '__main__' :

    # Initialization
    Ni = round(T/dt)
    t = np.linspace(0, T, Ni)
    V = V0 * np.ones(Ni)
    S = np.ones(Ni)
    nb_sim = 20000
    C = np.zeros(nb_sim)
    biais = np.zeros((3, len(np.arange(0.9,1.21,0.02))))
    j = 0

    # Main loop
    for rho in [-0.1, -0.2, -0.3]:
        for k in range(nb_sim):
            cov = [[1., rho], [rho, 1.]]
            W = np.random.multivariate_normal(mean, cov, Ni-1)
            for i in range(1, Ni):
                sigma = np.sqrt(V[i-1])
                S[i] = S[i-1] * np.exp((r - V[i-1]/2)*dt + W[i-1,0] * np.sqrt(dt * V[i-1]))
                V[i] = V[i-1] * np.exp(W[i-1,1] * chi * np.sqrt(dt))
            C[k] = S[Ni-1]
        itt = 0
        for X in np.arange(0.9,1.21,0.02):
            oo = np.array([c - X if c > X else 0 for c in C])
            biais[j, itt] = find_vol(oo.mean(), 1.0, X, T, r)
            print(j, ' ', itt, ' ', biais[j, itt])
            itt+=1
        j += 1

    # Plot
    plt.plot(np.arange(0.9,1.21,0.02), biais[0,:], color='b', label=r'$\rho = -0.1$')
    plt.plot(np.arange(0.9,1.21,0.02), biais[1,:], color='g', label=r'$\rho = -0.2$')
    plt.plot(np.arange(0.9,1.21,0.02), biais[2,:], color='r', label=r'$\rho = -0.3$')
    plt.ylabel('Implied volatility (%)')
    plt.xlabel(r'$K \div S$')
    plt.legend()
    plt.show()
