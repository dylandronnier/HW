#-*- utf8 -*-

"""
CALCULE LE SMILE PAR METHODE DE MONTE-CARLO

(comparaison avec la formule explicite)

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


""" Initial conditions """
V0 = 0.15*0.15 ### Initial variance

n = norm.pdf
N = norm.cdf

""" Some functions """

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

    Ni = round(T/dt)
    t = np.linspace(0, T, Ni)
    V = V0 * np.ones(Ni)
    nb_sim = 20000
    C = np.zeros(len(np.arange(0.8,1.25,0.02)))
    implied = np.zeros( len(np.arange(0.8,1.25,0.02) ))

    # Multiple paths
    for k in range(nb_sim):
        itt = 0
        W = np.random.normal(size=Ni-1)
        for i in range(1, Ni):
            sigma = np.sqrt(V[i-1])
            V[i] = V[i-1] * np.exp((a*(sigma0 - sigma) - chi**2/2)*dt + W[i-1] * chi * np.sqrt(dt))
        for X in np.arange(0.8,1.25,0.02):
            C[itt] += bs_price(1.0, X, T, r, np.sqrt(V.mean()))/nb_sim
            itt+=1

    itt = 0
    for X in np.arange(0.8,1.25,0.02):
        implied[itt] = find_vol(C[itt], 1.0, X, T, r)
        print("Implied volatility : ", implied[itt], " %")
        itt += 1

    smile = [find_vol(hw_price(1.0, x, T, r, 0.15, chi), 1.0, x, T, r) for x in np.arange(0.8,1.25,0.02)]
          
    # Plot
    plt.plot(np.arange(0.8,1.25,0.02), implied, color='g', label=r'$\gamma_\lambda = a(\sigma^* - \sqrt{\nu})$')
    plt.plot(np.arange(0.8,1.25,0.02), smile, color='r', label=r'HW formula')
    plt.ylabel('Implied volatility (%)')
    plt.xlabel(r'$K \div S$')
    plt.legend()
    plt.show()
