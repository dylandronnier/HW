#-*- utf8 -*-

"""
CALCULE LE SMILE DE VOLATILITE A PARTIR DE LA FORMULE DE HULL-WHITE

"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

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

        price = price
        diff = target_value - price  # our root

        if (abs(diff) < PRECISION):
            return sigma
        sigma = sigma + diff/vega # f(x) / f'(x)

    # value wasn't found, return best guess so far
    return sigma

if __name__ == '__main__' :

    abscisse = np.linspace(0.75, 1.25, 40)
    smile = np.array([[find_vol(hw_price(1.0, x, 0.5, 0., 0.15, beta), 1.0, x, 0.5, 0.0) for x in abscisse] for beta in [0.9, 1.0, 1.1]])


    # Plot
    plt.plot(abscisse, smile[0,:], color='b', label = r'$\beta = 0.9$')
    plt.plot(abscisse, smile[1,:], color='r', label = r'$\beta = 1.0$')
    plt.plot(abscisse, smile[2,:], color='g', label = r'$\beta = 1.1$')

    plt.ylabel('Implied volatility (%)')
    plt.xlabel(r'$K \div S$')
    plt.legend()
    plt.show()
