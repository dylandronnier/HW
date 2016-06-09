#-*- utf8 -*-

"""
INFLUENCE DE BETA SUR LE BIAIS

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

if __name__ == '__main__' :

    abscisse = np.linspace(0.9, 1.1, 40)

    HW = np.array([[hw_price(x, 1.0, 0.5, 0., 0.15, v0) for x in abscisse] for v0 in [0.0, 0.8, 1.0, 1.1]])

    # Plot
    plt.plot([0.9, 1.1], [0, 0], color='black')
    plt.plot(abscisse, (HW[1,:] - HW[0,:])/HW[0,:] * 100, color='b', label = r'$\beta = 0.8$')
    plt.plot(abscisse, (HW[2,:] - HW[0,:])/HW[0,:] * 100, color='r', label = r'$\beta = 1.0$')
    plt.plot(abscisse, (HW[3,:] - HW[0,:])/HW[0,:] * 100, color='g', label = r'$\beta = 1.1$')
    plt.ylabel('Price Bias (%)')
    plt.xlabel(r'$S \div K$')
    plt.legend(loc='lower right')
    plt.show()
