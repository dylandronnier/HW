#-*- utf8 -*-

"""
DIFFERENCE ENTRE BS ET HW

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
    t1 = k *(1 + k/4)* (d1 * d2 - 1) / 24
    t2 = k*k *((d1 * d2 -3)*(d1*d2 - 1) - d1*d1 - d2 * d2) / 120
    return S*N(d1)-K*np.exp(-r*T)*N(d2) + c1 * (t1 + t2)

if __name__ == '__main__' :

    abscisse = np.linspace(0.75, 1.25, 30)

    BS = np.array([bs_price(x, 1.0, 0.5, 0., 0.15) for x in abscisse])
    HW = np.array([hw_price(x, 1.0, 0.5, 0., 0.15, 1.0) for x in abscisse])

    plt.plot(abscisse, BS, color='r', label  = r'BS')
    plt.plot(abscisse, BS + 10 * (HW - BS), color='b', label = r'BS + 10(HW-BS)')
    plt.ylabel('Price')
    plt.xlabel(r'$S \div K$')
    plt.legend()
    plt.show()
