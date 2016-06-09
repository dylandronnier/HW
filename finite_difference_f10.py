#-*- utf8 -*-

""" 

CALCULE LE PRIX D'UNE OPTION DANS LE MODELE DE HULL-WHITE PAR EDP

modele decorrele - drif de la variance instantanee nul

"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

""" Parameters """

T = 0.5 ### horizon ###
dt = 0.00005 ### time step ###
beta = 1.0
K = 1.0
dx = 0.05
dsigma = 0.01
stock = np.arange(0.0, 2.0, dx)
variance = np.arange(0.0, 1.0, dsigma)
Big1 = np.array([[x*x*y for y in variance] for x in stock] )
Big2 = np.array([[y*y for y in variance] for x in stock] )

""" Function initialization """
f = np.array([[x - K if x > K else 0. for V in variance] for x in stock] )
fprev = f
m,n = f.shape
print('Taille du maillage : ', m, ' X ', n)

""" Main Loop """
for i in range(round(T/dt)):
    f[1:(m-1),1:(n-1)] = fprev[1:(m-1),1:(n-1)] + (fprev[2:,1:(n-1)]  + f[:(m-2),1:(n-1)] - 2*f[1:(m-1),1:(n-1)])*Big1[1:(m-1), 1:(n-1)]*(dt/dx/dx/2)
    + beta*beta*(fprev[1:(m-1),2:] + f[1:(m-1),:(n-2)] - 2*f[1:(m-1),1:(n-1)])*Big2[1:(m-1), 1:(n-1)]*(dt/dsigma/dsigma/2)
    
    fprev = f

""" Plot """
plt.plot(stock[15:25], f[15:25,1], color='r', label = r'$\sigma = 0.1$')
plt.plot(stock[15:25], f[15:25,2], color='g', label = r'$\sigma = 0.14$')
plt.plot(stock[15:25], f[15:25,4], color='b', label = r'$\sigma = 0.2$')
plt.ylabel('Price')
plt.xlabel(r'$S \div K$')
plt.legend()
plt.show()
