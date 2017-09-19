# Hull-White stochastic volatility model

## Presentation

Le [modèle de Hull et White](https://en.wikipedia.org/wiki/Hull%E2%80%93White_model) est un modèle basé sur l'équation de Black-Scholes qui suppose que la volatilité n'est pas constante et déterministe mais dépend du temps et est aléatoire. Ce modèle permet notamment d'expliquer le smile de volatilité.

Dans ce projet réalisé dans le cadre du cours de Mathématiques Financières de l'Ecole Nationale des Ponts et Chaussées, on se propose d'étudier les propriétés mathématiques du modèle et de le simuler pour pricer des options européennes.

## Travail réalisé

Dans la première partie du [rapport](https://github.com/dylandronnier/HW/blob/master/HW_projet.pdf)￼￼￼￼, on montre l'incomplétude du marché puis on price les calls européens dans le cas où le mouvement brownien intervenant dans la dynamique du et celui intervenant dans la dynamique de la volatilité sont décorrélés.

Dans la seconde partie simule ensuite des dynamiques de Hull et White. On price les options en utilisant une formule close, la résolution par EDP (différence finie) ou la méthode de Monte-Carlo. Chacun des scipts Python correspond à une figure sur le rapport. Par exemple `MC_f9.py` est le script Python permettant de tracer la figure 9 du rapport.


## Bibliothèques

Pour pouvoir lancer les scripts Python, les bibliothèques suivantes sont nécessaires:
* [Numpy](http://www.numpy.org/) et [Scipy](https://www.scipy.org/)
* [Yahoo! Finance](https://pypi.python.org/pypi/yahoo-finance) pour récupérer les prix d'actions
*  [Matplotlib](https://matplotlib.org/) pour l'affichage des graphiques
