#-*- utf8 -*-

""" 

RECUPERE L'HISTORIQUE DES DONNEES DU VXD

"""

from yahoo_finance import Share
import matplotlib.pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator, DateFormatter, datestr2num

""" Récupere les données """
vxn = Share('^VXN')
historic = vxn.get_historical('2012-11-01', '2015-06-05')
date = datestr2num([x['Date'] for x in historic])
value = [float(x['Close']) for x in historic]

""" Plot """
fig, ax = plt.subplots()
ax.plot_date(date, value, '-', color='purple')
ax.set_ylabel('VIX')
plt.show()


