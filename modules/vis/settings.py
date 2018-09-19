__author__ = 'Sebastian Bernasek'

import matplotlib.pyplot as plt

# labels
labelpad = 1
labelsize = 8

# tick labels
tickpad = 1
ticklabelsize = 7

# font params
plt.rcParams['font.size'] = ticklabelsize
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# axes params
plt.rcParams['axes.labelpad'] = labelpad
plt.rcParams['axes.labelsize'] = labelsize
plt.rcParams['axes.labelcolor'] = '000000'
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['axes.edgecolor'] = '000000'

#xtick params
plt.rcParams['xtick.major.width'] = 1
plt.rcParams['xtick.minor.width'] = 1
plt.rcParams['xtick.major.size'] = 3
plt.rcParams['xtick.minor.size'] = 1
plt.rcParams['xtick.major.pad'] = tickpad
plt.rcParams['xtick.labelsize'] = ticklabelsize
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['xtick.color'] = '000000'

# ytick params
plt.rcParams['ytick.major.width'] = 1
plt.rcParams['ytick.minor.width'] = 1
plt.rcParams['ytick.major.size'] = 3
plt.rcParams['ytick.minor.size'] = 1
plt.rcParams['ytick.major.pad'] = tickpad
plt.rcParams['ytick.labelsize'] = ticklabelsize
plt.rcParams['ytick.direction'] = 'out'
plt.rcParams['ytick.color'] = '000000'

# save params
plt.rcParams['savefig.pad_inches'] = 0.1
plt.rcParams['savefig.transparent'] = True
