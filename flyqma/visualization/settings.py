__author__ = 'Sebastian Bernasek'

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap


def build_transparent_cmap(fg, bg='w'):
    """
    Construct colormap with linear scaled transparency between <bg> and <fg> colors .
    """
    base_map = LinearSegmentedColormap.from_list(fg, [bg, fg])
    cmap = base_map(np.arange(base_map.N))
    cmap[:,-1] = np.linspace(0, 1, base_map.N)
    cmap = ListedColormap(cmap)
    return cmap


# labels
labelpad = 1
labelsize = 7

# tick labels
tickpad = 1
ticklabelsize = 6

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
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.1
plt.rcParams['savefig.transparent'] = True
