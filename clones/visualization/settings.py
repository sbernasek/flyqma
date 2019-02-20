__author__ = 'Sebastian Bernasek'

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.gridspec import GridSpec


def default_figure(func):
    """ Decorator for creating axis. """
    def wrapper(*args, ax=None, figsize=(2., 1.25), **kwargs):
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        return func(*args, ax=ax, **kwargs)
    return wrapper


def square_figure(func):
    """ Decorator for creating square axis without spines. """
    def wrapper(*args, ax=None, figsize=(2., 2.), **kwargs):
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        output = func(*args, ax=ax, **kwargs)
        ax.set_aspect(1)
        ax.axis('off')
        return output
    return wrapper


def joint_figure(func):
    """ Decorator for creating joint distribution figure. """

    def format_joint_axis(ax):
        """ Format joint axis. """
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('ln X')
        ax.set_ylabel('<ln X> among neighbors')
        ax.spines['left'].set_position(('outward', 2))
        ax.spines['bottom'].set_position(('outward', 2))

    def wrapper(self, *args, size_ratio=4, figsize=(2., 2.), **kwargs):

        # create figure
        fig = plt.figure(figsize=figsize)
        ratios = [size_ratio/(1+size_ratio), 1/(1+size_ratio)]
        gs = GridSpec(2, 2, width_ratios=ratios,
                      height_ratios=ratios[::-1], wspace=0, hspace=0)
        fig.ax_joint = fig.add_subplot(gs[1, 0])
        fig.ax_xmargin = fig.add_subplot(gs[0, 0])
        fig.ax_ymargin = fig.add_subplot(gs[1, 1])
        fig.ax_xmargin.axis('off')
        fig.ax_ymargin.axis('off')
        format_joint_axis(fig.ax_joint)

        # run plotting function
        output = func(self, fig, *args, **kwargs)

        # invert yaxis (after all plotting is done)
        fig.ax_joint.invert_yaxis()

        return output

    return wrapper


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
