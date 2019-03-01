import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def default_figure(func):
    """ Decorator for creating axis. """
    def wrapper(*args, ax=None, figsize=(2., 1.25), **kwargs):
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        output = func(*args, ax=ax, **kwargs)
        if output is None:
            output = plt.gcf()
        return output
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
        if output is None:
            output = plt.gcf()
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

        # set axis limits
        fig.ax_joint.set_xlim(fig.ax_joint.get_ylim())
        fig.ax_xmargin.set_xlim(fig.ax_joint.get_ylim())

        return output

    return wrapper
