import numpy as np
import matplotlib.pyplot as plt

from ...visualization import default_figure


class ModelSelectionVisualization:
    """ Methods for visualizing model selection procedure. """

    def plot_models(self, panelsize=(3, 2), **kwargs):
        """ Plot model for each number of components. """
        ncols = self.max_num_components - 2
        fig, axes = plt.subplots(ncols=ncols, figsize=(panelsize[0]*ncols, panelsize[1]))
        for i, model in enumerate(self.models):
            model.plot_pdfs(ax=axes[i], **kwargs)
            if i == np.argmin(self.BIC):
                axes[i].set_title('SELECTED')

    @default_figure
    def plot_information_criteria(self,
                                  bic=True,
                                  aic=True,
                                  ax=None,
                                  **kwargs):
        """
        Plot information criteria versus number of components.

        Args:

            bic (bool) - include BIC scores

            aic (bool) - include AIC scores

            ax (matplotlib.axes.AxesSubplot) - if None, create axis

        Returns:

            fig (matplotlib.figure)

        """

        # plot AIC scores
        if aic:
            ind = np.argmin(self.AIC)
            ax.plot(self.num_components, self.AIC, '.-r', label='AIC', markersize=10)
            ax.scatter(self.num_components[ind], self.AIC[ind], s=150, facecolor='y', lw=1, marker=(5, 1), zorder=99)

        # plot BIC scores
        if bic:
            ind = np.argmin(self.BIC)
            ax.plot(self.num_components, self.BIC, '.-b', label='BIC', markersize=10)
            ax.scatter(self.num_components[ind], self.BIC[ind], s=150, facecolor='y', lw=1, marker=(5, 1), zorder=99)

        # format axes
        ax.set_xlabel('Number of components in mixture')
        ax.set_ylabel('Information Criteria')
        ax.set_xticks(self.num_components)
        ax.set_yticks([])
        ax.legend(frameon=False, bbox_to_anchor=(0., 1.02, 1., .102), loc='upper center', ncol=2, mode="center", borderaxespad=0.)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
