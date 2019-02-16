import numpy as np
import matplotlib.pyplot as plt

from .classification import MixtureModelClassifier, BivariateMixtureClassifier
from ..visualization.settings import default_figure


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


class UnivariateModelSelection(ModelSelectionVisualization):

    def __init__(self, values, classify_on,
                 min_num_components=3,
                 max_num_components=8):
        """
        Perform model selection by choosing the model that minimizes BIC score.

        Args:

            values (np.ndarray[float]) - 1D array of sample values

            classify_on (str) - attribute label for sample values

            min_num_components (int) - minimum number of components in mixture

            max_num_components (int) - maximum number of components in mixture

        """

        self.values = values
        self.classify_on = classify_on
        self.min_num_components = min_num_components
        self.max_num_components = max_num_components
        self.num_components = range(min_num_components, max_num_components+1)
        self.models = self.fit_models()

    @staticmethod
    def fit_model(values, num_components, **kwargs):
        """ Fit model with specified number of components. """
        return MixtureModelClassifier(values,
                                    num_components=num_components,
                                    num_labels=num_components,
                                    **kwargs)

    def fit_models(self):
        """ Fit model with each number of components. """

        # define parameters
        args = (self.values,)
        kwargs = dict(classify_on=self.classify_on)

        # fit models
        models = []
        for num_components in self.num_components:
            model = self.fit_model(self.values, num_components, **kwargs)
            models.append(model)

        return models

    @property
    def BIC(self):
        """ BIC scores for each model. """
        return np.array([model.model.BIC for model in self.models])

    @property
    def BIC_optimal(self):
        """ Model with BIC optimal number of components. """
        return self.models[np.argmin(self.BIC)]

    @property
    def AIC(self):
        """ AIC scores for each model. """
        return np.array([model.model.AIC for model in self.models])

    @property
    def AIC_optimal(self):
        """ Model with AIC optimal number of components. """
        return self.models[np.argmin(self.AIC)]


class BivariateModelSelection(UnivariateModelSelection):

    @staticmethod
    def fit_model(values, num_components, **kwargs):
        """ Fit model with specified number of components. """
        return BivariateMixtureClassifier(values,
                                        num_components=num_components,
                                        num_labels=num_components,
                                        **kwargs)



# import scipy.stats as st
# from scipy.signal import argrelextrema
# from matplotlib.gridspec import GridSpec


# class Merge:
#     """
#     Class for merging distributions.
#     """

#     def __init__(self, support, densities, weights, key=None):
#         self.support = support
#         self.densities = densities
#         self.num_components = densities.shape[0]
#         self.support_size = densities.shape[1]
#         self.weights = weights.reshape(-1, 1)

#         # build pairwise distance matrix between components
#         self.distances = self.build_distance_matrix()

#         # initialize key mapping components before and after aggregation
#         if key is None:
#             key = dict(enumerate(np.arange(self.num_components)))
#         self.key = key

#     @staticmethod
#     def jensen_shannon_distance(x, y):
#         """ Returns Jensen-Shannon distance between <x> and <y>. """
#         return np.sqrt(0.5*(st.entropy(x, y, base=2)+st.entropy(y, x, base=2)))

#     def build_distance_matrix(self):
#         """ Returns matrix of pairwise Jensen-Shannon distances. """
#         yy, xx = np.meshgrid(*(np.arange(self.num_components),)*2)
#         xxd = self.densities[xx].reshape(-1, self.support_size)
#         yyd = self.densities[yy].reshape(-1, self.support_size)
#         distance = np.array([self.jensen_shannon_distance(x, y) for x, y in zip(xxd, yyd)]).reshape(*xx.shape)
#         return distance

#     @staticmethod
#     def find_neighbors(distance_matrix):
#         """ Returns ordered list of closest neighboring components. """
#         tri_ind = np.triu_indices(distance_matrix.shape[0], k=1)
#         sort_ind = np.argsort(distance_matrix[tri_ind])
#         ordered_triu = [indices[sort_ind] for indices in tri_ind]
#         return np.vstack(ordered_triu).T

#     @property
#     def neighbors(self):
#         """ Ordered list of neighboring components. """
#         return self.find_neighbors(self.distances)

#     def plot_distance_matrix(self, include_nearest=True, cmap=plt.cm.viridis_r, figsize=(3, 3), ax=None):
#         """ Plot distance matrix. """
#         if ax is None:
#             fig, ax = plt.subplots(figsize=figsize)
#         ax.tick_params(labelsize=8, pad=3)
#         mask = np.zeros(self.distances.shape)
#         mask[np.triu_indices(self.num_components, k=0)] = 1
#         distances = np.ma.masked_array(data=self.distances, mask=mask)
#         cmap.set_bad('k', alpha=0.8)
#         ax.imshow(distances, cmap=cmap)
#         ax.invert_yaxis()
#         if include_nearest:
#             s = 100 * (self.num_components)**(0.5)
#             ax.scatter(*self.neighbors[0], c='r', s=s, marker=(5, 1))
#         ax.set_xticks(np.arange(self.num_components))
#         ax.set_yticks(np.arange(self.num_components))

#     def plot_pdfs(self,
#               ax=None,
#               density=1000,
#               alpha=0.5,
#               ymax=None,
#               figsize=(3, 2)):
#         """
#         Plot density functions.
#         """

#         # create axes
#         if ax is None:
#             fig, ax = plt.subplots(figsize=figsize)

#         # plot individual component pdfs
#         pdfs = self.densities * self.weights.reshape(-1, 1)
#         for i, pdf in enumerate(pdfs):
#             if i in self.neighbors[0]:
#                 color = 'r'
#             else:
#                 color = 'k'
#             ax.fill_between(self.support, pdf, facecolors=color, alpha=alpha)

#         # format axis
#         if ymax is None:
#             model_pdf = pdfs.sum(axis=0)
#             maxima = model_pdf[argrelextrema(model_pdf, np.greater)]
#             ymax = 1.5*np.product(maxima)**(1/maxima.size)
#         ax.set_ylim(0, ymax)

#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)
#         ax.set_xlabel('Values', fontsize=8)
#         ax.set_ylabel('Density', fontsize=8)

#     def merge(self):
#         """ Combine two most similar components. """

#         # determine which components are merged
#         merged_components = self.neighbors[0]
#         merged = np.isin(np.arange(self.num_components), merged_components)

#         # evaluate weighted average of merged components
#         weights = self.weights[merged]
#         merged_weight = weights.sum().reshape(-1, 1)
#         merged_component = (self.densities[merged] * weights/merged_weight).sum(axis=0)

#         # append new components to non-merged components
#         merged_densities = np.vstack((self.densities[~merged], merged_component))
#         merged_weights = np.vstack((self.weights[~merged], merged_weight))

#         # define mapping of components before and after current aggregation
#         component_map = dict(enumerate(np.arange(self.num_components)[~merged]))
#         component_map = {v: k for k, v in component_map.items()}
#         component_map.update({component: self.num_components-2 for component in merged_components})

#         # combine with existing component mapping
#         self.key = {k: component_map[v] for k, v in self.key.items()}

#         return self.__class__(self.support, merged_densities, merged_weights, key=self.key)


# class Collapse:
#     """
#     Class for greedily collapsing the number of components in a mixture model.
#     """

#     def __init__(self, model, num_components=3, support_size=1000):
#         self.model = model
#         self.num_components = num_components
#         self.support = np.linspace(model.support.min(), model.support.max(), support_size)
#         self.initialize()

#     def initialize(self):
#         """ Initialize collapse object with top level merger. """
#         pdfs = self.model.evaluate_distribution_pdfs(self.support)
#         self.history = [Merge(self.support, pdfs, self.model.weights)]

#     def collapse(self, save=False):
#         """ Iteratively merge two most similar components until the target number of components is reached. """
#         merger = self.history[0]
#         while merger.num_components > self.num_components:
#             merger = merger.merge()
#             if save:
#                 self.history.append(merger)
#         return merger.key

#     def plot_distance_matrices(self, size=2, hspace=1.):
#         """ Plot distance matrices """
#         ncols = len(self.history)
#         gs = GridSpec(nrows=1, ncols=ncols, hspace=hspace)
#         fig = plt.figure(figsize=(ncols*size+(hspace*ncols-1), size))
#         for i, merger in enumerate(self.history):
#             ax = fig.add_subplot(gs[i])
#             merger.plot_distance_matrix(ax=ax)

#             # highlight ticklabels for merged axes
#             for k, ticklabels in enumerate([ax.xaxis.get_ticklabels(), ax.yaxis.get_ticklabels()]):
#                 for j, label in enumerate(ticklabels):
#                     if j  == merger.neighbors[0, k]:
#                         label.set_color('r')
#                         label.set_weight('bold')
#                         label.set_fontsize(12)

#         return fig
