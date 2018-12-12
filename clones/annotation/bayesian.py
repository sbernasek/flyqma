import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.signal import argrelextrema
from sklearn.cluster import k_means
from pomegranate import GeneralMixtureModel, LogNormalDistribution
from pomegranate import ExponentialDistribution
from .classifiers import Classifier


class BayesianProperties:

    @property
    def components(self):
        """ Model components. """
        return self.model.distributions

    @property
    def means(self):
        """ Mean of each component. """
        return np.trapz(self.evaluate_component_pdfs()*self.support, x=self.support)

    @property
    def weights(self):
        """ Normalized weight of each component. """
        return np.exp(self.model.weights)

    @property
    def num_samples(self):
        """ Number of samples. """
        return self.values.size

    @property
    def log_likelihood(self):
        """ Maximum log likelihood of the data given the fitted model. """
        return self.model.log_probability(self.values).mean()*self.num_samples

    @property
    def BIC(self):
        """ Bayesian information criterion for the fitted mixture model. """
        return (-2 * self.log_likelihood)+(self.num_parameters*np.log(self.num_samples))

    @property
    def AIC(self):
        """ Akaike information criterion for the fitted mixture model. """
        return (-2 * self.log_likelihood) + (2*self.num_parameters)

    @property
    def num_parameters(self):
        """ Number of model parameters. """
        num_parameters = sum([len(x.parameters) for x in self.components])
        num_weights = self.model.n - 1
        return num_parameters + num_weights


class BayesianVisualization:

    def plot_pdf(self,
                  ax=None,
                  density=1000,
                  alpha=0.5,
                  xmax=None,
                  ymax=None,
                  figsize=(3, 2)):
        """
        Plot model density function, colored by output label.
        """

        # create axes
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        # plot model pdf segments, colored by output label
        support_labels = self.classifier(self.support)
        model_pdf = self.model_pdf
        breakpoints = [0]+list(np.diff(support_labels).nonzero()[0]+1)+[None]
        for i, bp in enumerate(breakpoints[:-1]):
            indices = slice(bp, breakpoints[i+1])
            segment_support = self.support[indices]
            segment_pdf = model_pdf[indices]
            segment_labels = support_labels[indices]
            segment_color = self.cmap(segment_labels)
            ax.fill_between(segment_support, segment_pdf, color=segment_color)

        # plot overall model pdf
        ax.plot(self.support, model_pdf, '-', c='k', lw=2)

        # format axis
        if ymax is None:
            maxima = model_pdf[argrelextrema(model_pdf, np.greater)]
            ymax = 2.5*np.product(maxima)**(1/maxima.size)
        if xmax is None:
            xmax = np.percentile(self.support, 99)

        ax.set_xlim(0, xmax)
        ax.set_ylim(0, ymax)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('Values', fontsize=8)
        ax.set_ylabel('Density', fontsize=8)

    def plot_pdfs(self,
                  ax=None,
                  density=1000,
                  alpha=0.5,
                  xmax=None,
                  ymax=None,
                  figsize=(3, 2)):
        """
        Plot density function for each distribution, colored by output label.
        """

        # create axes
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        # plot empirical pdf
        ax.step(*self.pdf, where='post', color='r', linewidth=2)

        # plot individual component pdfs
        pdfs = self.evaluate_component_pdfs() * self.weights.reshape(-1, 1)
        for i, pdf in enumerate(pdfs):
            color = self.cmap(self.component_to_label[i])
            ax.fill_between(self.support, pdf, facecolors=color, alpha=alpha)

        # plot model pdf
        model_pdf = self.model_pdf
        ax.plot(self.support, model_pdf, '-', c='k', lw=2)

        # format axis
        if ymax is None:
            maxima = model_pdf[argrelextrema(model_pdf, np.greater)]
            ymax = 2.5*np.product(maxima)**(1/maxima.size)
        if xmax is None:
            xmax = np.percentile(self.support, 99)

        ax.set_ylim(0, ymax)
        ax.set_xlim(0, xmax)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('Values', fontsize=8)
        ax.set_ylabel('Density', fontsize=8)

    def plot_cdfs(self, figsize=(3, 2)):
        """ Plot cumulative distribution functions. """

        fig, ax = plt.subplots(figsize=figsize)

        # plot empirical cdf (data)
        ax.plot(self.support, self.cdf, '-r', lw=2)

        # plot component cdfs (components)
        for cdf in self.component_cdfs:
            ax.plot(self.support, cdf, '-k')

        # plot weighted component cdf (model fit)
        cdf = (self.component_cdfs*self.weights.reshape(-1, 1)).sum(axis=0)
        ax.plot(self.support, cdf, '-b', lw=2)


class BayesianClassifier(Classifier, BayesianProperties, BayesianVisualization):
    """
    Bayesian mixed log-normal model classifier.

    Attributes:

        model (pomegranate.GeneralMixtureModel) - frozen mixture model

        weights (np.ndarray[float]) - 1D array of component weights

    Inherited attributes:

        values (array like) - basis for clustering

        log (bool) - indicates whether clustering performed on log values

        n (int) - number of clusters

        num_labels (int) - number of output labels

        classifier (vectorized func) - maps value to group_id

        labels (np.ndarray[int]) - predicted labels

        cmap (matplotlib.colors.ColorMap) - colormap for group_id

        parameters (dict) - {param name: param value} pairs

        fig (matplotlib.figures.Figure) - histogram figure

    """

    def __init__(self, values, classify_on='r', **kwargs):
        """
        Fit a cell classifier to an array of values.

        Args:

            values (np.ndarray[float]) - 1-D vector of measured values

            classify_on (str) - measurement attribute from which values came

        Keyword arguments:

            n (int) - number of clusters

            log (bool) - indicates whether clustering performed on log values

            cmap (matplotlib.colors.ColorMap) - colormap for class_id

        Returns:

            classifier (CellClassifier)

        """

        # instantiate classifier
        super().__init__(values, **kwargs)
        self.support = np.sort(values)

        # fit model
        self.model = self._fit(self.values, self.n)

        # build classifier and posterior
        self.classifier = self.build_classifier()
        self.posterior = self.build_posterior()

        # assign labels
        self.labels = self.classifier(self.values)

        # store parameters
        self.classify_on = classify_on
        self.parameters['classify_on'] = classify_on

    def __call__(self, df):
        """ Assign class labels to measurements <df>. """
        return self.evaluate_classifier(df)

    @staticmethod
    def _fit(values, n=3):
        """
        Fit log-normal mixture model using likelihood maximization.

        Args:

            values (np.ndarray[float]) - 1D array of values

            n (int) - number of log-normal distributions

        Returns:

            model (pomegranate.GeneralMixtureModel)

        """
        x = values.reshape(-1, 1)
        args = (LogNormalDistribution, n, x)
        kwargs = dict(n_init=1000)
        return GeneralMixtureModel.from_samples(*args, **kwargs)


    def evaluate_classifier(self, df):
        """ Returns labels for measurements in <df>. """
        x =  df[self.classify_on].values.reshape(-1, 1)
        if self.log:
            x = np.log10(x)
        return self.classifier(x)

    def evaluate_posterior(self, df):
        """ Returns posterior across components for measurements in <df>. """
        x =  df[self.classify_on].values.reshape(-1, 1)
        if self.log:
            x = np.log10(x)
        return self.posterior(x)

    @property
    def order(self):
        """ Ordered component indices (low to high). """
        x = self.component_to_label
        return sorted(x, key=x.__getitem__)

    @property
    def component_groups(self):
        """ List of lists of components for each label. """
        x = self.component_to_label
        labels = np.unique(list(x.values()))
        return [[k for k, v in x.items() if v == l] for l in labels]

    @property
    def component_to_label(self):
        """ Returns dictionary mapping components to labels.  """
        n = self.num_labels
        cluster_means, cluster_labels, _ = k_means(self.means.reshape(-1,1), n)
        component_to_label = {}
        for label, c in enumerate(np.argsort(cluster_means.ravel())):
            for d in (cluster_labels==c).nonzero()[0]:
                component_to_label[d] = label
        return component_to_label

    def build_classifier(self):
        """ Build cell genotype classifier. """

        # build classifier that maps model distributions to genotypes.
        component_to_label = np.vectorize(self.component_to_label.get)

        def classifier(values):
            """ Returns <label> for <values>.  """
            return component_to_label(self.model.predict(values.reshape(-1,1)))

        return classifier

    def build_posterior(self):
        """ Build posterior probability function. """

        def posterior(values):
            """ Returns probabilities of each label for <values>.  """
            p = self.model.predict_proba(values.reshape(-1, 1))
            label_p = [p[:,i].sum(axis=1) for i in self.component_groups]
            return np.vstack(label_p).T

        return posterior

    @staticmethod
    def evaluate_pdf(distribution, support):
        """ Returns PDF of a continuous distribution. """
        return distribution.probability(support)

    @property
    def model_pdf(self):
        """ Returns model PDF over support. """
        return self.model.probability(self.support)

    @property
    def pdf(self):
        """ Returns empirical PDF over support. """
        num_bins = self.num_samples // 50
        bins = np.linspace(self.support.min(), self.support.max(), num_bins)
        counts, edges = np.histogram(self.support, bins=bins, normed=True)
        bin_centers = [(edges[i]+edges[i+1])/2. for i in range(len(edges)-1)]
        return edges[:-1], counts

    def evaluate_component_pdfs(self, support=None):
        """ Returns PDF of each component over <support>. """
        if support is None:
            support = self.support
        pdf = [self.evaluate_pdf(d, support) for d in self.components]
        pdf = np.vstack(pdf)
        return pdf

    @staticmethod
    def evaluate_cdf(distribution, support):
        """ Returns CDF of a discrete distribution. """
        density = distribution.probability(support)
        return np.cumsum(density) / density.sum()

    @property
    def cdf(self):
        """ Returns CDF over support. """
        return np.linspace(0, 1, self.values.size, endpoint=False)

    @property
    def component_cdfs(self):
        """ Returns CDF of each component over support. """
        cdf = [self.evaluate_cdf(d, self.support) for d in self.components]
        return np.vstack(cdf)


class MixedClassifier(BayesianClassifier):

    @staticmethod
    def _fit(values, n=3):
        """
        Fit exponential+log-normal mixture model using likelihood maximization.

        Args:

            values (np.ndarray[float]) - 1D array of values

            n (int) - number of log-normal distributions

        Returns:

            model (pomegranate.GeneralMixtureModel)

        """
        x = values.reshape(-1, 1)
        distributions = [ExponentialDistribution] + [LogNormalDistribution]*(n-1)
        args = (distributions, n, x)
        kwargs = dict(n_init=1000)
        return GeneralMixtureModel.from_samples(*args, **kwargs)
