import numpy as np
import matplotlib.pyplot as plt
from pomegranate import GeneralMixtureModel, LogNormalDistribution
from .classifiers import Classifier



class BayesianClassifier(Classifier):
    """
    Bayesian mixed log-normal model classifier.

    Attributes:

        model (pomegranate.GeneralMixtureModel) - frozen mixture model

        weights (np.ndarray[float]) - 1D array of component weights

    Inherited attributes:

        values (array like) - basis for clustering

        log (bool) - indicates whether clustering performed on log values

        n (int) - number of clusters

        classifier (vectorized func) - maps value to group_id

        labels (np.ndarray[int]) - predicted labels

        cmap (matplotlib.colors.ColorMap) - colormap for group_id

        parameters (dict) - {param name: param value} pairs

        fig (matplotlib.figures.Figure) - histogram figure

    """

    def __init__(self, values, classify_on='r_normalized', **kwargs):
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

        # fit model
        self.model = self._fit(self.values, self.n)
        self.weights = np.exp(self.model.weights)

        # build classifier and posterior
        self.classifier = self.build_classifier()
        self.posterior = self.build_posterior()

        # assign group labels
        self.labels = self.classifier(self.values)

        # store parameters
        self.classify_on = classify_on
        self.parameters['classify_on'] = classify_on

    def __call__(self, df):
        """ Assign class labels to measurements <df>. """
        return self.evaluate_classifier(df)

    def evaluate_classifier(self, df):
        """ Assign class labels to measurements <df>. """
        x =  df[self.classify_on].values.reshape(-1, 1)
        if self.log:
            x = np.log10(x)
        return self.classifier(x)

    def evaluate_posterior(self, df):
        """ Evaluate normalized posterior probability for samples in <df>. """
        x =  df[self.classify_on].values.reshape(-1, 1)
        if self.log:
            x = np.log10(x)
        return self.posterior(x)

    @property
    def order(self):
        """ Ordered distribution indices (low to high). """
        dist_to_gen = self.distribution_to_genotype
        return sorted(dist_to_gen, key=dist_to_gen.__getitem__)

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
        get_size = lambda distribution: len(distribution.parameters)
        num_parameters = sum([get_size(x) for x in self.model.distributions])
        num_weights = self.model.n - 1
        return num_parameters + num_weights

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

    @property
    def distribution_to_genotype(self):
        """ Returns dictionary mapping distributions to genotypes.  """
        flip = lambda f: f.__class__(map(reversed, f.items()))
        means = [dist.parameters[0] for dist in self.model.distributions]
        genotype_to_distribution = dict(enumerate(np.argsort(means)))
        return flip(genotype_to_distribution)

    def build_classifier(self):
        """ Build cell genotype classifier. """

        # build classifier that maps model distributions to genotypes.
        dist_to_genotype = np.vectorize(self.distribution_to_genotype.get)

        def classifier(values):
            """ Returns <genotypes> for <values>.  """
            return dist_to_genotype(self.model.predict(values.reshape(-1, 1)))

        return classifier

    def build_posterior(self):
        """ Build posterior probability function. """

        # get distribution order
        order = np.array(self.order)

        def posterior(values):
            """ Returns probabilities of each label for <values>.  """
            return self.model.predict_proba(values.reshape(-1, 1))[:, order]

        return posterior

    def show_pdf(self, ax=None, density=1000, alpha=0.5):
        """
        Show density function.
        """

        # build support
        vmin, vmax = max(self._values.min(), 0.01), self._values.max()
        support = np.linspace(vmin, vmax, num=density)

        # create axes
        if ax is None:
            fig, ax = plt.subplots(figsize=(3, 2))

        # define map
        genotype_dict = self.distribution_to_genotype

        # plot individual components
        for i, dist in enumerate(self.model.distributions):
            pdf = dist.probability(support)
            weight = self.weights[i]
            #ax.plot(support, weight*pdf, c=self.cmap(genotype_dict[i]))
            ax.fill_between(support, weight*pdf, facecolors=self.cmap(genotype_dict[i]), alpha=alpha)

        # plot pdf
        pdf = self.model.probability(support)
        ax.plot(support, pdf, '-', c='k', lw=2)

        ax.set_ylim(0, 4*pdf.mean())

        # format axis
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('Values', fontsize=8)
        ax.set_ylabel('Density', fontsize=8)
