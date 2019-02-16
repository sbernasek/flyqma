import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import k_means

from ..mixtures import UnivariateMixture, BivariateMixture
from .classifiers import Classifier
from .visualization import MixtureVisualization, BivariateMixtureVisualization


class MixtureModelClassifier(Classifier, MixtureVisualization):
    """
    Univariate mixed log-normal model classifier.

    Attributes:

        model (mixtures.UnivariateMixture) - frozen univariate mixture model

        num_components (int) - number of mixture components

        classifier (vectorized func) - maps values to labels

        labels (np.ndarray[int]) - predicted labels

    Inherited attributes:

        values (np.ndarray[float]) - basis for clustering

        num_labels (int) - number of output labels

        log (bool) - indicates whether clustering performed on log values

        cmap (matplotlib.colors.ColorMap) - colormap for labels

        parameters (dict) - {param name: param value} pairs

    """

    def __init__(self, values,
                 num_components=3,
                 num_labels=3,
                 fit_kw={},
                 **kwargs):
        """
        Fit a univariate mixture model classifier to an array of values.

        Args:

            values (np.ndarray[float]) - basis for clustering (not log-transformed)

            num_components (int) - number of mixture components

            num_labels (int) - number of class labels

            fit_kw (dict) - keyword arguments for fitting mixture model

        Keyword arguments:

            classify_on (str or list) - attribute(s) on which to cluster

            cmap (matplotlib.colors.ColorMap) - colormap for class_id

        """

        # instantiate classifier
        super().__init__(values, num_labels=num_labels, log=True, **kwargs)
        self.parameters['num_components'] = num_components

        # fit model
        self.model = self.fit(self.values, num_components, **fit_kw)

        # build classifier and posterior
        self.classifier = self.build_classifier()
        self.posterior = self.build_posterior()

        # assign labels
        self.labels = self.classifier(self.values)

    def __call__(self, df):
        """ Assign class labels to measurements <df>. """
        return self.evaluate_classifier(df)

    @property
    def num_components(self):
        """ Number of model components. """
        return self.model.n_components

    @property
    def means(self):
        """ Mean of each component. """
        return self.model.means

    @staticmethod
    def fit(values, num_components=3, **kwargs):
        """
        Fit univariate gaussian mixture model.

        Args:

            values (np.ndarray[float]) - 1D array of log-transformed values

            num_components (int) - number of model components

            kwargs: keyword arguments for fitting

        Returns:

            model (mixtures.UnivariateMixture)

        """
        return UnivariateMixture.from_logsample(values,
                                                num_components,
                                                **kwargs)

    def predict(self, values):
        """ Predict which component each of <values> belongs to. """
        return self.model.predict(values)

    def predict_proba(self, values):
        """
        Predict the posterior probability with which each of <values> belongs to each component.
        """
        return self.model.predict_proba(values)

    def build_posterior(self):
        """
        Build function that returns the posterior probability of each label given a series of values.
        """

        def posterior(values):
            """ Returns probabilities of each label for <values>.  """

            # evaluate posterior probability of each label for each value
            p = self.model.predict_proba(values)
            _posterior = [p[:,i].sum(axis=1) for i in self.component_groups]
            _posterior = np.vstack(_posterior).T

            # fix label probabilities for points outside the support bounds
            below = values < self.model.lbound
            above = values > self.model.ubound
            for rows, col in zip((below.ravel(), above.ravel()), [0, -1]):
                if rows.sum() == 0:
                    continue
                adjust = np.zeros((rows.sum(), self.num_labels), dtype=float)
                adjust[:, col] = 1.
                _posterior[rows] = adjust

            return _posterior

        return posterior

    def evaluate_classifier(self, df):
        """ Returns labels for measurements in <df>. """
        x =  df[self.classify_on].values
        if self.log:
            x = np.log(x)
        return self.classifier(x)

    def evaluate_posterior(self, df):
        """ Returns posterior across components for measurements in <df>. """
        x =  df[self.classify_on].values
        if self.log:
            x = np.log(x)
        return self.posterior(x)


class BivariateMixtureClassifier(BivariateMixtureVisualization,
                                 MixtureModelClassifier):
    """
    Bivariate mixed log-normal model classifier.

    Attributes:

        model (mixtures.BivariateMixture) - frozen bivariate mixture model

    Inherited attributes:

        values (np.ndarray[float]) - basis for clustering

        classify_on (str or list) - attribute(s) on which to cluster

        num_labels (int) - number of output labels

        num_components (int) - number of mixture components

        classifier (vectorized func) - maps values to labels

        labels (np.ndarray[int]) - predicted labels

        log (bool) - indicates whether clustering performed on log values

        cmap (matplotlib.colors.ColorMap) - colormap for labels

        parameters (dict) - {param name: param value} pairs

    """

    @staticmethod
    def fit(values, num_components=3, **kwargs):
        """
        Fit univariate gaussian mixture model.

        Args:

            values (np.ndarray[float]) - 1D array of log-transformed values

            num_components (int) - number of model components

            kwargs: keyword arguments for fitting

        Returns:

            model (mixtures.BivariateMixture)

        """
        return BivariateMixture.from_logsample(values, num_components, **kwargs)
