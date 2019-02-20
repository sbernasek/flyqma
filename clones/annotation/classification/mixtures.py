from os.path import join, exists
from os import mkdir
import pickle
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt

from ...utilities.io import IO

from .classifiers import Classifier, ClassifierIO
from ..mixtures import UnivariateMixture, BivariateMixture
from .visualization import MixtureVisualization, BivariateMixtureVisualization


class MixtureModelIO(ClassifierIO):
    """
    Methods for saving and loading classifier objects.
    """

    def save(self, dirpath, image=True, **kwargs):
        """
        Save classifier to specified path.

        Args:

            dirpath (str) - directory in which classifier is to be saved

            image (bool) - if True, save labeled histogram image

            kwargs: keyword arguments for image rendering

        """
        path = super().save(dirpath, image, **kwargs)

        # save model
        if self.model is not None:
            with open(join(path, 'model.pkl'), 'wb') as file:
                pickle.dump(file, self.model)

        return path

    @classmethod
    def load(cls, path):
        """
        Load classifier from file.

        Args:

            path (str) - path to classifier directory

        Returns:

            classifier (Classifier derivative)

        """
        io = IO()
        values = io.read_npy(join(path, 'values.npy'))
        parameters = io.read_json(join(path, 'parameters.json'))

        # load model
        with open(join(path, 'model.pkl'), 'rb') as file:
            model = pickle.load(file)

        return cls(values, model=model, **parameters)


class MixtureModelClassifier(MixtureModelIO, Classifier, MixtureVisualization):
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
                 model=None,
                 **kwargs):
        """
        Fit a univariate mixture model classifier to an array of values.

        Args:

            values (np.ndarray[float]) - basis for clustering (not log-transformed)

            num_components (int) - number of mixture components

            num_labels (int) - number of class labels

            fit_kw (dict) - keyword arguments for fitting mixture model

            model (mixtures.UnivariateMixture) - pre-fitted model

        Keyword arguments:

            classify_on (str or list) - attribute(s) on which to cluster

            cmap (matplotlib.colors.ColorMap) - colormap for class_id

        """

        # instantiate classifier
        super().__init__(values, num_labels=num_labels, log=True, **kwargs)
        self.parameters['num_components'] = num_components
        self.parameters['fit_kw'] = fit_kw

        # fit model
        if model is None:
            model = self.fit(self.values, num_components, **fit_kw)
        self.model = model

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
            below = values[:, 0] < self.model.lbound
            above = values[:, 0] > self.model.ubound
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

    def build_classifier(self):
        """
        Build function that returns the most probable label for each of a series of values.
        """

        def classifier(values):
            """ Returns <label> for <values> by maximizing posterior.  """
            return self.posterior(values).argmax(axis=1)

        return classifier


class BivariateMixtureClassifier(BivariateMixtureVisualization,
                                 MixtureModelClassifier):
    """
    Bivariate mixed log-normal model classifier.

    Attributes:

        model (mixtures.BivariateMixture) - frozen bivariate mixture model

    Inherited attributes:

        values (np.ndarray[float]) - basis for clustering

        classify_on (list) - attributes on which to cluster

        num_labels (int) - number of labels

        num_components (int) - number of mixture components

        classifier (vectorized func) - maps values to labels

        labels (np.ndarray[int]) - predicted labels

        log (bool) - indicates whether clustering performed on log values

        cmap (matplotlib.colors.ColorMap) - colormap for labels

        parameters (dict) - {param name: param value} pairs

    """

    def __getitem__(self, margin):
        """ Returns MixtureModelClassifier for specified margin. """
        return self.marginalize(margin)

    def marginalize(self, margin):
        """ Returns MixtureModelClassifier for specified margin. """

        # assemble marginalized properties
        values = self._values[:, [margin]]
        model = self.model[margin]

        # duplicate parameters
        parameters = deepcopy(self.parameters)
        parameters['classify_on'] = self.classify_on[0]
        _ = parameters.pop('log')

        return MixtureModelClassifier(values, model=model, **parameters)

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
