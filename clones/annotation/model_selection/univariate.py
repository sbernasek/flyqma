import numpy as np

from ..classification import MixtureModelClassifier
from .visualization import ModelSelectionVisualization


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
