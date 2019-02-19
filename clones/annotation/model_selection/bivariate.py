
from ..classification import BivariateMixtureClassifier
from .univariate import UnivariateModelSelection


class BivariateModelSelection(UnivariateModelSelection):
    """ Bivariate extension for model selection. """

    @staticmethod
    def fit_model(values, num_components, **kwargs):
        """ Fit model with specified number of components. """
        return BivariateMixtureClassifier(values,
                                        num_components=num_components,
                                        **kwargs)
