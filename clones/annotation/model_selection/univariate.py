from os.path import join, exists
from os import mkdir
import numpy as np

from ...utilities import IO

from ..classification import UnivariateMixtureClassifier
from .visualization import ModelSelectionVisualization


class SelectionIO:
    """
    Methods for saving and loading a model selection instance.
    """
    def save(self, dirpath, image=False, **kwargs):
        """
        Save classifier to specified path.

        Args:

            dirpath (str) - directory in which classifier is to be saved

            image (bool) - if True, save model image

            kwargs: keyword arguments for image rendering

        Returns:

            path (str) - model selection directory

        """

        # create directory for model selection
        path = join(dirpath, 'models')
        if not exists(path):
            mkdir(path)

        # save parameters
        io = IO()
        io.write_json(join(path, 'parameters.json'), self.parameters)

        # save values
        np.save(join(path, 'values.npy'), self.values)

        # save models
        for n, model in self._models.items():
            model.save(path, image=image, extension=n, **kwargs)

        return path

    @staticmethod
    def load_model(path):
        """ Load model from <path> directory. """
        return UnivariateMixtureClassifier.load(path)

    @classmethod
    def load(cls, path):
        """
        Load model selection instance from file.

        Args:

            path (str) - model selection directory

        Returns:

            selector (UnivariateModelSelection derivative)

        """
        io = IO()

        # load values and parameters
        values = io.read_npy(join(path, 'values.npy'))
        parameters = io.read_json(join(path, 'parameters.json'))
        attribute = parameters.pop('attribute')

        # load models
        n_min = parameters['min_num_components']
        n_max = parameters['max_num_components']

        models = {}
        for num_components in range(n_min, n_max+1):
            model_path = join(path, 'classifier_{:d}'.format(num_components))
            if exists(model_path):
                model = cls.load_model(model_path)
                model._values = values
                model.model.values = np.log(values)
                models[num_components] = model

        return cls(values, attribute, models=models, **parameters)


class UnivariateModelSelection(SelectionIO, ModelSelectionVisualization):
    """
    Class for performing univariate mixture model selection. The optimal model is chosen based on BIC score.
    """

    def __init__(self, values, attribute,
                 min_num_components=3,
                 max_num_components=8,
                 models=None):
        """
        Perform model selection by choosing the model that minimizes BIC score.

        Args:

            values (np.ndarray[float]) - 1D array of sample values

            attribute (str) - attribute label for sample values

            min_num_components (int) - minimum number of components in mixture

            max_num_components (int) - maximum number of components in mixture

            models (dict) - pre-fitted Classification instances keyed by number of components

        """

        self.values = values
        self.attribute = attribute
        self.min_num_components = min_num_components
        self.max_num_components = max_num_components
        self.num_components = range(min_num_components, max_num_components+1)

        # fit models
        if models is None:
            models = self.fit_models()
        self._models = models

    @staticmethod
    def fit_model(values, num_components, **kwargs):
        """ Fit model with specified number of components. """
        return UnivariateMixtureClassifier(values,
                                    num_components=num_components,
                                    num_labels=num_components,
                                    **kwargs)

    def fit_models(self):
        """ Fit model with each number of components. """

        # define parameters
        args = (self.values,)
        kwargs = dict(attribute=self.attribute)

        # fit models
        models_dict = {}
        for num_components in self.num_components:
            model = self.fit_model(self.values, num_components, **kwargs)
            models_dict[num_components] = model

        return models_dict

    @property
    def parameters(self):
        """ Dictionary of instance parameters. """
        return {
            'attribute': self.attribute,
            'min_num_components': self.min_num_components,
            'max_num_components': self.max_num_components}

    @property
    def models(self):
        """ List of models ordered by number of components. """
        return [m for n, m in sorted(self._models.items())]

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
