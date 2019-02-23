from os.path import join, exists
from os import mkdir
from copy import deepcopy
import numpy as np
import networkx as nx
from collections import Counter

from ..utilities import IO

from .spatial.sampling import NeighborSampler, CommunitySampler, RadialSampler
from .classification.mixtures import BivariateMixtureClassifier
from .model_selection import BivariateModelSelection


class AnnotationIO:
    """
    Methods for saving and loading an Annotation instance.
    """

    @property
    def parameters(self):
        """ Dictionary of parameter values. """
        return {k:v for k,v in self.__dict__.items() if k != 'classifier'}

    def save(self, dirpath, data=False, image=False, **kwargs):
        """
        Save annotator to specified path.

        Args:

            dirpath (str) - directory in which annotator is to be saved

            data (bool) - if True, save training data

            image (bool) - if True, save classifier image

            kwargs: keyword arguments for image rendering

        """

        # save parameters
        io = IO()
        io.write_json(join(dirpath, 'annotation.json'), self.parameters)

        # save classifier
        if self.classifier is not None:
            self.classifier.save(dirpath, data=data, image=image, **kwargs)

    @classmethod
    def load(cls, path):
        """
        Load annotator from file.

        Args:

            path (str) - path to annotation directory

        Returns:

            annotator (Annotation derivative)

        """

        io = IO()

        # load parameters
        parameters = io.read_json(join(path, 'annotation.json'))
        attribute = parameters.pop('attribute')

        # instantiate annotator
        annotator = cls(attribute, **parameters)

        # load classifier
        classifier_path = join(path, 'classifier')
        if exists(classifier_path):
            annotator.classifier = BivariateMixtureClassifier.load(classifier_path)

        return annotator


class Annotation(AnnotationIO):
    """
    Object for assigning labels to measurements. Object is trained on one or more graphs by fitting a bivariate mixture model and using a model selection procedure to select an optimal number of components.

    The trained model may then be used to label measurements in other graphs, either through direct prediction via the bivariate mixture model or through a hybrid prediction combining the bivariate mixture model with a marginal univariate model.

    Attributes:

        classifier (Classifier derivative) - callable object

        attribute (str) - attribute used to determine labels

    Parameters:

        sampler_type (str) - either 'radial', 'neighbors', 'community'

        sampler_kwargs (dict) - keyword arguments for sampler

        min_num_components (int) - minimum number of mixture components

        max_num_components (int) - maximum number of mixture components

        kwargs: keyword arguments for Classifier

    """

    def __init__(self, attribute,
                 sampler_type='radial',
                 sampler_kwargs={},
                 min_num_components=3,
                 max_num_components=10):
        """
        Instantiate annotation object.

        Args:

            attribute (str) - name of attribute used to classify cells

            sampler_type (str) - either 'radial', 'neighbors', 'community'

            sampler_kwargs (dict) - keyword arguments for sampler

            min_num_components (int) - minimum number of mixture components

            max_num_components (int) - maximum number of mixture components

        """
        self.attribute = attribute
        self.sampler_type = sampler_type
        self.sampler_kwargs = sampler_kwargs
        self.min_num_components = min_num_components
        self.max_num_components = max_num_components

    def __call__(self, graph, **kwargs):
        """ Returns labels for a graph of measurements. """
        return self.annotate(graph, **kwargs)

    @classmethod
    def from_data(cls, data, attribute, xykey=None, **kwargs):
        """
        Instantiate annotation object from measurement data.

        Args:

            data (pd.DataFrame) - measurement data containing <attribute>, as well as <xykey> fields

            attribute (str) - name of attribute used to classify cells

            xykey (list) - name of attributes defining measurement x/y position

            kwargs: keyword arguments for Annotation

        Returns:

            annotator (Annotation derivative)

        """
        annotator = cls(attribute, **kwargs)
        annotator.train(WeightedGraph(data, attribute, xykey=xykey))
        return annotator

    @classmethod
    def from_layer(cls, layer, attribute, **kwargs):
        """
        Instantiate from layer.

        Args:

            layer (data.Layer) - image layer instance

            attribute (str) - name of attribute used to classify cells

            kwargs: keyword arguments for Annotation

        Returns:

            annotator (Annotation derivative)

        """
        annotator = cls(attribute, **kwargs)
        annotator.train(layer.graph)
        return annotator

    @classmethod
    def copy(cls, src):
        """ Instantiate from another <source> annotator instance. """
        dst = cls(src.attribute)
        dst.__dict__.update(src.__dict__)
        return dst

    def train(self, *graphs):
        """
        Train classifier on a series of graphs.

        Args:

            graphs (Graph or WeightedGraph) - graphs of adjacent measurements

        """

        # generate sample
        if self.sampler_type == 'radial':
            data, keys = RadialSampler.multisample(self.attribute,
                                                *graphs,
                                                **self.sampler_kwargs)

        elif self.sampler_type == 'community':
            data, keys = CommunitySampler.multisample(self.attribute,
                                                *graphs,
                                                **self.sampler_kwargs)

        elif self.sampler_type == 'neighbors':
            data, keys = NeighborSampler.multisample(self.attribute,
                                               *graphs,
                                               **self.sampler_kwargs)

        else:
            raise ValueError('Sampler type ''{:s}'' not recognized.'.format(self.sampler_type))

        # run model selection
        selector = BivariateModelSelection(data,
            keys,
            min_num_components=self.min_num_components,
            max_num_components=self.max_num_components)

        # store BIC-optimal model
        self.classifier = selector.BIC_optimal

        return selector

    def get_sampler(self, graph, sampler_type=None, sampler_kwargs=None):
        """
        Instantiate sampler.

        Args:

            graph (spatial.WeightedGraph)

            sampler_type (str) - either 'radial', 'neighbors' or 'community'

            sampler_kwargs (dict) - keyword arguments for sampling

        Returns:

            sampler

        """

        # use default sampler configuration if none is specified
        if sampler_type is None:
            sampler_type = self.sampler_type

        if sampler_kwargs is None:
            sampler_kwargs = self.sampler_kwargs

        # instantiate sampler
        if sampler_type == 'radial':
            sampler = RadialSampler(graph, self.attribute, **sampler_kwargs)
        elif sampler_type == 'community':
            sampler = CommunitySampler(graph, self.attribute, **sampler_kwargs)
        elif sampler_type == 'neighbors':
            sampler = NeighborSampler(graph, self.attribute, **sampler_kwargs)
        else:
            raise ValueError('Sampler type ''{:s}'' not recognized.'.format(sampler_type))

        return sampler

    def get_sample(self, graph, sampler_type, sampler_kwargs):
        """
        Get sample to be annotated. A sample consists of a columns of measured levels adjoined to a column of levels averaged over the neighborhood of each measurement.

        Args:

            graph (spatial.WeightedGraph)

            sampler_type (str) - either 'radial', 'neighbors' or 'community'

            sampler_kwargs (dict) - keyword arguments for sampling

        Returns:

            sample (np.ndarray[float]) - sampled levels

        """

        # instantiate sampler
        sampler = self.get_sampler(graph, sampler_type, sampler_kwargs)

        # generate sample
        sample = sampler.sample

        # log transform sample
        if self.classifier.log:
            sample = np.log(sample)

        return sample

    def combine_posteriors(self, posterior, marginal_posterior, threshold=0.8):
        """
        Replace uncertain posterior probablilities with their more certain marginal counterparts. If the maximum marginal posterior probability for a given sample does not meet the specified threshold while the maximum bivarite posterior probability does, the latter value is used. Otherwise, the marginal value is used.

        Args:

            posterior (np.ndarray[float]) - posterior probabilities of each label

            marginal_posterior (np.ndarray[float]) - marginal posterior probabilities of each label

            threshold (float) - minimum marginal posterior probability of a given label before spatial context is considered

        Returns:

            combined (np.ndarray[float])

        """
        combined = deepcopy(marginal_posterior)
        mask = np.logical_and(posterior.max(axis=1) > threshold,
                              marginal_posterior.max(axis=1) < threshold)
        combined[mask, :] = posterior[mask, :]
        return combined

    def evaluate_marginal_posterior(self, sample, margin):
        """
        Evaluates posterior probability of each label using only the specified marginal distribution.

        Args:

            sample (np.ndarray[float]) - sample values

            margin (int) - index of desired margin

        Returns:

            marginal_posterior (np.ndarray[float])

        """
        return self.classifier[margin].posterior(sample[:, [margin]])

    @staticmethod
    def diffuse_posteriors(graph, posterior, alpha=0.9):
        """
        Diffuse estimated posterior probabilities of each label along the weighted edges of the graph.

        Args:

            graph (Graph) - graph connecting adjacent measurements

            posterior (np.ndarray[float]) - posterior probabiltiy of each label

            alpha (float) - attenuation factor

        Returns:

            diffused_posteriors (np.ndarray[float])

        """

        # compile normalized adjancy matrix
        adjacency = graph.adjacency_positional
        adjacency /= adjacency.sum(axis=0)

        # evaluate centrality
        external = np.eye(*adjacency.shape) - (alpha * adjacency)
        internal = (1 - alpha) * posterior
        diffused_posteriors = np.linalg.solve(external, internal)

        return diffused_posteriors

    def annotate(self, graph,
                 bivariate_only=False,
                 threshold=0.8,
                 alpha=0.9,
                 sampler_type=None,
                 sampler_kwargs=None):
        """
        Annotate graph of measurements.

        Args:

            graph (spatial.WeightedGraph)

            bivariate_only (bool) - if True, only use posteriors evaluated using the bivariate mixture model. Otherwise, use the marginal univariate posterior by default, replacing uncertain values with their counterparts estimated by the bivariate model.

            threshold (float) - minimum marginal posterior probability of a given label before spatial context is considered

            alpha (float) - attenuation factor

            sampler_type (str) - either 'radial', 'neighbors' or 'community'

            sampler_kwargs (dict) - keyword arguments for sampling

        Returns:

            labels (np.ndarray[int]) - labels for each measurement in graph

        """

        # get sample data
        sample = self.get_sample(graph=graph,
                                 sampler_type=sampler_type,
                                 sampler_kwargs=sampler_kwargs)

        # classify sample
        posterior = self.classifier.posterior(sample)

        # combine with posteriors estimated by univariate marginal classifier
        if not bivariate_only:
            marginal = self.evaluate_marginal_posterior(sample, 0)
            posterior = self.combine_posteriors(posterior, marginal, threshold)

        # diffuse posteriors
        if alpha is not None:
            posterior = self.diffuse_posteriors(graph, posterior, alpha=alpha)

        # assign labels
        labels = posterior.argmax(axis=1)

        return labels
