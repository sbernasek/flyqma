from ..annotation import WeightedGraph
from ..annotation import Annotation


class Training:
    """ Methods for graph construction and training an Annotation object. """

    @staticmethod
    def build_graph(measurements,
                    weighted_by='clonal_marker',
                    logratio=True):
        """
        Returns WeightedGraph object.

        Args:

            measurements (pd.DataFrame) - measurement data

            weighted_by (str) - attribute used to weight graph edges

            logratio (bool) - if True, weight by log-ratio of attribute level

        Returns:

            graph (spatial.WeightedGraph)

        """
        return WeightedGraph(measurements,
                             xykey=['x', 'y'],
                             weighted_by=weighted_by,
                             logratio=logratio)

    @staticmethod
    def train(*graphs, attribute='clonal_marker', **kwargs):
        """
        Train an Annotation model on the measurements in this layer then return the optimal model.

        Args:

            graphs (collection of spatial.WeightedGraph) - training data

            attribute (str) - measured attribute used to determine labels

            kwargs: keyword arguments for Annotation, including:

                sampler_type (str) - either 'radial', 'neighbors', 'community'

                sampler_kwargs (dict) - keyword arguments for sampler

                min_num_components (int) - minimum number of mixture components

                max_num_components (int) - maximum number of mixture components

                addtl_kwargs: keyword arguments for Classifier

        Returns:

            annotator (Annotator object) - annotator fit to training data

        """
        annotator = Annotation(attribute, **kwargs)
        selector = annotator.train(*graphs)
        return annotator
