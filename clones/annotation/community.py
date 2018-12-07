import numpy as np
from collections import Counter


class CommunityClassifier:
    """
    Classifier for assigning labels to communities.

    Attributes:

        classifier (CellClassifier) - individual cell classifier

    """

    def __init__(self, cells, cell_classifier, rule='weighted'):
        """
        Instantiate community classifier.

        Args:

            cells (pd.DataFrame) - cell data including community labels

            cell_classifier (BayesianClassifier) - labels individual cells

            rule (str) - voting rule, e.g. 'weighted' or 'majority'

        """
        self.classifier = self.build_classifier(cells, cell_classifier, rule)

    def __call__(self, communities):
        """ Classify communities. """
        return self.classifier(communities)

    @staticmethod
    def from_layer(layer, cell_classifier):
        """ Instantiate community classifier from a layer. """
        return CommunityClassifier(layer.data, cell_classifier)

    @staticmethod
    def get_mode(x):
        """ Returns most common value in an array. """
        mode, count = Counter(x).most_common(1)[0]
        return mode

    @classmethod
    def build_voter(cls, cell_classifier, rule='proportional'):
        """
        Build voting function.

        Args:

            cell_classifier (BayesianClassifier) - labels individual cells

            rule (str) - voting rule, e.g. 'weighted' or 'majority'

        """

        # aggregate votes via majority rules
        if rule == 'majority':
            def voter(x):
                return cls.get_mode(cell_classifier(x))

        # aggregate maximum mean posterior (proportional representation)
        elif rule == 'proportional':
            def voter(x):
                posterior = cell_classifier.evaluate_posterior(x)
                return posterior.mean(axis=0).argmax()

        # aggregate votes weighted by posterior probability of each label
        elif rule == 'weighted':
            def voter(x):
                posterior = cell_classifier.evaluate_posterior(x)
                confidence = posterior.max(axis=1)
                genotypes = posterior.argmax(axis=1)
                ind = np.argsort(genotypes)
                starts = np.searchsorted(genotypes[ind], np.arange(4))
                lengths = np.diff(starts)
                return np.argmax([confidence[ind][slice(s, s+l)].sum() for s, l in zip(starts[:-1], lengths)])

        else:
            raise ValueError('Voter rule not recognized.')

        return voter

    @classmethod
    def build_classifier(cls, cells, cell_classifier, rule='weighted'):
        """
        Build classifier assigning genotypes to graph communities.

        Args:

            cells (pd.DataFrame) - cell data including community labels

            cell_classifier (BayesianClassifier) - labels individual cells

            rule (str) - voting rule, e.g. 'weighted' or 'majority'

        """
        #majority_vote = lambda x: cls.get_mode(cell_classifier(x))
        voter = cls.build_voter(cell_classifier, rule=rule)
        communities = cells.groupby('community')
        community_to_genotype = communities.apply(voter).to_dict()
        community_to_genotype[-1] = -1
        return np.vectorize(community_to_genotype.get)
