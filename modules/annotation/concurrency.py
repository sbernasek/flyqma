import numpy as np
from scipy.spatial.distance import cdist

from .labeler import AttributeLabeler


class ConcurrencyLabeler(AttributeLabeler):
    """
    Object for labeling cell measurements as concurrent with other cell types.  Concurrency with each unique target cell type is determined by applying a maximum threshold on the x-distance between each cell and its nearest neighbor of the target type.

    Attributes:
    attribute (str) - measurement attribute used to denote cell type
    target_types (np.ndarray) - unique cell types labels
    min_pop (int) - minimum population size for inclusion of cell type
    max_distance (float) - maximum distance threshold for inclusion
    """

    def __init__(self, attribute='celltype', min_pop=5, max_distance=10):
        """
        Instantiate object for labeling measurements as concurrent with other cell types.

        Args:
        attribute (str) - measurement attribute used to denote cell type
        min_pop (int) - minimum population size for inclusion of cell type
        max_distance (float) - maximum distance threshold for inclusion
        """

        self.attribute = attribute
        self.target_types = self.cells[self.attribute].unique()
        self.min_pop = min_pop
        self.max_distance = max_distance

    def evaluate_distance(self, df, target_type):
        """
        Evaluate x-distance of all cells to nearest neighbor of target cell type.

        Args:
        df (pd.DataFrame) - cell measurement data
        target_type (str) - target cell type

        Returns:
        distances (np.ndarray[float]) - distances to target cell type
        """

        # get target cells
        targets = df[df[self.attribute]==target_type]

        # evaluate distances if there are enough target cells
        if len(targets) > self.min_pop:
            rs = lambda x: x.centroid_x.values.reshape(-1, 1)
            distances = cdist(rs(df), rs(targets)).min(axis=1)

        # otherwise set arbitrarily high distance
        else:
            distances = 10000*np.ones(len(df), dtype=np.float64)

        return distances

    def assign_labels(self, df):
        """
        Add boolean 'concurrent_<cell type>' field to cell measurement data for each unique cell type.

        Args:
        df (pd.DataFrame) - cell measurement data
        """
        for target_type in self.target_types:
            distances = self.evaluate_distance(df, target_type)
            df['concurrent_'+target_type] = (distances <= self.max_distance)
