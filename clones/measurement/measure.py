from warnings import filterwarnings, catch_warnings
import numpy as np
import pandas as pd
from scipy.ndimage.measurements import mean, standard_deviation, center_of_mass


class Measurements:
    """
    Object measures properties of labeled segments within an image.

    Attributes:

        colordepth (int) - number of color channels

        segment_ids (np.ndarray[float]) - ordered segment labels

        levels (dict) - {channel: np.ndarray[float]} - expression levels

        std (dict) - {channel: np.ndarray[float]} - expression std. deviation

        xpos (np.ndarray[float]) - segment centroid x-positions

        ypos (np.ndarray[float]) - segment centroid y-positions

        voxel_size (np.ndarray[float]) - segment voxel size

    """

    def __init__(self, im, labels):
        """
        Measure properties of labeled segments within an image.

        Args:

            im (np.ndarray[float]) - 3D array of pixel values

            labels (np.ndarray[int]) - cell segment labels

        """

        self.colordepth = im.shape[-1]

        # set segment ids (ordered)
        self.segment_ids = np.unique(labels[labels.nonzero()])

        # measure expression levels
        self.measure_expression(im, labels, self.segment_ids)

        # measure segment centroids
        self.measure_centroids(labels, self.segment_ids)

        # measure segment voxel sizes
        self.measure_segment_size(labels, self.segment_ids)

    def measure_expression(self, im, labels, segment_ids):
        """
        Measure expression levels.

        Args:

            im (np.ndarray[float]) - 3D array of pixel values

            labels (np.ndarray[int]) - cell segment labels

            segment_ids (np.ndarray[int]) - ordered segment IDs

        """

        # split R/G/B image channels
        drop = lambda x: x.reshape(*x.shape[:2])
        channels = [drop(x) for x in np.split(im, self.colordepth, axis=-1)]

        # compute means
        means = [mean(channel, labels, segment_ids) for channel in channels]

        # compute std
        with catch_warnings():
            filterwarnings('ignore')
            evaluate_std = lambda x: standard_deviation(x, labels, segment_ids)
            stds = [evaluate_std(channel) for channel in channels]

        # compile dictionaries
        self.levels = dict(enumerate(means))
        self.std = dict(enumerate(stds))

    def measure_centroids(self, labels, segment_ids):
        """
        Measure the centroid of each segment.

        Args:

            labels (np.ndarray[int]) - cell segment labels

            segment_ids (np.ndarray[int]) - ordered segment IDs

        """
        centroid_dict = self.evaluate_centroids(labels)
        centroids = [centroid_dict[seg_id] for seg_id in segment_ids]
        xpos, ypos = list(zip(*centroids))
        self.xpos = xpos
        self.ypos = ypos

    @staticmethod
    def evaluate_centroids(labels):
        """
        Evaluate center of mass of each label.

        * Note: scipy returns centroids as (y, x) which are flipped to (x, y)

        Args:

            labels (np.ndarray[int]) - segment label mask

        Returns:

            center_of_mass (dict) - {segment_id: [xpos, ypos]} pairs

        """

        seg_ids = np.unique(labels[labels!=0])
        coms = center_of_mass(labels, labels, seg_ids)
        return {seg_id: com[::-1] for seg_id, com in zip(seg_ids, coms)}

    def measure_segment_size(self, labels, segment_ids):
        """
        Measure the voxel size of each segment.

        Args:
        labels (np.ndarray[int]) - cell segment labels
        segment_ids (np.ndarray[int]) - ordered segment IDs
        """
        voxels = labels[labels!=0]
        bins = np.arange(0, segment_ids.max()+3, 1)
        counts, _ = np.histogram(voxels, bins=bins)
        self.voxel_counts = counts[segment_ids]

    def build_dataframe(self):
        """
        Build and return dataframe containing all measurements.

        Returns:

            measurements (pd.DataFrame) - measurement data

        """

        # construct dataframe
        measurement_data = dict(
            segment_id=self.segment_ids,
            centroid_x=self.xpos,
            centroid_y=self.ypos,
            pixel_count=self.voxel_counts)

        # add intensity measurements
        for channel_id in range(self.colordepth):

            # define keys
            key = 'ch{:d}'.format(channel_id)
            key_std = key + '_std'

            # store measured levels
            measurement_data[key] = self.levels[channel_id]
            measurement_data[key_std] = self.std[channel_id]

        return pd.DataFrame(measurement_data)
