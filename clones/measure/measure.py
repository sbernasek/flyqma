from warnings import filterwarnings, catch_warnings
import numpy as np
import pandas as pd
from scipy.ndimage.measurements import mean, standard_deviation, center_of_mass


class Measurements:
    """
    Object measures properties of labeled segments within an image.

    Attributes:
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
        im (np.ndarray[float]) - 2D array of RGB pixel values
        labels (np.ndarray[int]) - cell segment labels
        """

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
        im (np.ndarray[float]) - 2D array of RGB pixel values
        labels (np.ndarray[int]) - cell segment labels
        segment_ids (np.ndarray[int]) - ordered segment IDs
        """

        # split R/G/B image channels
        drop_axis = lambda x: x.reshape(*x.shape[:2])
        r, g, b = [drop_axis(x) for x in np.split(im, 3, axis=-1)]

        # compute means
        rmeans = mean(r, labels, segment_ids)
        gmeans = mean(g, labels, segment_ids)
        bmeans = mean(b, labels, segment_ids)

        # compute std
        with catch_warnings():
            filterwarnings('ignore')
            rstd = standard_deviation(r, labels, segment_ids)
            gstd = standard_deviation(g, labels, segment_ids)
            bstd = standard_deviation(b, labels, segment_ids)

        # compile dictionaries
        self.levels = dict(r=rmeans, g=gmeans, b=bmeans)
        self.std = dict(r=rstd, g=gstd, b=bstd)

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
        center_of_mass (dict) - {segment_id: [centroid_x, centroid_y]} pairs
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
                            r=self.levels['r'],
                            g=self.levels['g'],
                            b=self.levels['b'],
                            r_std=self.std['r'],
                            g_std=self.std['g'],
                            b_std=self.std['b'],
                            pixel_count=self.voxel_counts)

        return pd.DataFrame(measurement_data)


# class Measurement:
#     """
#     Object describes an individual cell measurement.

#     Format is designed for agreement with FlyEye Silhouette data.
#     """

#     def __init__(self, _id, centroid, color_avg, color_std, voxel_count):
#         self._id = _id
#         self.centroid_x = centroid[0]
#         self.centroid_y = centroid[1]
#         self.r = color_avg[0]
#         self.g = color_avg[1]
#         self.b = color_avg[2]
#         self.r_std = color_std[0]
#         self.g_std = color_std[1]
#         self.b_std = color_std[2]
#         self.pixel_count = voxel_count

#     def to_json(self):
#         """ Serialize measurement in JSON format. """
#         return {
#             'segment_id': int(self._id),
#             'centroid_x': self.centroid_x,
#             'centroid_y': self.centroid_y,

#             'r': self.r,
#             'g': self.g,
#             'b': self.b,

#             'r_std': self.r_std,
#             'g_std': self.g_std,
#             'b_std': self.b_std,

#             'pixel_count': int(self.pixel_count)}


# class Measurements:
#     """
#     Object describes a collection of cell measurements.

#     Format is designed for agreement with FlyEye Silhouette data.
#     """

#     def __init__(self, _ids, centroids, color_avgs, color_stds, volume):
#         self._ids = _ids
#         self.centroids = centroids
#         self.color_avgs = list(zip(*color_avgs))
#         self.color_stds = list(zip(*color_stds))
#         self.voxel_counts = volume
#         self.size = len(self._ids)

#     def get_measurement(self, index):
#         _id = self._ids[index]
#         centroid = self.centroids[index]
#         color_avg = self.color_avgs[index]
#         color_std = self.color_stds[index]
#         voxel_count = self.voxel_counts[index]
#         return Measurement(_id, centroid, color_avg, color_std, voxel_count)

#     def to_json(self):
#         return [self.get_measurement(i).to_json() for i in range(self.size)]
