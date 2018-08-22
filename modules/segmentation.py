"""
TO DO:

"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from matplotlib.colors import ListedColormap
from scipy import ndimage
from scipy.ndimage.measurements import center_of_mass


class Segmentation:
    def __init__(self, image, seed_kws={}, seg_kws={}):
        image.set_otsu_mask()
        self.labels = None
        self.seeds = self.get_seeds_from_distance(image.mask, **seed_kws)
        self.update_cmap()
        self.watershed(image.mask, **seg_kws)
        self.exclude_edge_segments()
        self.segment_ids = np.array(list(self.seeds.keys()))

    @staticmethod
    def array_to_dict(arr):
        """ Convert array to dictionary. """
        if arr is None:
            return {}
        else:
            return {i+1: row for i, row in enumerate(arr)}

    @staticmethod
    def find_maxima(im,
                    min_distance=1,
                    num_peaks=np.inf):
        """ Find local maxima of euclidean distance transform. """
        seeds = peak_local_max(im, min_distance=min_distance, num_peaks=num_peaks, exclude_border=False)
        return seeds

    @classmethod
    def get_seeds_from_distance(cls, mask,
                                sigma=2,
                                min_distance=1,
                                num_peaks=np.inf):
        """ Seed detection via euclidean distance transform of binary map. """

        # get values
        values = ndimage.distance_transform_edt(mask).astype(float)

        # apply gaussian filter
        if sigma is not None:
            values = ndimage.gaussian_filter(values, sigma)

        seeds = cls.find_maxima(values, min_distance=min_distance, num_peaks=num_peaks)
        return cls.array_to_dict(seeds)

    @classmethod
    def get_segment_mask(cls, im, seeds):
        """ Get mask for markers. """

        # create marker mask
        seed_mask = np.zeros_like(im, dtype=int)
        shape = np.array(seed_mask.shape).reshape(-1, 1)
        for seed_id, zyx in seeds.items():
            indices = zyx.reshape(-1, 1)
            accepted = np.alltrue((indices >= 0) & (indices < shape), axis=0)
            indices = indices[:, accepted]
            seed_mask[indices[0], indices[1]] = seed_id
        return seed_mask

    def watershed(self, mask, sigma=0.5, watershed_line=True):
        """ Run watershed segmentation. """

        # define distances
        distances = ndimage.distance_transform_edt(mask)
        distances = ndimage.gaussian_filter(distances, sigma=sigma)

        # run segmentation
        connectivity = ndimage.iterate_structure(ndimage.generate_binary_structure(2, 1), 1)
        markers = self.get_segment_mask(distances, self.seeds)
        self.labels = watershed(-distances, markers=markers, mask=mask, connectivity=connectivity, watershed_line=watershed_line)

    def update_cmap(self):
        """ Use current seeds to create colormap. """
        bg_color = np.array([[.8,.8,.8]])
        segment_colors = np.random.rand(len(self.seeds), 3)
        self.cmap = ListedColormap(np.vstack((bg_color, segment_colors)))

    @staticmethod
    def get_borders(im):
        """ Returns boolean array with borders masked as True. """
        mask = np.zeros_like(im, dtype=bool)
        mask[0, :] = True
        mask[-1, :] = True
        mask[:, 0] = True
        mask[:, -1] = True
        return mask

    def exclude_edge_segments(self):
        """ Removes segments overlaying the edge_mask. """
        excluded_segments = np.unique(self.labels[self.get_borders(self.labels)])
        exclusion_mask = np.isin(self.labels, excluded_segments)

        # set segments to zero and remove seeds
        self.labels[exclusion_mask] = 0
        list(map(self.seeds.__delitem__, filter(self.seeds.__contains__, excluded_segments)))

    def exclude_small_segments(self, min_area=10):
        """ Exclude small segments. """

        # identify small segments
        bins = np.arange(1, self.labels.max()+2)
        voxels = self.labels[self.labels!=0]
        counts, _ = np.histogram(voxels, bins=bins)

        excluded = bins[:-1][np.logical_and(np.isin(np.arange(1, counts.size+1), self.segment_ids), counts < min_area)]
        #excluded = bins[:-1][np.logical_and(counts > 0, counts < min_area)]

        # remove small segments
        self.labels[np.isin(self.labels, excluded)] = 0
        _ = [self.seeds.pop(seed) for seed in excluded]
        self.segment_ids = np.array(list(self.seeds.keys()))

    def get_centroids(self):
        """ Set centroids to center of mass of segmentation. """
        return self.evaluate_centroids(self.labels)

    @staticmethod
    def evaluate_centroids(labels):
        """
        Evaluate center of mass of each label.

        * Note: scipy returns centroids as (y, x) which are flipped to (x, y)
        """

        seg_ids = np.unique(labels[labels!=0])
        coms = center_of_mass(labels, labels, seg_ids)
        return {seg_id: com[::-1] for seg_id, com in zip(seg_ids, coms)}

    def show(self, figsize=(15, 15)):
        """ Visualize segmentation. """
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(self.labels, cmap=self.cmap)


