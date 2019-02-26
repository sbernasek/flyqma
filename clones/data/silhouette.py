from os.path import join, exists
from os import mkdir
from shutil import rmtree
import numpy as np
import pandas as pd
from scipy.spatial import qhull
from skimage.measure import regionprops
import PIL

from flyeye.data.silhouette import SilhouetteData

from ..utilities import IO
from ..validation.arguments import str2bool


class SilhouetteIO:
    """
    Methods for converting a stack to Silhouette readable format.

    The Silhouette container includes a FEED file:

    FEED.json ::

        "orientation": {"flip_about_xy": false, "flip_about_yz": false},
        "layer_ids": [ 0,1,2... ],
        "params": { param_name: param_value ... } }

    """

    @property
    def silhouette_path(self):
        """ Path to Silhouette directory. """
        return join(self.path, '{:d}.silhouette'.format(self._id))

    def load_silhouette_labels(self):
        """
        Load manually assigned labels from file.

        Returns:

            labels (pd.Series) - labels keyed by (layer_id, segment_id)

        """

        # load silhouette data
        silhouette = SilhouetteData(self.silhouette_path, recompile=True)

        # convert labels to numeric scale
        label_to_value = dict(M=0, H=1, W=2)
        labels = silhouette.labels.apply(label_to_value.get)
        labels.name = 'manual_label'

        return labels

    @property
    def _feed(self):
        return {
            "layer_ids": list(range(self.depth)),
            "orientation": {
                "flip_about_xy": False,
                "flip_about_yz": False},
            "params": {
                "cell_area": 200,
                "cut_line_thickness": 2,
                "max_cut_distance": 150,
                "max_loop": 20,
                "meanshift_sp": 5,
                "meanshift_sr": 30,
                "min_hull_distance": 5,
                "min_intensity": 30,
                "opening_size": 3,
                "small_cell_area": 25,
                "subsegment_cell_area": 400,
                "subsegment_max_deep": 3,
                "subsegment_opening_size": 0,
                "total_percentage": 0}}

    def write_silhouette(self, dst=None):
        """
        Write silhouette file.

        Args:

            dst (str) - destination directory

        """

        # if no directory is specified, use stack directory
        if dst is None:
            dst = self.silhouette_path

        # create silhouette directory
        if exists(dst):
            overwrite = str2bool(input('Destination exists. Overwrite? '))
            if overwrite:
                rmtree(dst)
            else:
                return
        mkdir(dst)

        # write feed file
        io = IO()
        io.write_json(join(dst, 'feed.json'), self._feed)

        # write layer files
        for layer in self:
            layer.write_silhouette(dst)


class SilhouetteLayerIO:
    """
    Methods for converting a layer to Silhouette readable format.

    A layer file is structured as follows:

    LAYER_ID.json :

        {
        "id": LAYER_ID
        "imageFilename": "LAYER_ID.png"
        "contours": [ ... contours ... ]

            {"centroid": [CONTOUR_CENTROID_X, CONTOUR_CENTROID_Y],
            "color_avg": {"b": X, "g": X, "r": X},
            "color_std": {"b": X, "g": X, "r": X},
            "id": CONTOUR_ID,
            "pixel_count": CONTOUR_AREA,
            "points": [[x1, y1], [x2, y2] ... ]}
        }

    """

    @staticmethod
    def _construct_hull(properties):
        """ Returns points on convex hull. """
        hull = qhull.ConvexHull(properties.coords)
        return hull.points[hull.vertices].astype(int).tolist()

    @staticmethod
    def _to_contour(record):
        """
        Parse record to contour.

        """
        return {
            'id': record.segment_id,
            'centroid':  record[['centroid_x', 'centroid_y']].tolist(),
            'color_avg': dict(record[['b', 'g', 'r']]),
            'color_std': {k[0]:v for k,v in dict(record[['b_std', 'g_std', 'r_std']]).items()},
            'pixel_count': record.pixel_count,
            'points': record['points']}

    def _to_silhouette(self):
        """ Returns Silhouette compatible JSON format. """
        properties = regionprops(self.labels.T)
        ctr_data = pd.Series({p.label: self._construct_hull(p) for p in properties}, name='points')
        data = self.data.join(ctr_data, on='segment_id')
        return {
            'id': self._id,
            'imageFilename': '{:d}.png'.format(self._id),
            'contours': data.apply(self._to_contour, axis=1).tolist()}

    def _write_png(self, dirpath):
        """ Write layer image to Silhouette-compatible PNG. """
        rgb_image = np.round(self.im * 2**8).astype(np.uint8)
        image = PIL.Image.fromarray(rgb_image)
        image.save(join(dirpath, '{:d}.png'.format(self._id)), format='png')

    def write_silhouette(self, dirpath, include_image=True):
        """
        Write silhouette compatible JSON to target directory.

        Args:

            dirpath (str) - target directory

            include_image (bool) - save layer image as png

        """
        filename = join(dirpath, '{:d}.json'.format(self._id))

        io = IO()
        io.write_json(filename, self._to_silhouette())

        if include_image:
            self._write_png(dirpath)
