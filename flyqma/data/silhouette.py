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

    def write_silhouette(self,
                         dst=None,
                         include_image=True,
                         channel_dict=None):
        """
        Write silhouette file.

        Args:

            dst (str) - destination directory

            include_image (bool) - save RGB image of each layer

            channel_dict (dict) - RGB channel names, keyed by channel index. If none provided, defaults to the first three channels in RGB order.

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
            layer.write_silhouette(dst,
                include_image=include_image,
                channel_dict=channel_dict)


class SilhouetteLayerIO:
    """
    Methods for converting a layer to Silhouette readable format. A layer file is structured as follows:

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
            "points": [[x1, y1], [x2, y2] ... ]}}


    """

    @staticmethod
    def _to_key(channel):
        """ Returns DataFrame key for <channel>. """
        return 'ch{:d}'.format(channel)

    @staticmethod
    def _construct_hull(properties):
        """ Returns points on convex hull. """
        hull = qhull.ConvexHull(properties.coords)
        return hull.points[hull.vertices].astype(int).tolist()

    def build_contours(self, channel_dict):
        """
        Convert dataframe to a list of contours (Silhouette format).

        Args:

            channel_dict (dict) - RGB channel names, keyed by channel index

        Returns:

            contours (list) - list of contour dictionaries

        """

        # compile contour data
        assert self.labels is not None, 'Labels are not defined.'
        properties = regionprops(self.labels.T)
        ctr_data = pd.Series({p.label: self._construct_hull(p) \
                             for p in properties}, name='points')
        data = self.data.join(ctr_data, on='segment_id')

        # coerce into Silhouette contour-list format
        data['id'] = data.segment_id
        data['centroid'] = data[['centroid_x', 'centroid_y']].values.tolist()

        # identify key maps for RGB channels
        mean_dict = {self._to_key(k): v for k, v in channel_dict.items()}
        std_dict = {k+'_std': v+'_std' for k, v in mean_dict.items()}
        keys = ['ch{:d}'.format(x) for x in range(self.colordepth)] + list(mean_dict.values())

        # append RGB mean intensities to dataframe
        data[list(mean_dict.values())] = data[list(mean_dict.keys())]
        data[list(std_dict.values())] = data[list(std_dict.keys())]
        data['color_avg'] = data[keys].to_dict(orient='records')

        # append RGB intensity variation to dataframe
        std_data = data[[k+'_std' for k in keys]]
        std_data.columns = keys
        data['color_std'] = std_data.to_dict(orient='records')

        # compile contour list
        keys = ['id', 'centroid', 'pixel_count', 'points',
                'color_avg', 'color_std']
        contours = data[keys].to_dict(orient='records')

        return contours

    def _to_silhouette(self, channel_dict):
        """
        Returns Silhouette compatible JSON format.

        Args:

            channel_dict (dict) - RGB channel names, keyed by channel index

        Returns:

            layer_dict (dict) - Silhouette compatible dictionary

        """

        # return contour dictionary
        return {
            'id': self._id,
            'imageFilename': '{:d}.png'.format(self._id),
            'contours': self.build_contours(channel_dict)}

    def _write_png(self, dst, channel_dict):
        """
        Write layer image to Silhouette-compatible RGB image in PNG format.

        Args:

            dst (str) - destination directory

            channel_dict (dict) - RGB channel names, keyed by channel index

        """
        rev_channel_dict = {v: k for k, v in channel_dict.items()}
        image_8bit = np.round(self.im * 2**8).astype(np.uint8)
        image_RGB = image_8bit[..., [rev_channel_dict[c] for c in 'rgb']]
        image = PIL.Image.fromarray(image_RGB)
        image.save(join(dst, '{:d}.png'.format(self._id)), format='png')

    def write_silhouette(self, dst,
                         include_image=True,
                         channel_dict=None):
        """
        Write silhouette compatible JSON to target directory.

        Args:

            dst (str) - destination directory

            include_image (bool) - save layer image as png

            channel_dict (dict) - RGB channel names, keyed by channel index. If none provided, defaults to the first three channels in RGB order.

        """

        # define rgb map
        if channel_dict is None:
            channel_dict = dict(enumerate('rgb'))

        filename = join(dst, '{:d}.json'.format(self._id))

        io = IO()
        io.write_json(filename, self._to_silhouette(channel_dict=channel_dict))

        if include_image:
            self._write_png(dst, channel_dict=channel_dict)
