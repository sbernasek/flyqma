from os.path import join, abspath, exists
import json
import pandas as pd


class ReadSilhouette:
    """

    Read-only interface to a FlyEye Silhouette file.

    Attributes:

        path (str) - path to Silhouette file

        feed (dict) - feed file containing layer IDs

        feud (dict) - feud file containing cell type labels

    Properties:

        is_flipped_about_yz (bool) - if True, invert about YZ plane

        is_flipped_about_xy (bool) - if True, invert about XY plane

    """
    def __init__(self, path):
        """
        Instantiate interface to silhouette file.

        Args:

            path (str) - path to silhouette file

        """
        self.path = abspath(path)

        # load feed and feud files
        self.feed = self.read_json('feed.json')
        self.feud = self.read_json('feud.json')

    def read_json(self, filename):
        """
        Read contents of specified JSON file.

        Args:

            filename (str) - filename

        Returns:

            out (dict) - file contents

        """
        filepath = join(self.path, filename)
        with open(filepath, 'r') as f:
            out = json.load(f)
        return out


class ReadSilhouetteData(ReadSilhouette):
    """

    Read-only interface to data within a FlyEye Silhouette file.

    Upon instantiation, individual cell measurements are aggregated into a data.cells.Cells compatible DataFrame.

    Measurement data must be read on a layer-by-layer basis the first time a Silhouette object is instantiated. Following this initial reading, the aggregated measurement data are serialized and stored within the silhouette file. These serialized measurements may then be accessed directly during future use. The recompile flag indicates whether the serialized measurements should be ignored upon instantiation.

    Attributes:

        df (pd.DataFrame) - cell measurement data

    Inherited attributes:

        path (str) - path to Silhouette file

        feed (dict) - feed file containing layer IDs

        feud (dict) - feud file containing cell type labels

    """

    def __init__(self, path, recompile=False):
        """
        Instantiate interface to silhouette file data.

        Args:

            path (str) - path to silhouette file

            recompile (bool) - if True, recompile measurements from all layers

        """
        super().__init__(path)
        self.load(recompile=recompile)

    @property
    def labels(self):
        """ pd.Series of labels keyed by (layer_id, segment_id). """
        return self.df.set_index(['layer', 'segment_id'])['label']

    def compile_measurements(self):
        """ Compile measurements from all layers (slow access). """
        labels = self.read_labels()
        self.df = self.read_contours(labels)

    def save_measurements(self):
        """ Save serialized measurements for fast access. """
        self.df.to_json(join(self.path, 'measurements.json'))

    def load_measurements(self):
        """ Load serialized measurements (fast access). """
        self.df = pd.read_json(join(self.path, 'measurements.json'))

    def load(self, recompile=False):
        """
        Read all contour and orientation data from silhouette file.

        Args:

            recompile (bool) - if True, recompile measurements from all layers

        """

        # check whether measurements are available
        measurements_available = exists(join(self.path, 'measurements.json'))

        # load available measurements if recompile flag is false
        if measurements_available and not recompile:
            self.load_measurements()

        # otherwise, recompile and save measurements
        else:
            self.compile_measurements()
            self.save_measurements()

    def read_labels(self):
        """
        Load segment labels from silhouette file.

        Returns:

            labels (dict) - {layer_id: {contour_id: label}} entries for each layer

        """

        # compile labels for all layers
        labels = {}
        for layer in self.feud['layers']:

            # compile {contour_id: contour_label} dictionary for current layer
            annotations = {}
            for contour in layer['contours']:
                label = contour.get('label', None)
                if label is None or label.strip() == '':
                    continue
                annotations[contour['id']] = contour['label']

            # store labels for current layer
            labels[layer['id']] = annotations

        return labels

    @staticmethod
    def parse_contour(contour):
        """
        Convert contour to list format.

        Args:

            contour (dict) - contour from silhouette file

        Returns:

            ctr_list (list) - values in data.cells.Cells compatible list format

        """

        # extract values
        centroid = contour['centroid']
        pixel_count = contour['pixel_count']
        segment = contour['points']
        color_avg = contour['color_avg']
        color_std = contour['color_std']

        # reorganize values
        ctr_list = [contour['id'],
                    centroid[0], centroid[1],
                    segment,
                    pixel_count]

        # parse measurements
        keys1, avgs = list(zip(*color_avg.items()))
        keys2, stds = list(zip(*color_std.items()))
        ctr_list.extend(avgs)
        ctr_list.extend(stds)
        assert sum([k1!=k2 for k1,k2 in zip(keys1,keys2)])==0, 'Contour keys do not match.'

        return keys1, ctr_list

    def read_contours(self, all_labels={}, include_unlabeled=False):
        """
        Read contours from silhouette file.

        Args:

            all_labels (dict) - {layer_id: {contour_id: label}} for each layer

            include_unlabeled (bool) - if True, include unlabeled segments

        Returns:

            df (pd.DataFrame) - data.cells.Cells compatible dataframe of contours

        """

        # read contours from all layers
        contours = []
        for layer_id in self.feed['layer_ids']:

            # load labels for current layer
            labels = all_labels.get(layer_id, None)

            # skip layers without any labels
            if labels is None and not include_unlabeled:
                continue

            # read layer from silhouette file
            layer = self.read_json('{:d}.json'.format(layer_id))

            # read all contours within layer
            for contour in layer['contours']:

                # get label for current contour
                if labels is None:
                    label = None
                else:
                    label = labels.get(contour['id'], None)

                if label is None and not include_unlabeled:
                    continue

                # convert to list format
                keys, ctr_list = self.parse_contour(contour)
                ctr_list.extend([layer_id, label])

                # store contours from current layer
                contours.append(ctr_list)

        # compile dataframe
        columns = ['segment_id',
                   'centroid_x',
                   'centroid_y',
                   'segment',
                   'pixel_count']
        columns += keys
        columns += ['{:s}_std'.format(k) for k in keys]
        columns += ['layer', 'label']
        df = pd.DataFrame(contours, columns=columns)

        # delete duplicate RGB channel labels
        for i, c in enumerate('rgb'):
            if 'ch{:d}'.format(i) in df.columns:
                df.drop(c, axis=1, inplace=True)
                df.drop(c+'_std', axis=1, inplace=True)

        return df
