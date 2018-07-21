"""
TO DO:

"""
import os
import json
import numpy as np
import pandas as pd
import gc
import glob
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import mean, standard_deviation
from modules.io import IO
from modules.image import MultichannelImage
from modules.contours import Contours
from modules.kde import KDE
from modules.rbf import RBF
from modules.masking import ForegroundMask, RBFMask
from modules.segmentation import Segmentation
from modules.annotation import Annotation


class Stack:
    def __init__(self, tif_path, bits=2**12, params={}):
        disc_name = tif_path.split('/')[-1].split('.')[0]
        self.disc_name = disc_name
        self.genotype_path = tif_path.rsplit('/', maxsplit=1)[0]
        self.tif_path = tif_path
        self.bits = bits
        self.stack = self.read_tif(tif_path) / bits
        self.shape = self.stack.shape
        self.depth = self.shape[0]
        self.params = params
        self.df = None
        self.path = None

    @staticmethod
    def read_tif(path):
        """ Loads raw tif. Note: images are flipped from BGR to RGB """
        io = IO()
        stack = io.read_tiff(path)
        if len(stack.shape) == 3:
            stack = stack.reshape(1, *stack.shape)
        stack = np.swapaxes(stack, 1, 2)
        stack = np.swapaxes(stack, 2, 3)
        return stack[:, :, :, ::-1]

    def get_layer(self, layer_id=0):
        """ Instantiate MultiChannel image of single layer. """
        im = self.stack[layer_id, :, :, :]
        layer = Layer(im, layer_id=layer_id)
        return layer

    def load_layer(self, layer_id=0):
        """ Load segmented MultiChannel image of single layer. """
        layer_path = os.path.join(self.path, '{:d}'.format(layer_id))
        im = self.stack[layer_id, :, :, :]
        return Layer.load(im, layer_path)

    def segment(self, bg='b',
                    segmentation_kw={},
                    fg_kw={},
                    annotation_kw={},
                    save_kw={},
                    save=False):
        """ Segment and annotate all layers in stack. """

        # create directory and save parameters
        self.params = dict(segmentation_kw=segmentation_kw, fg_kw=fg_kw, annotation_kw=annotation_kw)
        if save:
            dirpath = self.create_directory()

        # segment and annotate each layer
        dfs = []
        for layer_id, im in enumerate(self.stack):
            layer = Layer(im, layer_id=layer_id)
            layer.segment(bg=bg, **segmentation_kw)
            layer.compile_dataframe()
            layer.set_foreground(**fg_kw)
            layer.annotate(**annotation_kw)

            # save layer segmentation
            if save:
                layer_dirpath = layer.save(dirpath, **save_kw)
            dfs.append(layer.df)

        # compile dataframe
        self.df = pd.concat(dfs).reset_index(drop=True)
        if save:
            self.save(dirpath)

    def create_directory(self):
        """ Create directory for segmentation results. """

        # instantiate IO
        io = IO()

        # create directory
        dirname = self.disc_name
        dirpath = os.path.join(self.genotype_path, dirname)
        dirpath = io.make_dir(dirpath, force=True)

        return dirpath

    def save(self, dirpath):
        """ Save segmentation parameters and results. """

        # instantiate IO
        io = IO()

        # save metadata to json
        metadata = dict(path=self.tif_path,
                        bits=self.bits,
                        params=self.params)
        io.write_json(os.path.join(dirpath, 'metadata.json'), metadata)

        # save contours to json
        contours = self.df.to_json()
        io.write_json(os.path.join(dirpath, 'contours.json'), contours)

        return dirpath

    @staticmethod
    def from_segmentation(path):
        """ Load segmented Stack from file. """

        io = IO()

        # load metadata
        metadata_path = os.path.join(path, 'metadata.json')
        metadata = io.read_json(metadata_path)

        # parse metadata
        tif_path = metadata['path']
        bits = metadata['bits']
        params = metadata['params']

        # instantiate stack
        stack = Stack(tif_path, bits=bits, params=params)
        stack.path = path

        # load measurements
        contours_path = os.path.join(path, 'contours.json')
        stack.df = pd.read_json(io.read_json(contours_path))

        return stack


class Layer(MultichannelImage):

    def __init__(self, im, labels=None, layer_id=0):
        MultichannelImage.__init__(self, im, labels=labels)
        self.channels = dict(r=0, g=1, b=2)
        self.layer_id = int(layer_id)

    @staticmethod
    def load(im, path):
        """ Instantiate layer with saved contour labels. """

        # load labels and metadata
        io = IO()
        labels = io.read_npy(os.path.join(path, 'labels.npy'))
        metadata = io.read_json(os.path.join(path, 'layer_metadata.json'))

        # instantiate layer
        layer_id = metadata['layer_id']
        layer = Layer(im, labels=labels, layer_id=layer_id)

        # assign background
        bg = metadata['bg']
        layer.bg = bg

        # assign data
        df = pd.read_json(io.read_json(os.path.join(path, 'contours.json')))
        layer.df = df

        # set foreground
        md_path = os.path.join(os.path.join(path, os.pardir), 'metadata.json')
        params = io.read_json(md_path)['params']
        layer.set_foreground(**params['fg_kw'])

        return layer

    def segment(self, bg='b', **kwargs):
        """ Run watershed segmentation on specified background channel. """
        self.bg = bg
        background = self.get_channel(bg)
        background.segment(**kwargs)
        self.labels = background.labels

    def measure(self):
        """ Measure fluorescence intensities to generate contours. """

        # get image channels
        drop_axis = lambda x: x.reshape(*x.shape[:2])
        r, g, b = [drop_axis(x) for x in np.split(self.im, 3, axis=-1)]

        # get segment ids (ordered)
        segment_ids = np.unique(self.labels[self.labels.nonzero()])

        # get centroids
        centroid_dict = Segmentation.evaluate_centroids(self.labels)
        centroids = [centroid_dict[seg_id] for seg_id in segment_ids]

        # compute means
        rmeans = mean(r, self.labels, segment_ids)
        gmeans = mean(g, self.labels, segment_ids)
        bmeans = mean(b, self.labels, segment_ids)
        color_avg = (rmeans, gmeans, bmeans)

        # compute std
        rstd = standard_deviation(r, self.labels, segment_ids)
        gstd = standard_deviation(g, self.labels, segment_ids)
        bstd = standard_deviation(b, self.labels, segment_ids)
        color_std = (rstd, gstd, bstd)

        # compute segment size
        voxels = self.labels[self.labels!=0]
        bins = np.arange(0, segment_ids.max()+3, 1)
        counts, _ = np.histogram(voxels, bins=bins)
        voxel_counts = counts[segment_ids]

        # createlist of contour dicts (useless but fits with Silhouette)
        data = (segment_ids, centroids, color_avg, color_std, voxel_counts)
        contours = Contours(*data).to_json()

        return contours

    def compile_dataframe(self):
        """ Compile dataframe from cell measurements. """
        columns = ['id', 'centroid_x', 'centroid_y',
                   'r', 'r_std', 'g', 'g_std', 'b', 'b_std', 'pixel_count']
        contours = self.measure()
        df = pd.DataFrame.from_records(contours, columns=columns)
        df['layer'] = self.layer_id

        # normalize by background
        for channel in 'rgb'.strip(self.bg):
            df[channel+'_normalized'] = df[channel] / df[self.bg]

        self.df = df

    def set_foreground(self, bandwidth=100, n=2):
        """ Threshold cell position density to set foreground. """
        kde = KDE(self.df, bandwidth=bandwidth, n=n)
        self.df['foreground'] = kde.mask
        self.fg_mask = ForegroundMask(self.shape, kde)

    def annotate(self, q=95,
                 channel='r_normalized',
                 weighted=True,
                 fg_only=True,
                 upper_bound=90):
        """ Annotate layer. """
        kw = dict(q=q, channel=channel, weighted=weighted, fg_only=fg_only, upper_bound=upper_bound)
        self.annotation = Annotation(self.df, **kw)
        self.df['genotype'] = self.annotation(self.df.index)

    def plot_annotation(self, cmap=None, fig_kw={}, clone_kw={}):
        """ Show annotation channel overlayed with clone segments. """

        # define colormap
        if cmap is None:
            cmap = plt.cm.Greys

        # get layer
        if 'normalized' in self.annotation.channel:
            im = self.get_channel(self.annotation.channel.split('_')[0])
        else:
            im = self.get_channel(self.annotation.channel)

        # show layer
        fig = im.show(segments=False, cmap=cmap, **fig_kw)

        # add clones
        self.annotation.plot_clones(fig.axes[0], **clone_kw)

        # mask background
        if self.annotation.fg_only:
            self.fg_mask.add_contourf(fig.axes[0], alpha=0.5)

        return fig

    def save(self,
             segmentation_path,
             segmentation=True,
             image=True,
             foreground=True,
             annotation=True,
             dpi=100):
        """ Save segmentation parameters and results. """

        # instantiate IO
        io = IO()

        # create directory
        dirpath = os.path.join(segmentation_path, '{:d}'.format(self.layer_id))
        dirpath = io.make_dir(dirpath, force=True)

        # save contours
        contours = self.df.to_json()
        io.write_json(os.path.join(dirpath, 'contours.json'), contours)

        # save metadata
        metadata = dict(layer_id=self.layer_id, bg=self.bg)
        io.write_json(os.path.join(dirpath, 'layer_metadata.json'), metadata)

        # save segmentation data
        if segmentation:
            labels_path = os.path.join(dirpath, 'labels.npy')
            np.save(labels_path, self.labels)

        # set image keyword arguments
        im_kw = dict(format='png', dpi=dpi, bbox_inches='tight', pad_inches=0)

        # save segmentation image
        if image:
            im_path = os.path.join(dirpath, 'segmentation.png')
            bg = self.get_channel(self.bg, copy=False)
            fig = bg.show(segments=True)
            fig.axes[0].axis('off')
            fig.savefig(im_path, **im_kw)
            fig.clf()
            plt.close(fig)
            gc.collect()

        # save foreground image
        if foreground:
            im_path = os.path.join(dirpath, 'foreground.png')
            bg = self.get_channel(self.bg, copy=False)
            fig = bg.show(segments=False)
            self.fg_mask.add_contourf(fig.axes[0], alpha=0.5)
            fig.savefig(im_path, **im_kw)
            fig.clf()
            plt.close(fig)
            gc.collect()

        # save annotation image
        if annotation:
            im_path = os.path.join(dirpath, 'annotation.png')
            fig = self.plot_annotation()
            fig.savefig(im_path, **im_kw)
            fig.clf()
            plt.close(fig)
            gc.collect()

        return dirpath




