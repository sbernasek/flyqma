import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from modules.io import IO
import gc
from matplotlib.gridspec import GridSpec


class PathSelector:

    def __init__(self, disc_id, layer, figsize=None, **kw):

        # plot figure
        self.create_figure()
        self.add_image(layer, **kw)

        # instantiate interface objects
        self.pts = []
        self.active_polygon = False

        # set paths to segmentation and selection directories
        self.selection_path = os.path.join('./stack_selection', str(disc_id))

        # set attributes
        self.exemplar = False
        self.traceback = []
        self.clicks = []
        self.keys = []

        # establish connection
        self.connect()

    def connect(self):
        self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.cid_press = self.fig.canvas.mpl_connect('key_press_event', self.on_key)

    def disconnect(self):
        self.fig.canvas.mpl_disconnect(self.cid_click)
        self.fig.canvas.mpl_disconnect(self.cid_press)

    def exit(self, msg=None):
        """ Save selection, disconnect event handling, and overlay message. """
        self.save()
        self.disconnect()
        if msg is not None:
            self.overlay(msg)

    def on_key(self, event):
        self.keys.append(event)

        # save and disconnect
        if event.key == 'x':
            self.include = True
            self.duplicate = False
            self.exit('SAVED')

        # mark as excluded and exit
        elif event.key == 'e':
            self.include = False
            self.duplicate = False
            self.overlay('EXCLUDED')
            self.exit()

        # mark as duplicate and exit
        elif event.key == 'd':
            self.include = False
            self.duplicate = True
            self.overlay('DUPLICATE')
            self.exit()

        # mark as exemplar
        elif event.key == 'g':
            self.exemplar = True

        # undo
        elif event.key == 'z':
            self.undo()

    def on_click(self, event):
        self.clicks.append(event)
        pt = (event.xdata, event.ydata)
        if None not in pt:
            self.add_point(pt)

    def add_point(self, pt):

        # store point
        self.pts.append(pt)

        # update previous marker and add new marker
        self.update_marker(color='y', markersize=5)
        self.add_marker(*pt, color='r', markersize=10)

        # update polygon
        if self.active_polygon:
            self.update_polygon()
        elif len(self.pts) == 3:
            self.add_polygon()
            self.active_polygon = True

    def undo(self):
        self.remove_point()
        self.remove_polygon()
        if self.active_polygon:
            self.add_polygon()

    def remove_point(self):
        _ = self.pts.pop()
        self.remove_marker()
        self.update_marker(color='r', markersize=10)
        if len(self.pts) < 3:
            self.active_polygon = False

    def add_marker(self, x, y, color='k', markersize=10):
        for ax in self.axes:
            ax.plot(x, y, '.', color=color, markersize=markersize, zorder=2)

    def remove_marker(self):
        for ax in self.axes:
            ax.lines[-1].remove()

    def update_marker(self, color, markersize):
        for ax in self.axes:
            if len(ax.lines) > 0:
                ax.lines[-1].set_color(color)
                ax.lines[-1].set_markersize(markersize)

    def clear_markers(self):
        """" Remove all markers. """
        while len(self.axes[0].lines) > 0:
            self.remove_marker()

    def add_polygon(self):
        for ax in self.axes:
            poly = Polygon(self.pts,
                       ec=(1,0,0,1), lw=1,
                       fc=(1,1,1,0.2), fill=False,
                       zorder=1, closed=True)
            ax.add_patch(poly)

    def remove_polygon(self):
        for ax in self.axes:
            ax.patches[0].remove()

    def update_polygon(self):
        self.remove_polygon()
        self.add_polygon()

    def create_figure(self, figsize=(7, 7)):
        self.fig = plt.figure(figsize=figsize)
        ax = self.fig.subplots()
        self.axes = (ax,)

    def add_image(self, layer, channel=None):
        if channel is not None:
            layer.get_channel(channel).show(segments=False, ax=self.axes[0])
        else:
            layer.show(segments=False, ax=self.axes[0])

    def overlay(self, msg):
        """ Overlay string centered on image. """
        for ax in self.axes:
            ax.images[0].set_alpha(0.5)
            xlim, ylim = ax.get_xlim(), ax.get_ylim()
            ax.text(np.mean(xlim), np.mean(ylim), msg, color='r', fontsize=24, ha='center', va='center')

    def save_msg(self):
        self.overlay('SELECTION\nSAVED')

    def exclusion_msg(self):
        self.overlay('EXCLUDED')

    def save(self, image=True):
        """ Save selection and selection graphic to file. """

        # save selection image
        if image:
            self.clear_markers()
            im_kw = dict(format='png', dpi=100, bbox_inches='tight', pad_inches=0)
            im_path = os.path.join(self.selection_path, 'selection.png')
            self.fig.savefig(im_path, **im_kw)
            # self.fig.clf()
            # plt.close(self.fig)
            # gc.collect()

        # save coordinates
        io = IO()
        pts = np.array(self.pts)
        io.write_npy(os.path.join(self.selection_path, 'selection.npy'), pts)


class MultiPathSelector(PathSelector):

    def create_figure(self):
        fig = plt.figure(figsize=(2*6.75, 2*2.25))
        gs = GridSpec(nrows=1, ncols=3, wspace=.01, hspace=.01)

        self.fig = fig

        ax0 = fig.add_subplot(gs[0])
        ax1 = fig.add_subplot(gs[1])
        ax2 = fig.add_subplot(gs[2])

        self.axes = (ax0, ax1, ax2)
        for ax in self.axes:
            ax.set_aspect(1)
            ax.axis('off')

    def add_image(self, layer, cmap=None):
        titles = dict(b='DAPI', g='PntGFP', r='UbiRFP')
        for ch, ax in zip('brg', self.axes):
            layer.get_channel(ch).show(segments=False, cmap=cmap, ax=ax)
            ax.set_title(titles[ch], fontsize=14)
        for ax in self.axes:
            ax.set_aspect(1)
        plt.tight_layout()


class LayerSelector(MultiPathSelector):

    def __init__(self, disc_id, layer, figsize=None, **kw):

        MultiPathSelector.__init__(self, disc_id, layer, figsize=figsize, **kw)

        # set paths to segmentation and selection directories
        d_id = str(disc_id)
        l_id = str(layer.layer_id)

        selection_path = os.path.join('./stack_selection', d_id, l_id)
        if not os.path.exists(selection_path):
            os.mkdir(selection_path)
        self.selection_path = selection_path

    def create_figure(self):
        fig = plt.figure(figsize=(6.75, 2.25))
        gs = GridSpec(nrows=1, ncols=3, wspace=.01, hspace=.01)

        self.fig = fig

        ax0 = fig.add_subplot(gs[0])
        ax1 = fig.add_subplot(gs[1])
        ax2 = fig.add_subplot(gs[2])

        self.axes = (ax0, ax1, ax2)
        for ax in self.axes:
            ax.set_aspect(1)
            ax.axis('off')

    def add_image(self, layer, cmap=None):
        titles = dict(b='DAPI', g='PntGFP', r='UbiRFP')
        for ch, ax in zip('brg', self.axes):
            layer.get_channel(ch).show(segments=False, cmap=cmap, ax=ax)
            ax.set_title(titles[ch], fontsize=14)
        for ax in self.axes:
            ax.set_aspect(1)
        plt.tight_layout()


class StackSelector:

    def __init__(self, disc_id, stack, layers, **kw):

        # set paths to segmentation and selection directories
        self.selection_path = os.path.join('./stack_selection', str(disc_id))

        # create figure
        fig, layer_to_row, ax_to_layer = self.create_figure(stack, layers)
        self.fig = fig
        self.rows = layer_to_row
        self.ax_to_layer = ax_to_layer

        # add images
        self.add_stack_images(stack, layers, **kw)

        # set attributes
        self.traceback = []
        self.saved = False

        # establish connection
        self.connect()

    @staticmethod
    def load(stack, **kw):
        sel = StackSelector(stack, **kw)
        sel.disconnect()
        _ = [row.load() for row in sel.rows.values()]
        return sel

    def connect(self):
        self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.cid_press = self.fig.canvas.mpl_connect('key_press_event', self.on_key)

    def disconnect(self):
        self.fig.canvas.mpl_disconnect(self.cid_click)
        self.fig.canvas.mpl_disconnect(self.cid_press)

    def exit(self):
        """ Save and exit. """
        self.save()
        self.disconnect()
        self.fig.clf()
        plt.close(self.fig)

    def on_key(self, event):
        """ Key action. """

        row = self.get_event_row(event)

        # save and disconnect
        if event.key == 'x':
            try:
                self.exit()
            except Exception as error:
                self.traceback.append(error)

        # mark as excluded and exit
        elif event.key == 'n':
            row.include = False
            row.overlay('NEURONS\n&\nCONES')

        # mark as duplicate and exit
        elif event.key == 'd':
            row.include = False
            row.duplicate = True
            row.overlay('DUPLICATE')

        # mark as exemplar
        elif event.key == 'e':
            row.exemplar = True

        # undo
        elif event.key == 'z':
            row.undo()

        # clear
        elif event.key == 'm':
            row.clear()

    def get_event_row(self, event):
        """ Get row where event took place. """
        return self.rows[self.ax_to_layer[event.inaxes]]

    def on_click(self, event):

        """ Click action. """
        row = self.get_event_row(event)
        pt = (event.xdata, event.ydata)
        if None not in pt:
            row.add_point(pt)

    def create_figure(self, stack, layers):

        fig = plt.figure(figsize=(6.75, 2.25*len(layers)))
        gs = GridSpec(nrows=len(layers), ncols=3, wspace=.01, hspace=.01)

        layer_to_row, ax_to_layer = {}, {}
        for i, layer_id in enumerate(layers):

            # create axes
            ax0 = fig.add_subplot(gs[i*3])
            ax1 = fig.add_subplot(gs[i*3+1])
            ax2 = fig.add_subplot(gs[i*3+2])
            axes=(ax0, ax1, ax2)

            # instantiate row
            layer_path = os.path.join(self.selection_path, '{:d}'.format(layer_id))
            layer_to_row[layer_id] = Row(layer_path, axes)
            for ax in axes:
                ax_to_layer[ax] = layer_id

        return fig, layer_to_row, ax_to_layer

    def add_stack_images(self, stack, layers, **kw):

        for i, layer_id in enumerate(layers):

            row = self.rows[layer_id]
            (ax0, ax1, ax2) = row.axes

            # get and plot layer
            layer = stack.get_layer(layer_id)
            self.add_layer_images(layer, row.axes, **kw)

            # add layer number
            ax0.text(0, 0, ' {:d}'.format(layer_id), fontsize=14, color='y', va='top')

            if layer_id == layers[0]:
                name = stack.disc_name
                ax0.set_title('Disc {}\nDAPI'.format(name), fontsize=14)
                ax1.set_title('Disc {}\nUbiRFP'.format(name), fontsize=14)
                ax2.set_title('Disc {}\nPntGFP'.format(name), fontsize=14)

    @staticmethod
    def add_layer_images(layer, axes, cmap=None):

        # add image to each axis
        ax0, ax1, ax2 = axes
        _ = layer.get_channel('b').show(segments=False, ax=ax0, cmap=cmap)
        _ = layer.get_channel('r').show(segments=False, ax=ax1, cmap=cmap)
        _ = layer.get_channel('g').show(segments=False, ax=ax2, cmap=cmap)
        for ax in axes:
            ax.set_aspect(1)
        plt.tight_layout()

    def save(self, image=True):
        """ Save selection and selection graphic to file. """

        # save points for all rows
        for row in self.rows.values():
            row.clear_markers()
            row.save()

        # save selection image
        if image:
            kw = dict(format='png', dpi=200, bbox_inches='tight', pad_inches=0)
            im_path = os.path.join(self.selection_path, 'selection.png')
            self.fig.savefig(im_path, **kw)

        self.saved = True


class Row:

    def __init__(self, path, axes):
        self.path = path
        self.axes = axes

        # set attributes
        self.include = True
        self.duplicate = False
        self.exemplar = False
        self.active_polygon = False

        # initialize points list
        self.pts = []
        self.traceback = []

    def load(self):
        io = IO()
        pts = io.read_npy(os.path.join(self.path, 'selection.npy'))
        self.pts = pts.tolist()

        md = io.read_json(os.path.join(self.path, 'md.json'))
        self.include = md['include']
        self.duplicate = md['duplicate']
        self.exemplar = md['exemplar']

        if len(self.pts) >= 3:
            self.add_polygon()

        if self.include==False and self.duplicate==False:
            self.overlay('NEURONS\n&\nCONES')

        if self.include==False and self.duplicate==True:
            self.overlay('DUPLICATE')

    def clear(self):
        """ Clear row. """
        self.pts = []
        self.clear_markers()
        if len(self.axes[0].patches) > 0:
            self.remove_polygon()
            self.active_polygon = False

    def add_point(self, pt):

        # store point
        self.pts.append(pt)

        # update previous marker and add new marker
        self.update_marker(color='y', markersize=5)
        self.add_marker(*pt, color='r', markersize=10)

        # update polygon
        if self.active_polygon:
            self.update_polygon()
        elif len(self.pts) == 3:
            self.add_polygon()
            self.active_polygon = True

    def undo(self):
        self.remove_point()
        self.remove_polygon()
        if self.active_polygon:
            self.add_polygon()

    def remove_point(self):
        _ = self.pts.pop()
        self.remove_marker()
        self.update_marker(color='r', markersize=10)
        if len(self.pts) < 3:
            self.active_polygon = False

    def add_marker(self, x, y, color='k', markersize=10):
        for ax in self.axes:
            ax.plot(x, y, '.', color=color, markersize=markersize, zorder=2)

    def remove_marker(self):
        for ax in self.axes:
            ax.lines[-1].remove()

    def update_marker(self, color, markersize):
        for ax in self.axes:
            if len(ax.lines) > 0:
                ax.lines[-1].set_color(color)
                ax.lines[-1].set_markersize(markersize)

    def clear_markers(self):
        """" Remove all markers. """
        while len(self.axes[0].lines) > 0:
            self.remove_marker()

    def add_polygon(self):
        for ax in self.axes:
            poly = Polygon(self.pts,
                       ec=(1,1,0,1), lw=1,
                       fc=(1,1,1,0.2), fill=False,
                       zorder=1, closed=True)
            ax.add_patch(poly)

    def remove_polygon(self):
        for ax in self.axes:
            ax.patches[0].remove()

    def update_polygon(self):
        self.remove_polygon()
        self.add_polygon()

    def overlay(self, msg):
        """ Overlay string centered on image. """
        for ax in self.axes:
            ax.images[0].set_alpha(0.5)
            xlim, ylim = ax.get_xlim(), ax.get_ylim()
            ax.text(np.mean(xlim), np.mean(ylim), msg, color='k', fontsize=18, ha='center', va='center')

    def save(self):
        """ Save selected points and metadata to file. """

        io = IO()
        pts = np.array(self.pts)
        io.write_npy(os.path.join(self.path, 'selection.npy'), pts)
        md = dict(include=self.include,
                  duplicate=self.duplicate,
                  exemplar=self.exemplar)
        io.write_json(os.path.join(self.path, 'md.json'), md)
