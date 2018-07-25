import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from modules.io import IO


class PathSelector:

    def __init__(self, layer, channel=None, **kw):

        # plot figure
        self.create_figure(**kw)

        if channel is not None:
            ch = layer.get_channel(channel)
            ch.show(segments=False, ax=self.ax)
        else:
            layer.show(segments=False, ax=self.ax)

        # instantiate interface objects
        self.pts = []
        self.active_polygon = False
        self.path = layer.path
        self.include = True
        self.connect()

    def connect(self):
        self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.cid_press = self.fig.canvas.mpl_connect('key_press_event', self.on_key)

    def disconnect(self):
        self.fig.canvas.mpl_disconnect(self.cid_click)
        self.fig.canvas.mpl_disconnect(self.cid_press)

    def create_figure(self, **kw):
        self.fig = plt.figure(**kw)
        self.ax = self.fig.subplots()

    def plot(self, x, y):
        self.ax.scatter(x, y)

    def on_click(self, event):
        pt = (event.xdata, event.ydata)
        if None not in pt:
            self.add_point(pt)

    def undo(self):
        self.remove_point()
        self.remove_polygon()
        if self.active_polygon:
            self.add_polygon()

    def remove_point(self):
        _ = self.pts.pop()
        self.ax.lines[-1].remove()
        if len(self.ax.lines) > 0:
            self.ax.lines[-1].set_color('r')
            self.ax.lines[-1].set_markersize(20)

        if len(self.pts) < 3:
            self.active_polygon = False

    def remove_polygon(self):
        self.ax.patches[0].remove()

    def add_point(self, pt):

        # store point
        self.pts.append(pt)

        # add marker to plot
        if len(self.ax.lines) > 0:
            self.ax.lines[-1].set_color('y')
            self.ax.lines[-1].set_markersize(10)

        self.ax.plot(*pt, '.', markersize=20, color='r')

        # update polygon
        if self.active_polygon:
            self.update_polygon(pt)
        elif len(self.pts) == 3:
            self.add_polygon()
            self.active_polygon = True

    def add_polygon(self):
        poly = Polygon(self.pts, ec=(1,1,0,1), lw=3, fc=(1,1,1,0.2), fill=False, zorder=1, closed=True)
        self.ax.add_patch(poly)

    def update_polygon(self, pt):
        pt_arr = np.array(pt).reshape(1, 2)
        self.ax.patches[0].remove()
        self.add_polygon()

    def on_key(self, event):

        # save and disconnect
        if event.key == 'x':
            self.save()
            self.disconnect()
            self.overlay()

        elif event.key == 'e':
            self.include = False

        # undo
        elif event.key == 'z':
            self.undo()

    def overlay(self):
        self.ax.images[0].set_alpha(0.5)
        xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim()
        self.ax.text(np.mean(xlim), np.mean(ylim), 'SELECTION\nSAVED', color='r', fontsize=78, ha='center', va='center')

    def save(self):
        io = IO()
        io.write_npy(os.path.join(self.path, 'selection.npy'), np.array(self.pts))
        md = dict(include=self.include)
        io.write_json(os.path.join(self.path, 'md.json'), md)
