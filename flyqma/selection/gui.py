from os.path import join
import matplotlib.pyplot as plt

from .interface import StackInterface


class GUI(StackInterface):
    """
    GUI for manually selecting regions within each layer of an image stack.

    Following instantiation, run "GUI.connect()" to begin event handling.

    Key actions:

        Z: remove last added point

        M: reset all points in layer

        S: save selection

        X: disconnect and exit GUI

        N: mark layer as neurons+cones and exclude

        D: mark layer as duplicate and exclude

        E: mark layer as exemplar

    """

    def __init__(self, stack):
        """
        Instantiate GUI.

        Args:

            stack (Stack) - image stack

        """
        super().__init__(stack)

        # set attributes
        self.traceback = []
        self.saved = False

    @staticmethod
    def load(stack):
        """
        Load selection GUI from file.

        Args:

            stack (Stack) - image stack

        Returns:

            gui (GUI) - disconnected gui

        """
        gui = GUI(stack)
        _ = [interface.load() for interface in gui.layer_to_interface.values()]
        return gui

    def save(self, image=True):
        """
        Save selection path for each layer.

        Args:

            image (bool) - if True, save overall image of selections

        """

        # save each layer
        for layer_gui in self.layer_to_interface.values():
            layer_gui.clear_markers()
            layer_gui.save()

        # save selection image
        if image:
            kw = dict(format='png', dpi=200, bbox_inches='tight', pad_inches=0)
            im_path = join(self.path, 'selection.png')
            self.fig.savefig(im_path, **kw)

        self.saved = True

    def connect(self):
        """ Connect event handling. """
        self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.cid_press = self.fig.canvas.mpl_connect('key_press_event', self.on_key)

    def disconnect(self):
        """ Disconnect event handling. """
        self.fig.canvas.mpl_disconnect(self.cid_click)
        self.fig.canvas.mpl_disconnect(self.cid_press)

    def exit(self):
        """ Disconnect and exit GUI. """
        self.disconnect()
        self.fig.clf()
        plt.close(self.fig)

    def which_layer(self, event):
        """ Returns layer ID where event took place. """
        return self.layer_to_interface[self.ax_to_layer[event.inaxes]]

    def on_click(self, event):
        """ Click action: add point. """
        row = self.which_layer(event)
        pt = (event.xdata, event.ydata)
        if None not in pt:
            row.add_point(pt)

    def on_key(self, event):
        """
        Key actions.

            Z: remove last added point

            M: reset all points in layer

            S: save selection

            X: disconnect and exit GUI

            N: mark layer as neurons+cones and exclude

            D: mark layer as duplicate and exclude

            E: mark layer as exemplar

        """

        row = self.which_layer(event)

        # save and disconnect
        if event.key == 's':
            self.save()

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
