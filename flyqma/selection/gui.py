from os.path import join
import matplotlib.pyplot as plt

from .interface import StackInterface


class GUI(StackInterface):
    """
    GUI for manually selecting regions within each layer of an image stack.

    Following instantiation, run "GUI.connect()" to begin event handling.

    Key actions:

        T: remove last added point

        Y: reset all points in layer

        E: exclude entire layer

        W: save ROI data

        Q: disconnect and exit GUI

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

            T: remove last added point

            Y: reset all points in layer

            E: exclude entire layer

            W: save ROI data

            Q: disconnect and exit GUI

        """

        row = self.which_layer(event)

        # save and disconnect
        if event.key == 'w':
            self.save()

        # save and disconnect
        elif event.key == 'q':
            try:
                self.exit()
            except Exception as error:
                self.traceback.append(error)

        # mark for exclusion
        elif event.key == 'e':
            row.include = False
            row.overlay('EXCLUDED')

        # undo
        elif event.key == 't':
            row.undo()

        # clear
        elif event.key == 'y':
            row.clear()
