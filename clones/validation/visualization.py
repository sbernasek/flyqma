import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize


class SweepVisualization:
    """ Methods for visualizing fluorescence values. """

    @property
    def max_fluroescence(self):
        """ Peak fluorescence across sweep. """
        batch = self.load_benchmark(0, self.num_ambiguities-1)
        return np.percentile(batch.classifier.values, 99)

    @property
    def fnorm(self):
        """ Fluorescence normalization across sweep. """
        return Normalize(vmin=0, vmax=self.max_fluroescence)

    def plot_fluorescence(self,
                          replicate_id=0,
                          resolution=4,
                          norm=None,
                          cmap=plt.cm.viridis,
                          figsize=(10, 10),
                          **kwargs):

        # determine shape
        shape = self.ambiguities.size, self.batches.shape[1]
        xs = range(0, shape[1], resolution)
        ys = range(0, shape[0], resolution)
        nrows, ncols = len(ys), len(xs)

        # determine normalization
        if norm == 'global':
            norm = self.fnorm

        # create figure
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(nrows, ncols, wspace=0, hspace=0)

        # plot fluorescence values
        for i, ambiguity_id in enumerate(xs):
            for j, batch_id in enumerate(ys[::-1]):

                # add axis
                ax = fig.add_subplot(gs[i, j])

                # load BatchBenchmark then generate and plot measurement data
                batch = self.load_benchmark(batch_id, ambiguity_id)
                simulation = batch.benchmark_simulation(replicate_id)
                simulation.plot_measurements(ax=ax, norm=norm, **kwargs)

                # format axis
                ax.set_xlim(-1.1, 1.1)
                ax.set_ylim(-1.1, 1.1)
                ax.axis('off')

        return fig
