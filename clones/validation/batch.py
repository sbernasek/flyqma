from time import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .training import Training
from .simulation import SimulationBenchmark
from .io import Pickler


class BatchBenchmark(Pickler, Training):
    """
    Class for benchmarking a batch of simulations.

    Attributes:

        batch (growth.sweep.batch.Batch) - batch of simulations

        ambiguity (float) - fluorescence ambiguity

        num_replicates (int) - number of fluorescence replicates

        data (pd.DataFrame) - synthetic measurement data

        attribute (str) - attribute on which cell measurements are classified

        graphs (dict) - graph objects for each replicate, keyed by replicate_id

        annotator (Annotation) - object that assigns labels to measurements

        results (pd.DataFrame) - classification scores for each replicate

        runtime (float) - total benchmark evaluation runtime

    """

    def __init__(self, batch,
                 ambiguity=0.1,
                 num_replicates=1,
                 attribute='fluorescence',
                 logratio=False,
                 training_kw={},
                 testing_kw={}):
        """
        Instantiate batch benchmark.

        Args:

            batch (growth.sweep.batch.Batch) - batch of simulations

            ambiguity (float) - fluorescence ambiguity

            num_replicates (int) - number of fluorescence replicates

            attribute (str) - attribute on which measurements are classified

            logratio (bool) - if True, weight graph edges by log-ratio of attribute level. otherwise, use the absolute difference

            training_kw (dict) - keyword arguments for annotator training

            testing_kw (dict) - keyword arguments for annotator application

        """
        self.batch = batch
        self.ambiguity = ambiguity
        self.num_replicates = num_replicates
        self.attribute = attribute
        self.logratio = logratio

        self.training_kw = training_kw
        self.testing_kw = testing_kw

        # initialize attributes
        self.data = None
        self.annotator = None
        self.results = None
        self.runtime = None

    def __getitem__(self, replicate_id):
        """ Returns SimulationBenchmark for <replicate_id>. """
        measurements = self.data.iloc[self.replicates.indices[replicate_id], :]
        return SimulationBenchmark(measurements.copy(), **self.params)

    def save(self, filepath, save_measurements=False):
        """
        Save serialized instance.

        Args:

            filepath (str) - destination of serialized object

            save_measurements (bool) - if True, include measurements

        """
        if not save_measurements:
            self.data = None
        super().save(filepath)

    @property
    def params(self):
        """ Parameters for SimulationBenchmark. """
        return dict(annotator=self.annotator,
                    attribute=self.attribute,
                    logratio=self.logratio,
                    training_kw=self.training_kw,
                    testing_kw=self.testing_kw)

    @property
    def multiindex(self):
        """ Multilevel index for replicates. """
        return ['growth_replicate', 'fluorescence_replicate']

    @property
    def replicates(self):
        """ Replicates iterator (pd.GroupBy) """
        return self.data.groupby(self.multiindex)

    def build_graphs(self):
        """ Build graph objects for each replicate. """
        kw = dict(attribute=self.attribute, logratio=self.logratio)
        return {_id: self.build_graph(df, **kw) for _id, df in self.replicates}

    def measure(self):
        """ Returns dataframe of synthetic measurements. """
        return self.batch.measure(self.ambiguity, self.num_replicates)

    def evaluate_benchmarks(self):
        """
        Evaluate benchmark for each replicate.
        """

        # iterate over replicates
        results = {}
        for replicate_id, replicate in self.replicates:

            # evaluate benchmark for current replicate
            benchmark = SimulationBenchmark(replicate.copy(),
                                            graph=self.graphs[replicate_id],
                                            **self.params)

            # store results
            results[replicate_id] = dict(labels=benchmark.MAE,
                                         levels_only=benchmark.MAE_levels,
                                         spatial_only=benchmark.MAE_spatial)

        # compile dataframe
        results = pd.DataFrame.from_dict(results, orient='index')
        results.index.set_names(self.multiindex, inplace=True)

        return results

    def run(self):
        """ Run benchmark on batch. """

        # generate synthetic measurements
        start = time()
        self.data = self.measure()

        # build graphs
        self.graphs = self.build_graphs()

        # train annotation object
        self.annotator = self.train(*list(graphs.values()),
                                    attribute=self.attribute,
                                    **self.training_kw)

        # evaluate benchmarks
        self.results = self.evaluate_benchmarks()
        self.runtime = time() - start

    def benchmark_simulation(self, replicate_id=0):
        """
        Returns SimulationBenchmark for <replicate_id>.

        This is separate from the large batch of measurement data generated by the run method because the annotator is only trained on an individual replicate.

        """
        sim = self.batch[replicate_id]
        measurements = sim.measure(ambiguity=self.ambiguity)
        return SimulationBenchmark(measurements, **self.params)
