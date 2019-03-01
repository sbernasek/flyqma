from os.path import join, exists
from time import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..utilities import Pickler
from ..annotation import Annotation

from .training import Training
from .simulation import SimulationBenchmark


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
                 attribute='clonal_marker',
                 logratio=True,
                 train_globally=True,
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

            train_globally (bool) - if True, train annotator on entire batch

            training_kw (dict) - keyword arguments for annotator training

            testing_kw (dict) - keyword arguments for annotator application

        """
        self.batch = batch
        self.ambiguity = ambiguity
        self.num_replicates = num_replicates
        self.attribute = attribute
        self.logratio = logratio
        self.train_globally = train_globally

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

    def save(self, dirpath, data=False):
        """
        Save serialized batch instance to <dirpath>.

        Args:

            dirpath (str) - destination of serialized object

            data (bool) - if True, include measurement data (training data)

        """
        if not data:
            self.data = None
            self.graphs = None

        if self.annotator is not None:
            self.annotator.save(dirpath, data=data)
            self.annotator = None

        super().save(join(dirpath, 'batch.pkl'))

    @staticmethod
    def load(dirpath):
        """ Load batch from <dirpath>. """

        batch = Pickler.load(join(dirpath, 'batch.pkl'))

        # load annotator
        if exists(join(dirpath, 'annotation.json')):
            annotator = Annotation.load(dirpath)
            batch.annotator = annotator

        return batch

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
        kw = dict(weighted_by=self.attribute, logratio=self.logratio)
        return {_id: self.build_graph(r, **kw) for _id, r in self.replicates}

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
            bmark = SimulationBenchmark(replicate.copy(),
                                            graph=self.graphs[replicate_id],
                                            **self.params)

            # store results
            results[replicate_id] = dict(

                labels_MAE=bmark.scores['labels'].MAE,
                level_only_MAE=bmark.scores['level_only'].MAE,
                spatial_only_MAE=bmark.scores['spatial_only'].MAE,
                community_MAE=bmark.scores['labels_comm'].MAE,

                labels_PCT=bmark.scores['labels'].percent_correct,
                level_only_PCT=bmark.scores['level_only'].percent_correct,
                spatial_only_PCT=bmark.scores['spatial_only'].percent_correct,
                community_PCT=bmark.scores['labels_comm'].percent_correct)

        # compile dataframe
        results = pd.DataFrame.from_dict(results, orient='index')
        results.index.set_names(self.multiindex, inplace=True)

        return results

    def run(self, train=True):
        """
        Run benchmark on batch.

        Args:

            train (bool) - if True, train global classifier

        """

        # generate synthetic measurements
        start = time()
        self.data = self.measure()

        # build graphs
        self.graphs = self.build_graphs()

        # train annotation object
        if train and self.train_globally:
            self.annotator = self.train(*list(self.graphs.values()),
                                        attribute=self.attribute,
                                        **self.training_kw)

        elif not self.train_globally:
            self.annotator = None

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
