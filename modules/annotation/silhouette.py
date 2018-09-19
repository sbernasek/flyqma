import os

from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from ..utilities.io import IO
from .classifiers import CellClassifier

from modules.figure_settings import *



class Silhouette:
    """ Class manages individual silhouette file. """

    def __init__(self, silhouette_path):
        self.path = silhouette_path
        self.name = silhouette_path.rsplit('/', maxsplit=1)[1].split('_')[0]
        feud = self.get_feud(silhouette_path)
        self.annotations = {l['id']: l['contours'] for l in feud['layers']}
        self.annotated_layers = list(self.annotations.keys())
        self.nlabels = {k:len(v) for k, v in self.annotations.items()}

        # set layer paths
        layer_paths = glob(os.path.join(silhouette_path, '*[0-9].json'))
        self.layer_paths = {int(p.rsplit('/', maxsplit=1)[1].strip('.json')): p for p in layer_paths}
        self.df = self.get_stack_df()

    @staticmethod
    def get_feud(path):
        """ Load feud file. """
        io = IO()
        feud = io.read_json(os.path.join(path, 'feud.json'))
        return feud

    @staticmethod
    def parse_contours(contours):
        """ Unpack contours. """
        df = pd.DataFrame(contours)
        df[['centroid_x', 'centroid_y']] = pd.DataFrame(df.centroid.tolist(), columns=['centroid_x', 'centroid_y'])
        df.drop('centroid', axis=1, inplace=True)
        df.drop('color_std', axis=1, inplace=True)
        get_rgb = lambda x: (x['r'], x['g'], x['b'])
        df[['r', 'g', 'b']] = pd.DataFrame([get_rgb(x) for x in df.color_avg.tolist()], columns=['r', 'g', 'b'])
        df.drop('color_avg', axis=1, inplace=True)
        df['r_normalized'] = df.r/df.b
        df['g_normalized'] = df.g/df.b
        return df

    def get_layer_df(self, layer_id):
        """ Get dataframe of all segmented contours within a single layer. """
        io = IO()
        contours = io.read_json(self.layer_paths[layer_id])['contours']
        df = self.parse_contours(contours)

        # match annotations with corresponding segments
        annotations = pd.DataFrame(self.annotations[layer_id])
        df = pd.merge(df, annotations, on='id', how='outer')
        df.loc[df[df.label=='r8'].index, 'label'] = np.nan

        return df

    def get_stack_df(self):
        """ Get dataframe of all segmented contours across all layers. """
        dfs = []
        for layer_id in self.annotated_layers:
            df = self.get_layer_df(layer_id)
            df['layer'] = layer_id
            dfs.append(df)
        return pd.concat(dfs)

    def build_classifier(self, groups=None):
        """ Compile cell classifier using entire imagestack. """
        if groups is None:
            groups = {0: 0, 1: 1, 2: 2}
        cell_classifier = CellClassifier.from_cells(self.df, groups=groups, classify_on='r', log=False, n=len(groups))
        self.classifier = cell_classifier

    def apply_classifier(self):
        """ Assign genotypes for individual layer. """
        self.df['genotype'] = self.classifier(self.df)

    def annotate(self, groups=None):
        self.build_classifier(groups)
        self.apply_classifier()

        # convert human annotations to (0, 1, 2) values
        ind = (~self.df.label.isna())
        key = dict(m=0, h=1, w=2)
        get_label = lambda x: key[x[0]]
        self.df['labeled'] = np.nan
        self.df.loc[ind, 'labeled'] = self.df[ind].label.apply(get_label)

    def get_scoring(self):
        """ Get scoring for subset of annotated segments. """
        df = self.df[~self.df.label.isna()]
        return Scoring(df[['labeled', 'genotype']].values)



class Scoring:

    def __init__(self, records):
        self.df = pd.DataFrame(records, columns=('measured', 'predicted'))
        self.n = len(self.df)

    def __add__(self, x):
        merged = pd.concat([self.df, x.df])
        return Scoring(merged[['measured', 'predicted']])

    def score(self, **kw):
        self.compare()
        self.plot_matrix(**kw)

    def compare(self):
        self.df['difference'] = abs(self.df.measured-self.df.predicted)
        self.df['correct'] = (self.df.difference==0)
        self.percent_correct = self.df.correct.sum() / self.n

    def plot_matrix(self, **kw):
        measured = self.df.measured.values.astype(int)
        predicted = self.df.predicted.values.astype(int)
        self.matrix = ErrorMatrix(measured, predicted, **kw)


class ErrorMatrix:

    def __init__(self, measured, predicted, text=None, figsize=(2, 2), **kw):
        self.counts = self.build_matrix(measured, predicted)
        ax = self.create_figure(figsize=figsize)
        self.plot_matrix(ax, text=text, **kw)

    def save(self, name='classifier_scores', directory='../graphics', fmt='pdf', dpi=300, transparent=True, rasterized=True):
        path = os.path.join(directory, name+'.{}'.format(fmt))
        kw = dict(dpi=dpi, transparent=transparent, rasterized=rasterized)
        self.fig.savefig(path, format=fmt, **kw)

    @staticmethod
    def build_matrix(measured, predicted):
        """ Compute 2D histogram """
        counts, _, _ = np.histogram2d(measured, predicted, bins=np.arange(3.5))
        return counts.astype(np.int64)

    def create_figure(self, figsize=(3, 3)):
        self.fig = plt.figure(figsize=figsize)
        ax = self.fig.subplots()
        return ax

    def plot_matrix(self, ax, text=None, fontsize=7):
        """ Plot classification matrix. """

        # compute error rates
        rates = self.counts.astype(np.float64)
        rates /= self.counts.sum(axis=1).reshape(-1, 1)

        # plot image
        ax.imshow(rates.T, cmap=plt.cm.Reds, vmin=0, vmax=1)
        ax.invert_yaxis()

        # add labels
        kw = dict(horizontalalignment='center', verticalalignment='center')
        for i in range(3):
            for j in range(3):
                if text=='rates':
                    s = '{:0.1%}'.format(rates[i, j])
                    ax.text(i, j, s=s, fontsize=fontsize, **kw)
                elif text=='counts':
                    s = '{:d}'.format(self.counts[i, j])
                    ax.text(i, j, s=s, fontsize=fontsize, **kw)
                else:
                    continue

        # format axes
        self.format(ax, fontsize=fontsize)

    @staticmethod
    def format(ax, fontsize=7):
        ax.set_xlabel('Human label', fontsize=fontsize+1)
        ax.set_ylabel('Automated label', fontsize=fontsize+1)
        ax.set_xticks(np.arange(2.5))
        ax.set_xticklabels(['0x', '1x', '2x'], fontsize=fontsize)
        ax.set_yticks(np.arange(2.5))
        ax.set_yticklabels(['0x', '1x', '2x'], fontsize=fontsize)
        ax.set_aspect(1)
        #ax.set_title('Error: {:0.2%}'.format(error), fontsize=12)
