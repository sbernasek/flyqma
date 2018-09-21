from os.path import join, exists
from os import mkdir
import shutil
import json
import numpy as np


def safeload(loader):
    """ Decorator for checking files exist before attempting to load them. """
    def wrapper(self, path):
        out = None
        if exists(path):
            out = loader(path)
        return out
    return wrapper


class IO:
    """
    Class provides assorted methods for reading/writing different filetypes.
    """

    @staticmethod
    def make_dir(dir_name, path=None, force=False):
        """ Create Silhouette3D container. """

        # compile directory name
        if path is not None:
            dir_name = join(path, dir_name)

        # check if directory exists
        if exists(dir_name):
            if force == True:
                shutil.rmtree(dir_name, ignore_errors=True)
            else:
                print('Will not overwrite existing directory.')
                return dir_name

        # make dir
        mkdir(dir_name)

        return dir_name

    @safeload
    def read_json(path):
        """ Read data from json. """
        with open(path, 'r') as f:
             data = json.load(f)
        return data

    @staticmethod
    def write_json(path, data):
        """ Write data to json. """
        with open(path, 'w') as f:
             json.dump(data, f)

    @safeload
    def read_tiff(path):
        """ Read stack from tiff """
        import tifffile as tf
        im = tf.imread(path)
        return im

    @staticmethod
    def render_tiff(path, stack):
        """ Render stack as tiff. """
        if stack is not None:
            tf.imsave(path, stack)

    @safeload
    def read_csv(path):
        import pandas as pd
        return pd.read_csv(path, index_col=0)

    @staticmethod
    def write_csv(path, df):
        if df is not None:
            df.to_csv(path)

    @safeload
    def read_npy(path):
        return np.load(path)

    @staticmethod
    def write_npy(path, arr):
        if arr is not None:
            np.save(path, arr)

