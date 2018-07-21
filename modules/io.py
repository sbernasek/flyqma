import os
import shutil
import json
import tifffile as tf
import numpy as np
import pandas as pd
import time


def safeload(loader):
    def wrapper(self, path):
        out = None
        if os.path.exists(path):
            out = loader(path)
        return out
    return wrapper


class IO:

    @staticmethod
    def make_dir(dir_name, path=None, force=False):
        """ Create Silhouette3D container. """

        # compile directory name
        if path is not None:
            dir_name = os.path.join(path, dir_name)

        # check if directory exists
        if os.path.exists(dir_name):
            if force == True:
                shutil.rmtree(dir_name)
            else:
                print('Will not overwrite existing directory.')
                return dir_name

        # make dir
        os.mkdir(dir_name)

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
        im = tf.imread(path)
        return im

    @staticmethod
    def render_tiff(path, stack):
        """ Render stack as tiff. """
        if stack is not None:
            tf.imsave(path, stack)

    @safeload
    def read_csv(path):
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

