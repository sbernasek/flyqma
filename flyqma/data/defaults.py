import numpy as np


class Defaults:
    """
    Object contains default parameters. Defaults are stored as dictionaries within the instance attributes:

    Attributes:

        parameters (dict) - {parameter set: parameters dict} pairs for:

        preprocessing (dict) - default parameters for image preprocessing

        seeds (dict) - default parameters for seed detection

        segmentation (dict) - default parameters for segmentation

    """

    def __init__(self):
        """
        Instantiate object with default parameter dictionaries as attributes
        """

        self.parameters = {

            'preprocessing':
                {
                'median_radius': 2,
                'gaussian_sigma': (2, 2),
                'clip_limit': 0.03,
                'clip_factor': 20
                },

            'seeds':
                {
                'sigma': 2,
                'min_distance': 1,
                'num_peaks': np.inf
                },

            'segmentation':
                {
                'sigma': 0.5,
                'watershed_line': True
                }}

    def __call__(self, key, specified):
        """
        Adds default values for unspecified <key> parameters.

        Args:

            key (str) - name of parameter set

            specified (dict) - {name: value} pairs for specified <key> parameters

        Returns:

            parameters (dict) - {name: value} pairs for all <key> parameters

        """
        return self.append_defaults(key, specified)

    def append_defaults(self, key, specified):
        """
        Add default values for unspecified <key> parameters.

        Args:

            key (str) - name of parameter set

            specified (dict) - {name: value} pairs for specified <key> parameters

        Returns:

            parameters (dict) - {name: value} pairs for all <key> parameters

        """
        assert key in self.parameters.keys(), 'Parameter set not recognized.'
        default = self.parameters[key]
        return self._append_defaults(specified, default)

    @staticmethod
    def _append_defaults(specified, default):
        """
        Add default values for unspecified parameters.

        Args:

            specified (dict) - {name: value} pairs for specified parameters

            default (dict) - {name: default value} pairs for all parameters

        Returns:

            parameters (dict) - {name: value} pairs for all parameters

        """
        parameters = {}
        for k, v in default.items():
            if k in specified.keys():
                parameters[k] = specified[k]
            else:
                parameters[k] = v
        return parameters
