# Default silhouette3d parameters.

preprocessing_kw = {'median_radius': 1,
                    'sigma': 0.5,
                    'clip_limit': 0.1,
                    'foreground_threshold': 0.65}

voter_kw = {'diameter': 4,
            'clip_limit': None,
            'h': 0.05,
            'dilation': 1}

seed_kw = {'mode': 'voter', 'voter_kw': voter_kw}

seg_kw = {'dilation': 0,
          'mode': 'image',
          'kwargs': {}}

split_kw = {}

convergence_kw = {'min_delta_v': 0.98,
                  'min_compactness': None,
                  'max_depth': 10}

min_volume = 75

default_parameters = dict(preprocessing_kw=preprocessing_kw,
                     seed_kw=seed_kw,
                     seg_kw=seg_kw,
                     split_kw=split_kw,
                     convergence_kw=convergence_kw,
                     min_volume=min_volume)
