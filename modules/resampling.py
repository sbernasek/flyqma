def resample(cells, size=None, cutoff=None):
    """ Resample uniformly in X. """

    # sort values
    x = cells.sort_values('centroid_x')['centroid_x']

    if size is None:
        size = len(x)

    # apply threshold on upper bound
    if cutoff is not None:
        threshold = np.percentile(x.values, cutoff)
    else:
        threshold = x.max()+1

    # get unique x values
    xunique = np.unique(x.values)

    # filter points below threshold
    xx = x[x<=threshold]

    # get probabilities
    x_to_count = np.vectorize(Counter(xx).get)

    # get intervals
    intervals = np.diff(xunique)
    unique_below_threshold = (xunique[:-1]<=threshold)
    intervals = intervals[unique_below_threshold]

    # assign probabilities
    x_to_cumul = np.vectorize(dict(zip(xunique[:-1][unique_below_threshold], intervals/intervals.sum())).get)

    p = x_to_cumul(xx)/x_to_count(xx)

    p[np.isnan(p)] = 0

    # generate sample
    sample_ind = np.random.choice(xx.index, size=size, p=p)
    #xu, yu = xx[sample_ind], yy[sample_ind]

    return cells.loc[sample_ind]

def resample_df(df, size=None):
    """ Resample dataframe uniformly in X """

    # resample all
    resampled = []
    for exp_id in df.experiment.unique():
        exp = df[df.experiment==exp_id]
        for disc_id in exp.disc_id.unique():
            disc = exp[exp.disc_id==disc_id]
            for layer_id in disc.layer.unique():
                layer = disc[disc.layer==layer_id]
                for genotype in (0, 1, 2):
                    cells = layer[layer.genotype==genotype]
                    if len(cells) > 5:
                        resampled_cells = resample(cells, size=size)
                    resampled.append(resampled_cells)
    resampled = pd.concat(resampled)
    return resampled
