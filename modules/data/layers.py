

class Layer(MultichannelImage):

    def __init__(self, im, labels=None, layer_id=0, stack_path='.'):
        MultichannelImage.__init__(self, im, labels=labels)
        self.channels = dict(r=0, g=1, b=2)
        self.layer_id = int(layer_id)
        self.path = os.path.join(stack_path, '{:d}'.format(self.layer_id))
        get_sel_path = lambda p, d, l: os.path.join(p, 'selections', d, l)
        self.selection_path = get_sel_path(*self.path.rsplit('/', maxsplit=2))
        self.graph = None

    @staticmethod
    def load(im, path):
        """ Instantiate layer with saved contour labels. """

        # load labels and metadata
        io = IO()
        labels = io.read_npy(os.path.join(path, 'labels.npy'))
        metadata = io.read_json(os.path.join(path, 'layer_metadata.json'))

        # instantiate layer
        layer_id = metadata['layer_id']
        stack_path = path.rsplit('/', maxsplit=1)[0]
        layer = Layer(im, labels=labels, layer_id=layer_id, stack_path=stack_path)

        # assign background
        bg = metadata['bg']
        layer.bg = bg

        # assign data
        df = pd.read_json(io.read_json(os.path.join(path, 'contours.json')))
        layer.df = df

        # load metadata
        md_path = os.path.join(os.path.join(path, os.pardir), 'metadata.json')
        params = io.read_json(md_path)['params']

        # set selection
        md = io.read_json(os.path.join(layer.selection_path, 'md.json'))
        layer.include = bool(md['include'])

        # build graph
        layer.build_graph(**params['graph_kw'])

        # annotate
        layer.cell_classifier = CellClassifier.load(stack_path)
        #layer.annotate(community_genotype=False)

        return layer

    def segment(self, bg='b', **kwargs):
        """ Run watershed segmentation on specified background channel. """
        self.bg = bg
        background = self.get_channel(bg)
        background.segment(**kwargs)
        self.labels = background.labels

    def measure(self):
        """ Measure fluorescence intensities to generate contours. """

        # get image channels
        drop_axis = lambda x: x.reshape(*x.shape[:2])
        r, g, b = [drop_axis(x) for x in np.split(self.im, 3, axis=-1)]

        # get segment ids (ordered)
        segment_ids = np.unique(self.labels[self.labels.nonzero()])

        # get centroids
        centroid_dict = Segmentation.evaluate_centroids(self.labels)
        centroids = [centroid_dict[seg_id] for seg_id in segment_ids]

        # compute means
        rmeans = mean(r, self.labels, segment_ids)
        gmeans = mean(g, self.labels, segment_ids)
        bmeans = mean(b, self.labels, segment_ids)
        color_avg = (rmeans, gmeans, bmeans)

        # compute std
        rstd = standard_deviation(r, self.labels, segment_ids)
        gstd = standard_deviation(g, self.labels, segment_ids)
        bstd = standard_deviation(b, self.labels, segment_ids)
        color_std = (rstd, gstd, bstd)

        # compute segment size
        voxels = self.labels[self.labels!=0]
        bins = np.arange(0, segment_ids.max()+3, 1)
        counts, _ = np.histogram(voxels, bins=bins)
        voxel_counts = counts[segment_ids]

        # createlist of contour dicts (useless but fits with Silhouette)
        data = (segment_ids, centroids, color_avg, color_std, voxel_counts)
        contours = Contours(*data).to_json()

        return contours

    def compile_dataframe(self):
        """ Compile dataframe from cell measurements. """
        columns = ['id', 'centroid_x', 'centroid_y',
                   'r', 'r_std', 'g', 'g_std', 'b', 'b_std', 'pixel_count']
        contours = self.measure()
        df = pd.DataFrame.from_records(contours, columns=columns)
        df['layer'] = self.layer_id

        # normalize by background
        for channel in 'rgb'.strip(self.bg):
            df[channel+'_normalized'] = df[channel] / df[self.bg]

        self.df = df

    def set_foreground(self, bandwidth=100, n=2):
        """ Threshold cell position density to set foreground. """
        kde = KDE(self.df, bandwidth=bandwidth, n=n)
        self.df['foreground'] = kde.mask
        self.fg_mask = ForegroundMask(self.shape, kde)

    def build_graph(self, q=95, weighted_by='r_normalized'):
        """
        Compile weighted graph connecting adjacent cells.

        Args:
        q (float) - edge length quantile above which edges are pruned
        weighted_by (str) - quantity used to weight edges
        """
        self.graph = WeightedGraph(self.df, q=q, weighted_by=weighted_by)

    def annotate(self, community_genotype=False):
        """ Annotate layer. """

        # assign single-cell classifier label
        self.df['genotype'] = self.cell_classifier(self.df)

        # assign cluster labels
        if community_genotype:
            annotation = Annotation(self.graph, self.cell_classifier)
            self.df['community_genotype'] = annotation(self.df)

    def mark_boundaries(self, basis='genotype', max_edges=0):
        """ Assign boundary label to cells with an edge to another clone. """

        # assign genotype to edges
        assign_genotype = np.vectorize(dict(self.df[basis]).get)
        edge_genotypes = assign_genotype(self.graph.edges)

        # find edges traversing clones
        boundaries = (edge_genotypes[:, 0] != edge_genotypes[:, 1])

        # get number of clone-traversing edges per node
        boundary_edges = self.graph.edges[boundaries]
        edge_counts = Counter(boundary_edges.ravel())

        # assign boundary label to nodes with too many clone-traversing edges
        boundary_nodes = [n for n, c in edge_counts.items() if c>max_edges]
        self.df['boundary'] = False
        self.df.loc[boundary_nodes, 'boundary'] = True

    def plot_graph(self, channel='r', figsize=(15, 15), cmap=None, **kw):
        """ Plot graph on top of relevant channel. """

        # create axis
        fig, ax = plt.subplots(figsize=figsize)

        # add image
        _ = self.get_channel(channel).show(ax=ax, segments=False, cmap=cmap)

        # add graph
        self.annotation.graph.show(ax=ax, **kw)

        return fig

    def plot_annotation(self, cmap='grey', fig_kw={}, clone_kw={}):
        """ Show annotation channel overlayed with clone segments. """

        if cmap == 'grey':
            cmap = plt.cm.Greys

        # get layer
        weighted_by = self.annotation.graph.weighted_by
        if 'normalized' in self.annotation.graph.weighted_by:
            im = self.get_channel(weighted_by.split('_')[0])
        else:
            im = self.get_channel(weighted_by)

        # show layer
        fig = im.show(segments=False, cmap=cmap, **fig_kw)

        # add clones
        self.annotation.plot_clones(fig.axes[0], **clone_kw)

        # mask background
        if self.annotation.fg_only:
            self.fg_mask.add_contourf(fig.axes[0], alpha=0.5)

        return fig

    def save_contours(self):
        io = IO()
        contours = self.df.to_json()
        io.write_json(os.path.join(self.path, 'contours.json'), contours)

    def save(self,
             segmentation=True,
             image=True,
             foreground=False,
             annotation=False,
             dpi=100):
        """ Save segmentation parameters and results. """

        # instantiate IO
        io = IO()

        # create directory
        dirpath = io.make_dir(self.path, force=True)

        # save contours
        self.save_contours()
        #contours = self.df.to_json()
        #io.write_json(os.path.join(dirpath, 'contours.json'), contours)

        # save metadata
        metadata = dict(layer_id=self.layer_id, bg=self.bg)
        io.write_json(os.path.join(dirpath, 'layer_metadata.json'), metadata)

        # save segmentation data
        if segmentation:
            labels_path = os.path.join(dirpath, 'labels.npy')
            np.save(labels_path, self.labels)

        # set image keyword arguments
        im_kw = dict(format='png', dpi=dpi, bbox_inches='tight', pad_inches=0)

        # save segmentation image
        if image:
            im_path = os.path.join(dirpath, 'segmentation.png')
            bg = self.get_channel(self.bg, copy=False)
            fig = bg.show(segments=True)
            fig.axes[0].axis('off')
            fig.savefig(im_path, **im_kw)
            fig.clf()
            plt.close(fig)
            gc.collect()

        # save foreground image
        if foreground:
            im_path = os.path.join(dirpath, 'foreground.png')
            bg = self.get_channel(self.bg, copy=False)
            fig = bg.show(segments=False)
            self.fg_mask.add_contourf(fig.axes[0], alpha=0.5)
            fig.savefig(im_path, **im_kw)
            fig.clf()
            plt.close(fig)
            gc.collect()

        # save annotation image
        if annotation:
            im_path = os.path.join(dirpath, 'annotation.png')
            fig = self.plot_annotation()
            fig.savefig(im_path, **im_kw)
            fig.clf()
            plt.close(fig)
            gc.collect()

        return dirpath
