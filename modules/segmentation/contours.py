

class Contour:
    """
    Object describes an individual cell measurement.

    Format is designed for agreement with FlyEye Silhouette data.
    """

    def __init__(self, _id, centroid, color_avg, color_std, voxel_count):
        self.id = _id
        self.centroid_x = centroid[0]
        self.centroid_y = centroid[1]
        self.r = color_avg[0]
        self.g = color_avg[1]
        self.b = color_avg[2]
        self.r_std = color_std[0]
        self.g_std = color_std[1]
        self.b_std = color_std[2]
        self.pixel_count = voxel_count

    def to_json(self):
        """ Serialize contour in JSON format. """
        return {
            'id': int(self.id),
            'centroid_x': self.centroid_x,
            'centroid_y': self.centroid_y,

            'r': self.r,
            'g': self.g,
            'b': self.b,

            'r_std': self.r_std,
            'g_std': self.g_std,
            'b_std': self.b_std,

            'pixel_count': int(self.pixel_count)}


class Contours:
    """
    Object describes a collection of cell measurements.

    Format is designed for agreement with FlyEye Silhouette data.
    """

    def __init__(self, _ids, centroids, color_avgs, color_stds, volume):
        self.ids = _ids
        self.centroids = centroids
        self.color_avgs = list(zip(*color_avgs))
        self.color_stds = list(zip(*color_stds))
        self.voxel_counts = volume
        self.size = len(self.ids)

    def get_contour(self, index):
        _id = self.ids[index]
        centroid = self.centroids[index]
        color_avg = self.color_avgs[index]
        color_std = self.color_stds[index]
        voxel_count = self.voxel_counts[index]
        return Contour(_id, centroid, color_avg, color_std, voxel_count)

    def to_json(self):
        return [self.get_contour(idx).to_json() for idx in range(self.size)]
