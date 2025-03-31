"""=================================================================================================
simple kmeans clustering on latent representation

Michael Gribskov     31 March 2025
================================================================================================="""


class Group:
    """=============================================================================================


    ============================================================================================="""
    group_n = 0

    def __init__(self, id=0):
        """-----------------------------------------------------------------------------------------

        -----------------------------------------------------------------------------------------"""
        self.id = id if id else Group.group_n
        Group.group_n += 1
        self.values = []
        self.centroid = []

    def centroid(self):
        """-----------------------------------------------------------------------------------------
        calculate centroid position from values

        :return:
        -----------------------------------------------------------------------------------------"""
        sum = []
        for v in self.values:
            pass

    def distance(self, point):
        """-----------------------------------------------------------------------------------------
        calculate distance of point from centroid

        :param point: list  data vector for a data item
        :return: list       distance to centroid, same dimension as point
        -----------------------------------------------------------------------------------------"""
        d = 0.0
        c = self.centroid
        for i in range(len(c)):
            d += c[i] - point[i]

        return d


# --------------------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    k = 8
    group = []
    for g in range(k):
        group.append(Group())

    # read data
    datafile = 'data/latent.txt'
    datain = open(datafile, 'r')
    data = []
    for line in datain:
        field = line.rstrip().split()
        item = {'id': field[0], 'value': [float(v) for v in field[1:]]}
        data.append(item)

    print(f'values read from {datafile}: {len(data)}')

    # assign initial centroids from data points
    step = int(len(data) / k)
    pos = 0
    for g in group:
        g.centroid = data[pos]['value']
        pos += step

    error = 1.0
    delta = 0.0
    while error > delta:
        # assign points to centroids
        # recalculate centroids
        # calculate error and delta

    exit(0)
