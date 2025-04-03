"""=================================================================================================
simple kmeans clustering on latent representation

Michael Gribskov     31 March 2025
================================================================================================="""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


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

    def get_centroid(self):
        """-----------------------------------------------------------------------------------------
        calculate centroid position from values

        :return:
        -----------------------------------------------------------------------------------------"""
        n = len(self.values)
        if n == 0:
            self.centroid = [0.0 for _ in self.values[0]['value']]
            return self.centroid

        sum = [0.0 for _ in self.values[0]['value']]
        for v in self.values:
            for i in range(len(v['value'])):
                sum[i] += v['value'][i] / n

        self.centroid = sum
        return sum

    def distance(self, point):
        """-----------------------------------------------------------------------------------------
        calculate distance of point from centroid

        :param point: list  data vector for a data item
        :return: list       distance to centroid, same dimension as point
        -----------------------------------------------------------------------------------------"""
        d = 0.0
        c = self.centroid
        for i in range(len(c)):
            d += abs(c[i] - point['value'][i])

        return d


# --------------------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    k = 7
    group = []
    for g in range(k):
        group.append(Group())

    # read data
    datafile = 'data/latent.txt'
    datain = open(datafile, 'r')
    data = []
    for line in datain:
        field = line.rstrip().split()
        item = [float(v) for v in field[1:]]
        data.append(item)

    datamat = np.array(data)
    print(f'values read from {datafile}: {len(data)}')

    kmeans = KMeans(n_clusters=7, init="k-means++", n_init=100, verbose=2)
    kmeans.fit(datamat)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    print(f'{labels}\n{centers}')
    print(f'error: {kmeans.inertia_:.2f}')

    exit(0)
