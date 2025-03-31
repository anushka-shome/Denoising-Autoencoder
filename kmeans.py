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
        item = {'id': field[0], 'value': [float(v) for v in field[1:]]}
        data.append(item)

    print(f'values read from {datafile}: {len(data)}')

    # assign initial centroids from data points
    step = int(len(data) / k)
    pos = 0
    for g in group:
        g.centroid = data[pos]['value']
        pos += step

    error_old = 10000000000000
    delta = 1.0
    cycle = 0
    while delta > 0.001:
        cycle += 1
        print(f'\ncycle {cycle}')
        error = 0

        # assign points to centroids
        for d in data:
            mindist = 10000
            mingroup = group[0]
            for g in group:
                dist = g.distance(d)
                if dist < mindist:
                    mindist = dist
                    mingroup = g
            mingroup.values.append(d)
            error += mindist

        delta = abs(error - error_old)
        error_old = error
        print(f'cycle {cycle} error: {error:.1f}\t{delta:.8g}')
        if delta < 0.001:
            break
        for g in group:
            # recalculate centroids
            centroid = g.get_centroid()
            print(f'{g.id}\t{len(g.values)}\t{centroid}')
            g.values = []


    exit(0)
