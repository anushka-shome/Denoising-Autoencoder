"""=================================================================================================
simple kmeans clustering on latent representation

Michael Gribskov     31 March 2025
================================================================================================="""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# --------------------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    k = 6

    # read labels
    rownames = []
    origlabel = {}
    labelfile = 'data/labels.txt'
    label = open(labelfile, 'r')
    n = 0
    for line in label:
        i, id = line.rstrip().split(',')
        cut = id.find('.')
        linepos = id.find('_')
        if linepos > -1 and linepos < cut:
            cut = linepos
        name = id[:cut]

        if name not in origlabel:
            origlabel[name] = n
            n += 1
        rownames.append(origlabel[name])

    label2name = []
    for gname in origlabel:
        label2name.append(gname)

    label.close()

    # read data
    datafile = 'data/latent2.txt'
    datain = open(datafile, 'r')
    data = []
    for line in datain:
        field = line.rstrip().split()
        item = [float(v) for v in field[1:]]
        data.append(item)

    datain.close()
    datamat = np.array(data)
    print(f'values read from {datafile}: {len(data)}')

    kmeans = KMeans(n_clusters=7, init="k-means++", n_init=100, verbose=2)
    kmeans.fit(datamat)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    print(f'{labels}\n{centers}')
    print(f'error: {kmeans.inertia_:.2f}')

    print('labels')
    for g in origlabel:
        print(f'{origlabel[g]}\t{g}')

    # Plot the data points and cluster centers
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    scatter = ax.scatter(datamat[:, 1], datamat[:, 2], datamat[:, 3], c=rownames, cmap='Dark2')
    ax.scatter(centers[:, 1], centers[:, 2], centers[:, 3], marker='x', color='black', s=50)
    plt.title('K-Means Clustering')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Feature 3')

    # handles, labels = ax.get_legend_handles_labels()
    s = scatter.legend_elements()
    legend1 = ax.legend(*scatter.legend_elements(), loc="best", title="Groups")
    ax.add_artist(legend1)
    # ax.legend(labels=label2name)
    tt = legend1.get_texts()
    t = 0
    for lab in label2name:
        legend1.get_texts()[t].set_text(lab)
        t += 1

    plt.show(block=True)

    exit(0)
