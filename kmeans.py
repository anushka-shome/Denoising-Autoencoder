"""=================================================================================================
simple kmeans clustering on latent representation

Michael Gribskov     31 March 2025
================================================================================================="""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.datasets import make_blobs

def plot(X, labels, probabilities=None, parameters=None, ground_truth=False, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    labels = labels if labels is not None else np.ones(X.shape[0])
    probabilities = probabilities if probabilities is not None else np.ones(X.shape[0])
    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    # The probability of a point belonging to its labeled cluster determines
    # the size of its marker
    proba_map = {idx: probabilities[idx] for idx in range(len(labels))}
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_index = np.where(labels == k)[0]
        for ci in class_index:
            ax.plot(
                X[ci, 0],
                X[ci, 1],
                "x" if k == -1 else "o",
                markerfacecolor=tuple(col),
                markeredgecolor="k",
                markersize=4 if k == -1 else 1 + 5 * proba_map[ci],
            )
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    preamble = "True" if ground_truth else "Estimated"
    title = f"{preamble} number of clusters: {n_clusters_}"
    if parameters is not None:
        parameters_str = ", ".join(f"{k}={v}" for k, v in parameters.items())
        title += f" | {parameters_str}"
    ax.set_title(title)
    plt.tight_layout()

# --------------------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    k = 10

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
    datafile = 'data/latent.out'
    datain = open(datafile, 'r')
    data = []
    for line in datain:
        field = line.rstrip().split()
        item = [float(v) for v in field[1:]]
        data.append(item)

    datain.close()
    datamat = np.array(data)
    print(f'values read from {datafile}: {len(data)}')

    # HDBScan
    # hdb = HDBSCAN(min_cluster_size=8, min_samples=7).fit(datamat)
    # labels = hdb.labels_
    # hdb.fit(datamat)
    # plot(
    #     datamat,
    #     hdb.labels_,
    #     hdb.probabilities_)

    # kmeans

    kmeans = KMeans(n_clusters=k, init="k-means++", n_init=100, verbose=2)
    kmeans.fit(datamat)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    print(f'{labels}\n{centers}')
    print(f'error: {kmeans.inertia_:.2f}')

    print('labels')
    for g in origlabel:
        print(f'{origlabel[g]}\t{g}')

    # Plot the data points and cluster centers
    dim = [0,1,2]
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    scatter = ax.scatter(datamat[:, dim[0]], datamat[:, dim[1]], datamat[:, dim[2]], c=rownames, cmap='Dark2', s=20,
                         alpha=0.7)
    ax.scatter(centers[:, dim[0]], centers[:, dim[1]], centers[:, dim[2]], marker='x', color='black', s=50)
    plt.title('K-Means Clustering')
    ax.set_xlabel(f'Feature {dim[0]}')
    ax.set_ylabel(f'Feature {dim[1]}')
    ax.set_zlabel(f'Feature {dim[2]}')

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
