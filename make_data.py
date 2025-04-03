"""=================================================================================================
Combine the binary form of fingerprints with class, subclass, and file labels,also add length
(number of motifs) for each of the curated samples

Michael Gribskov     03 April 2025
================================================================================================="""
from collections import defaultdict

# --------------------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    metadata = {}
    # read labels
    data = []
    labelfile = 'data/labels.txt'
    labelin = open(labelfile, 'r')
    n = 0
    for line in labelin:
        thisdata = []
        i, id = line.rstrip().split(',')
        thisdata.append(id)

        dotpos = id.find('.')
        linepos = id.find('_')
        class_label = id[:dotpos]
        subclass_label = class_label
        if linepos > -1 and linepos < dotpos:
            class_label = id[:linepos]
        thisdata.append([class_label, subclass_label])

        if class_label in metadata:
             if subclass_label in metadata[class_label]

    labelin.close()
    print(f'datarecords read from {labelfile}: {len(data)}')

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
    exit(0)
