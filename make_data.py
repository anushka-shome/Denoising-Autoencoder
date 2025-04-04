"""=================================================================================================
Combine the binary form of fingerprints with class, subclass, and file labels,also add length
(number of motifs) for each of the curated samples

Michael Gribskov     03 April 2025
================================================================================================="""
from collections import defaultdict
import pandas as pd

# --------------------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    label_summary = {}
    metadata = []
    # read labels
    data = []
    labelfile = 'data/labels.txt'
    labelin = open(labelfile, 'r')
    n = 0
    for line in labelin:
        thisdata = []
        i, name = line.rstrip().split(',')
        thisdata.append(name)

        dotpos = name.find('.')
        linepos = name.find('_')
        class_label = name[:dotpos]
        subclass_label = class_label
        if linepos > -1 and linepos < dotpos:
            class_label = name[:linepos]
        metadata.append([class_label, subclass_label])

        if class_label in label_summary:
            if subclass_label in label_summary[class_label]:
                label_summary[class_label][subclass_label] += 1
            else:
                label_summary[class_label][subclass_label] = 1
        else:
            label_summary[class_label] = {subclass_label: 1}

    labelin.close()
    print(f'datarecords read from {labelfile}: {len(metadata)}')

    print('\nclass and subclass labels')
    for class_label in label_summary:
        class_n = 0
        for subclass_label in label_summary[class_label]:
            class_n += label_summary[class_label][subclass_label]

        print(f'{class_label:14s}\t\t{class_n:5d}')
        for subclass_label in label_summary[class_label]:
            print(f'\t{subclass_label:10s}\t{label_summary[class_label][subclass_label]:9d}')

    # read data, transpose original file to have motifs as columns
    datafile = 'data/binary_data.csv'
    df = pd.read_csv(datafile, header=None).T
    df['sum'] = df.sum(axis=1)
    df.loc['total'] = df.sum(numeric_only=True, axis=0)
    df = df.loc[:, (df > 4).any(axis=0)]
    print(df)
    df.to_csv('data/binary.min4.csv', header=None)

    exit(0)
