# 一次计算所有数据集数据
from Clusters import cls
from PToL import pTol
from Propagation import LablePropagation
from Evaluations import calEval
import scipy.io as scio
def main(dataset='movie', evaluation='.', cluster='AgglomerativeClustering', n_clusters=1, ctrl=True):
    datafile = ['twitteru', 'flickru', 'movie', 'SBU-3DFE', 'SJAFFE', 'Yeast-alpha', 'Yeast-cdc', 'Yeast-cold', 'Yeast-diau', 'Yeast-dtt',
                'Yeast-elu', 'Yeast-heat', 'Yeast-spo', 'Yeast-spo5', 'Yeast-spoem']

    # Alpha = [0.69,0.89,0.89,0.89,0.85,0.92,0.89,0.88,0.83,0.81,0.82,0.59,0.69]
    # Alpha = [0.60, 0.70, 0.70, 0.89, 0.89, 0.89, 0.85, 0.92, 0.89, 0.88, 0.83, 0.81, 0.82]
    Alpha = {'twitteru':0.30, 'flickru':0.30,'movie':0.60, 'SBU-3DFE':0.70, 'SJAFFE':0.70, 'Yeast-alpha':0.89, 'Yeast-cdc':0.89, 'Yeast-cold':0.89, 'Yeast-diau':0.85, 'Yeast-dtt':0.92,
                'Yeast-elu':0.89, 'Yeast-heat':0.88, 'Yeast-spo':0.83, 'Yeast-spo5':0.81, 'Yeast-spoem':0.82}

    # K = [66, 13, 5, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24]
    K = {'twitteru':65, 'flickru':60,'movie':66, 'SBU-3DFE':13, 'SJAFFE':5, 'Yeast-alpha':24, 'Yeast-cdc':24, 'Yeast-cold':24, 'Yeast-diau':24, 'Yeast-dtt':24,
                'Yeast-elu':24, 'Yeast-heat':24, 'Yeast-spo':24, 'Yeast-spo5':24, 'Yeast-spoem':24}
    datapath = 'Datasets\LDLDatasets\\' + dataset
    data = scio.loadmat(datapath)
    X = data['features']
    labels = pTol(data['labels'])
    n_clusters = K[dataset]  # the num of clusters
    kLabels = cls(X, cluster, n_clusters)
    print(kLabels)
    LPA = LablePropagation(maxstep=300)
    point = 0
    n = 0
    N = len(X)
    for i in range(n_clusters):
        members = kLabels == i
        LPA.fit(X[members], labels[members], alpha=Alpha[dataset], ctrl=ctrl)
        labels_tmp = LPA.labels
        # print(labels_tmp)
        point_tmp, n_tmp = calEval(data['labels'][members], labels_tmp, kernel=evaluation)
        point += point_tmp
        n += n_tmp
    print(point, n)
    p = point / n
    print('dataset:{},Point:{},N:{}'.format(dataset, p, N - n))

if __name__ == '__main__':
    # dataset = input('Please input the name of dataset:')
    # evaluation = input('Please input the evaluation:')
    # cluster = input('Please input the method of clustering:')
    # main(dataset=dataset, evaluation=evaluation)
    main()