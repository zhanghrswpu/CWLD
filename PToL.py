import numpy as np
# obtain the logical labels
def pTol(y,threshold = 0.5):
    label_shape = y.shape
    labels = np.zeros(label_shape)
    for i in range(label_shape[0]):
        threshold_tmp = 0
        index = np.argsort(y[i])
        for j in range(len(index)):
            if threshold_tmp >= threshold:
                break
            threshold_tmp += y[i][index[len(index)-j-1]]
            labels[i][index[len(index)-j-1]] = 1
    return labels