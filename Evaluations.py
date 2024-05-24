import numpy as np
import scipy
from scipy.spatial.distance import pdist
np.set_printoptions(threshold = np.inf)
np.set_printoptions(suppress = True)

def kl(P, Q):
    return scipy.stats.entropy(P, Q)


# kl散度loss
def kl_loss(pred, val):
    sum_ = 0
    for i, pred_i in enumerate(pred):
        sum_ += kl(val[i], pred_i)
    return sum_ / len(pred)

def calEval(y,y_pred,kernel):
    dis = 0
    n = len(y)
    if(kernel == 'chebyshev'):
        for i in range(len(y)):
            # print(y[i], y_pred[i])
            x = [y[i],y_pred[i]]
            d = pdist(x, kernel)
            # print(y[i],y_pred[i],d)
            dis += d
    elif kernel == 'cosine':
        for i in range(len(y)):
            # print(y[i], y_pred[i])
            d = np.dot(y[i], y_pred[i]) / (np.linalg.norm(y[i]) * np.linalg.norm(y_pred[i]))
            # print(y[i],y_pred[i],d)
            dis += d
    elif kernel == 'clark':
        for i in range(len(y)):
            # print(y[i], y_pred[i])
            d = np.sqrt(sum(((y[i]-y_pred[i])**2)/ ((y[i]+y_pred[i])**2 + 1e-5)))
            if d == np.inf or np.isnan(d):
                n = n - 1
                continue
            # print(y[i],y_pred[i],d)
            dis += d
    elif kernel == 'canberra':
        for i in range(len(y)):
            # print(y[i], y_pred[i])
            x = [y[i], y_pred[i]]
            # d = pdist(x, kernel)
            d = sum(np.abs(y[i] - y_pred[i]) / (y[i]+y_pred[i]))
            if d == np.inf or np.isnan(d):
                n = n - 1
                continue
            # print(y[i],y_pred[i],d)
            dis += d
    elif kernel == 'KL':
        for i in range(len(y)):
            # print(y[i], y_pred[i])
            # d = sum(y[i] * np.log((y[i])/(y_pred[i] + 1e-5)))
            d = kl(y[i], y_pred[i])
            if d == np.inf or np.isnan(d):
                n = n - 1
                continue
            # print(y[i],y_pred[i],d)
            dis += d
    else:
        for i in range(len(y)):
            for j in range(y.shape[1]):
                # print(y[i], y_pred[i])
                d = np.min([y[i,j],y_pred[i,j]])
                # print(y[i],y_pred[i],d)
                dis += d
    return dis,n

def calEval(y,y_pred,kernel):
    dis = 0
    n = len(y)
    if(kernel == 'chebyshev'):
        for i in range(len(y)):
            # print(y[i], y_pred[i])
            x = [y[i],y_pred[i]]
            d = pdist(x, kernel)
            # print(y[i],y_pred[i],d)
            dis += d
    elif kernel == 'cosine':
        for i in range(len(y)):
            # print(y[i], y_pred[i])
            d = np.dot(y[i], y_pred[i]) / (np.linalg.norm(y[i]) * np.linalg.norm(y_pred[i]))
            # print(y[i],y_pred[i],d)
            dis += d
    elif kernel == 'clark':
        for i in range(len(y)):
            # print(y[i], y_pred[i])
            d = np.sqrt(sum(((y[i]-y_pred[i])**2)/ ((y[i]+y_pred[i])**2 + 1e-5)))
            if d == np.inf or np.isnan(d):
                n = n - 1
                continue
            # print(y[i],y_pred[i],d)
            dis += d
    elif kernel == 'canberra':
        for i in range(len(y)):
            # print(y[i], y_pred[i])
            x = [y[i], y_pred[i]]
            # d = pdist(x, kernel)
            d = sum(np.abs(y[i] - y_pred[i]) / (y[i]+y_pred[i] + 1e-5))
            if d == np.inf or np.isnan(d):
                n = n - 1
                continue
            # print(y[i],y_pred[i],d)
            dis += d
    elif kernel == 'KL':
        for i in range(len(y)):
            # print(y[i], y_pred[i])
            # d = sum(y[i] * np.log((y[i])/(y_pred[i] + 1e-5)))
            d = kl(y[i], y_pred[i])
            if d == np.inf or np.isnan(d):
                n = n - 1
                continue
            # print(y[i],y_pred[i],d)
            dis += d
    else:
        for i in range(len(y)):
            for j in range(y.shape[1]):
                # print(y[i], y_pred[i])
                d = np.min([y[i,j],y_pred[i,j]])
                # print(y[i],y_pred[i],d)
                dis += d
    return dis,n