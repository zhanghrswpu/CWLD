import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
from sklearn.cluster import Birch
import scipy.io as scio
import os

os.environ["OMP_NUM_THREADS"] = '10'
# 假设data是你的数据
datapath = 'Datasets\LDLDatasets\\' + 'yeast-alpha'
data = scio.loadmat(datapath)
X = data['features']

# 创建一个空列表来存储每个K值对应的簇内误差平方和
inertia_values = []

# 尝试不同的K值
for k in range(1, 35, 2):
    kmeans = MiniBatchKMeans(n_clusters=k, random_state=0,batch_size=4096)
    kmeans.fit(X)
    inertia_values.append(kmeans.inertia_)
    print(k, kmeans.inertia_)

# 绘制肘部法则图
plt.plot(range(1, 35, 2), inertia_values, marker='o')
plt.xlabel('k')
plt.ylabel('SSE')
# plt.title('Elbow Method for Optimal K')
plt.show()
