import numpy as np
np.set_printoptions(threshold = np.inf)
np.set_printoptions(suppress = True)
class LablePropagation:
    def __init__(self, epsilon=1e-20, maxstep=100):
        self.epsilon = epsilon # set the precision of the iteration
        self.maxstep = maxstep # set the maxstep of the iteration
        self.y_probabilistic =None # the probabilistic label matrix
        self.y_logic = None  # the logical label matrix
        self.n_instance = None # the num of instances
        self.n_class = None # the num of class labels
        self.labels = None # the final probabilistic label matrix

    def init_param(self, X_data, y_data):
        # initialize the parameters
        self.n_instance = X_data.shape[0]
        self.n_class = y_data.shape[1]
        self.y_logic = y_data # the logical matrix is initialized by the MLL data
        self.y_probabilistic = self.y_logic # the probabilistic matrix is initialized by the logical one
        return

    def cal_dis2(self, node1, node2):
        # calculate the distance between samples (Euclidean distance) based on the feature space
        return (node1 - node2) @ (node1 - node2)


    def get_Vector(self, ctrl=True):
        if(ctrl==True):
            # extact the common vector fy0, which is represented by the virtual-label center of the label space
            fy0 = self.y_probabilistic.sum(axis=0)/self.n_instance
            # fuses the hidden information of label space into the common vector by heat coonduction.
            # step 1, label nodes to instance nodes
            fx = []
            for i in range(self.n_instance):
                fx_tmp = 0
                for j in range(self.n_class):
                    fx_tmp += fy0[j]*self.y_logic[i,j]

                if(self.y_logic.sum(axis=1)[i] != 0):
                    fx.append(fx_tmp/self.y_logic.sum(axis=1)[i])
                else:
                    fx.append(fx_tmp)
            # step 2, back to label nodes
            fy1 = []
            for i in range(self.n_class):
                fy1_tmp = 0
                for j in range(self.n_instance):
                    fy1_tmp += fx[j]*self.y_logic[j,i]
                if(self.y_logic.sum(axis=0)[i] != 0):
                    fy1.append(fy1_tmp/self.y_logic.sum(axis=0)[i])
                else:
                    fy1.append(fy1_tmp)
            return fy1
        else:
            return self.y_probabilistic.sum(axis=0)/self.n_instance




    def fit(self, X_data, y_data,alpha = 0.8, ctrl=True):
        self.init_param(X_data, y_data)
        step = 0
        # iteratively update the probabilistic matrix unless the stopping criterion is satisfied
        while step < self.maxstep:
            step += 1
            # obtain the common vector
            V = self.get_Vector(ctrl)
            V =np.array(V)
            y_probabilistic_new = alpha * V + (1 - alpha) * self.y_logic
            # print(np.abs(self.W - new_W).sum())
            if np.abs(self.y_probabilistic - y_probabilistic_new).sum() < self.epsilon:
                break
            self.y_probabilistic = y_probabilistic_new
        # print(step)
        self.labels = self.y_probabilistic
        # regularize the final probabilistic label matrix
        for i in range(len(self.labels)):
            self.labels[i] /= self.labels[i].sum()
        return

