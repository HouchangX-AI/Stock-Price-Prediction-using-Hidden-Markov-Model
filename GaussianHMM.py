from hmm import _BaseHMM
import numpy as np
from math import pi, sqrt, exp, pow, log
from numpy.linalg import det, inv
from sklearn import cluster


class GaussianHMM(_BaseHMM):
    """
    发射概率为高斯分布的HMM
    参数：
    emit_means: 高斯发射概率的均值
    emit_covars: 高斯发射概率的方差
    """

    def __init__(self, n_state=1, x_size=1, iter=20):
        _BaseHMM.__init__(self, n_state=n_state, x_size=x_size, iter=iter)
        self.emit_means = np.zeros((n_state, x_size))  # 高斯分布的发射概率均值
        self.emit_covars = np.zeros((n_state, x_size, x_size))  # 高斯分布的发射概率协方差
        for i in range(n_state):
            self.emit_covars[i] = np.eye(x_size)  # 初始化为均值为0，方差为1的高斯分布函数

    def _init(self, X):
        # 通过K均值聚类，确定状态初始值
        mean_kmeans = cluster.KMeans(n_clusters=self.n_state)
        mean_kmeans.fit(X)
        self.emit_means = mean_kmeans.cluster_centers_
        for i in range(self.n_state):
            self.emit_covars[i] = np.cov(X.T) + 0.01 * np.eye(len(X[0]))

    # 二元高斯分布函数
    def gauss2D(self, x, mean, cov):
        # x, mean, cov均为numpy.array类型
        z = -np.dot(np.dot((x - mean).T, inv(cov)), (x - mean)) / 2.0
        temp = pow(sqrt(2.0 * pi), len(x)) * sqrt(det(cov))
        return (1.0 / temp) * exp(z)

    def emit_prob(self, x):  # 求x在状态k下的发射概率
        prob = np.zeros((self.n_state))
        for i in range(self.n_state):
            prob[i] = self.gauss2D(x, self.emit_means[i], self.emit_covars[i])
        return prob

    def generate_x(self, z):  # 根据状态生成x p(x|z)
        return np.random.multivariate_normal(
            self.emit_means[z][0], self.emit_covars[z][0], 1
        )

    def predict_sequence(self, X, predict_length=1):
        # alpha shape: T*N
        alpha, _ = self.forward(X, np.ones((len(X), self.n_state)))
        print("alpha: ", alpha)
        alpha = alpha[-1]
        predictions = []
        for _ in range(predict_length):
            posterier_hidden_prob = alpha / alpha.sum()
            p_mu = np.dot(self.emit_means.T, self.transmat_prob)

            x = np.dot(p_mu, posterier_hidden_prob)
            predictions.append(x[0])
            alpha = self.emit_prob(x) * np.dot(alpha, self.transmat_prob)

        return predictions

    def emit_prob_updated(self, X, post_state):  # 更新发射概率
        for k in range(self.n_state):
            for j in range(self.x_size):
                self.emit_means[k][j] = np.sum(post_state[:, k] * X[:, j]) / np.sum(
                    post_state[:, k]
                )

            X_cov = np.dot(
                (X - self.emit_means[k]).T,
                (post_state[:, k] * (X - self.emit_means[k]).T).T,
            )
            self.emit_covars[k] = X_cov / np.sum(post_state[:, k])
            if det(self.emit_covars[k]) == 0:  # 对奇异矩阵的处理
                self.emit_covars[k] = self.emit_covars[k] + 0.01 * np.eye(len(X[0]))
