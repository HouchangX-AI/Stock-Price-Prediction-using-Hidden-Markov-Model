import numpy as np
from abc import ABCMeta, abstractmethod


class _BaseHMM:
    """
    n_state : number of hidden states
    n_iter : 
    x_size : dimension of obs
    start_prob : initial prob
    transmat_prob : transition prob
    """

    __metaclass__ = ABCMeta

    def __init__(self, n_state=1, x_size=1, iter=20):
        self.n_state = n_state
        self.x_size = x_size
        # initialize
        self.start_prob = np.ones(n_state) * (1.0 / n_state)
        self.transmat_prob = np.ones((n_state, n_state)) * (1.0 / n_state)
        self.trained = False
        self.n_iter = iter

    @abstractmethod
    def _init(self, X):
        pass

    @abstractmethod
    def emit_prob(self, x):  # emission prob
        return np.array([0])

    # 虚函数
    @abstractmethod
    def generate_x(self, z):  # generate x according to p(x|z)
        return np.array([0])

    # 虚函数：发射概率的更新
    @abstractmethod
    def emit_prob_updated(self, X, post_state):
        pass

    # 通过HMM生成序列
    def generate_seq(self, seq_length):
        X = np.zeros((seq_length, self.x_size))
        Z = np.zeros(seq_length)
        Z_pre = np.random.choice(self.n_state, 1, p=self.start_prob)
        X[0] = self.generate_x(Z_pre)
        Z[0] = Z_pre

        for i in range(seq_length):
            if i == 0:
                continue
            # P(Zn+1)=P(Zn+1|Zn)P(Zn)
            Z_next = np.random.choice(
                self.n_state, 1, p=self.transmat_prob[Z_pre, :][0]
            )
            Z_pre = Z_next
            # P(Xn+1|Zn+1)
            X[i] = self.generate_x(Z_pre)
            Z[i] = Z_pre

        return X, Z

    # Evaluation
    def X_prob(self, X, Z_seq=np.array([])):
        X_length = len(X)
        if Z_seq.any():
            Z = np.zeros((X_length, self.n_state))
            for i in range(X_length):
                Z[i][int(Z_seq[i])] = 1
        else:
            Z = np.ones((X_length, self.n_state))

        _, c = self.forward(X, Z)  # P(x,z)

        prob_X = np.sum(np.log(c))  # P(X)
        return prob_X

    # def predict_next_prob(self, X, x_next, Z_seq=np.array([]), istrain=True):
    #     if self.trained == False or istrain == False:
    #         self.train(X)

    #     X_length = len(X)
    #     if Z_seq.any():
    #         Z = np.zeros((X_length, self.n_state))
    #         for i in range(X_length):
    #             Z[i][int(Z_seq[i])] = 1
    #     else:
    #         Z = np.ones((X_length, self.n_state))

    #     alpha, _ = self.forward(X, Z)  # P(x,z)
    #     prob_x_next = self.emit_prob(np.array([x_next])) * np.dot(
    #         alpha[X_length - 1], self.transmat_prob
    #     )
    #     return prob_x_next

    def decode(self, X, istrain=True):
        """
        Viterbi
        :param X: observations
        :param istrain: 
        :return: hidden states
        """
        if self.trained == False or istrain == False:
            self.train(X)

        X_length = len(X)
        state = np.zeros(X_length)

        pre_state = np.zeros((X_length, self.n_state))
        max_pro_state = np.zeros((X_length, self.n_state))

        _, c = self.forward(X, np.ones((X_length, self.n_state)))
        max_pro_state[0] = self.emit_prob(X[0]) * self.start_prob * (1 / c[0])

        # forward
        for i in range(X_length):
            if i == 0:
                continue
            for k in range(self.n_state):
                prob_state = (
                    self.emit_prob(X[i])[k]
                    * self.transmat_prob[:, k]
                    * max_pro_state[i - 1]
                )
                max_pro_state[i][k] = np.max(prob_state) * (1 / c[i])
                pre_state[i][k] = np.argmax(prob_state)

        # backward
        state[X_length - 1] = np.argmax(max_pro_state[X_length - 1, :])
        for i in reversed(range(X_length)):
            if i == X_length - 1:
                continue
            state[i] = pre_state[i + 1][int(state[i + 1])]

        return state

    #
    # def train_batch(self, X, Z_seq=list()):
    #
    #     self.trained = True
    #     X_num = len(X)
    #     self._init(self.expand_list(X))

    #     if Z_seq == list():
    #         Z = []
    #         for n in range(X_num):
    #             Z.append(list(np.ones((len(X[n]), self.n_state))))
    #     else:
    #         Z = []
    #         for n in range(X_num):
    #             Z.append(np.zeros((len(X[n]), self.n_state)))
    #             for i in range(len(Z[n])):
    #                 Z[n][i][int(Z_seq[n][i])] = 1

    #     for e in range(self.n_iter):  # EM
    #         #  E
    #         print("iter: ", e)
    #         b_post_state = []
    #         b_post_adj_state = np.zeros(
    #             (self.n_state, self.n_state)
    #         )
    #         b_start_prob = np.zeros(self.n_state)
    #         for n in range(X_num):
    #             X_length = len(X[n])
    #             alpha, c = self.forward(X[n], Z[n])  # P(x,z)
    #             beta = self.backward(X[n], Z[n], c)  # P(x|z)

    #             post_state = alpha * beta / np.sum(alpha * beta)  # Normalize
    #             b_post_state.append(post_state)
    #             post_adj_state = np.zeros((self.n_state, self.n_state))
    #             for i in range(X_length):
    #                 if i == 0:
    #                     continue
    #                 if c[i] == 0:
    #                     continue
    #                 post_adj_state += (
    #                     (1 / c[i])
    #                     * np.outer(alpha[i - 1], beta[i] * self.emit_prob(X[n][i]))
    #                     * self.transmat_prob
    #                 )

    #             if np.sum(post_adj_state) != 0:
    #                 post_adj_state = post_adj_state / np.sum(post_adj_state) # Normalize
    #             b_post_adj_state += post_adj_state
    #             b_start_prob += b_post_state[n][0]

    #         # M
    #         b_start_prob += 0.001 * np.ones(self.n_state)
    #         self.start_prob = b_start_prob / np.sum(b_start_prob)
    #         b_post_adj_state += 0.001
    #         for k in range(self.n_state):
    #             if np.sum(b_post_adj_state[k]) == 0:
    #                 continue
    #             self.transmat_prob[k] = b_post_adj_state[k] / np.sum(
    #                 b_post_adj_state[k]
    #             )

    #         self.emit_prob_updated(self.expand_list(X), self.expand_list(b_post_state))

    def expand_list(self, X):
        C = []
        for i in range(len(X)):
            C += list(X[i])
        return np.array(C)

    # 针对于单个长序列的训练
    def train(self, X, Z_seq=np.array([])):
        self.trained = True
        X_length = len(X)
        self._init(X)

        if Z_seq.any():
            Z = np.zeros((X_length, self.n_state))
            for i in range(X_length):
                Z[i][int(Z_seq[i])] = 1
        else:
            Z = np.ones((X_length, self.n_state))

        for e in range(self.n_iter):  # EM
            print(e, " iter")
            # E
            alpha, c = self.forward(X, Z)  # P(x,z)
            beta = self.backward(X, Z, c)  # P(x|z)

            post_state = alpha * beta
            post_adj_state = np.zeros((self.n_state, self.n_state))
            for i in range(X_length):
                if i == 0:
                    continue
                if c[i] == 0:
                    continue
                post_adj_state += (
                    (1 / c[i])
                    * np.outer(alpha[i - 1], beta[i] * self.emit_prob(X[i]))
                    * self.transmat_prob
                )
            # M
            self.start_prob = post_state[0] / np.sum(post_state[0])
            for k in range(self.n_state):
                self.transmat_prob[k] = post_adj_state[k] / np.sum(post_adj_state[k])

            self.emit_prob_updated(X, post_state)

    def forward(self, X, Z):
        X_length = len(X)
        alpha = np.zeros((X_length, self.n_state))  # P(x,z)
        alpha[0] = self.emit_prob(X[0]) * self.start_prob * Z[0]

        c = np.zeros(X_length)
        c[0] = np.sum(alpha[0])
        alpha[0] = alpha[0] / c[0]

        for i in range(X_length):
            if i == 0:
                continue
            alpha[i] = (
                self.emit_prob(X[i]) * np.dot(alpha[i - 1], self.transmat_prob) * Z[i]
            )
            c[i] = np.sum(alpha[i])
            if c[i] == 0:
                continue
            alpha[i] = alpha[i] / c[i]

        return alpha, c

    def backward(self, X, Z, c):
        X_length = len(X)
        beta = np.zeros((X_length, self.n_state))  # P(x|z)
        beta[X_length - 1] = np.ones((self.n_state))
        for i in reversed(range(X_length)):
            if i == X_length - 1:
                continue
            beta[i] = (
                np.dot(beta[i + 1] * self.emit_prob(X[i + 1]), self.transmat_prob.T)
                * Z[i]
            )
            if c[i + 1] == 0:
                continue
            beta[i] = beta[i] / c[i + 1]

        return beta
