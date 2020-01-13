import tushare
import numpy as np
import matplotlib.pyplot as plt
from GaussianHMM import GaussianHMM

# from hmmlearn import hmm

# data = tushare.get_hist_data("hs300")["close"]
# data = data.to_numpy()


if __name__ == "__main__":
    data = tushare.get_hist_data("hs300")["close"]
    data = data.to_numpy()

    # num_iters = 100
    # num_states = [4, 8, 16, 32]
    # for n in num_states:
    #     prob = []
    #     for i in range(num_iters):
    #         hmm = GaussianHMM(n_state=n, x_size=1, iter=i)
    #         hmm.train(data.reshape(-1, 1))
    #         prob.append(hmm.X_prob(data.reshape(-1, 1)))
    #     plt.plot(range(num_iters), prob, label="states number: {}".format(n))
    # plt.legend()
    # plt.xlabel("number of iterations")
    # plt.ylabel("loglikelihood")
    # plt.show()

    hmm = GaussianHMM(n_state=8, x_size=1, iter=20)
    hmm.train(data.reshape(-1, 1))
    predictions = hmm.predict_sequence(data.reshape(-1, 1), predict_length=10)
    print(predictions)
    # print("means:", hmm.emit_means)
    # print("covs:", hmm.emit_covars)
    # print("initial: ", hmm.emit_prob)
    # print("transition: ", hmm.transmat_prob)
