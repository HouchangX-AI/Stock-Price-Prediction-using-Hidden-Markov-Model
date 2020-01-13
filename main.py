import tushare
import numpy as np
import matplotlib.pyplot as plt
from GaussianHMM import GaussianHMM

# from hmmlearn import hmm

# data = tushare.get_hist_data("hs300")["close"]
# data = data.to_numpy()


if __name__ == "__main__":
    data = tushare.get_hist_data("hs300")["close"]
    n_days = len(data)
    train_data, test_data = (
        data[: int(n_days * 0.8)].to_numpy(),
        data[int(n_days * 0.8) :].to_numpy(),
    )
    print("length of training data: ", len(train_data))
    print("length of test data: ", len(test_data))

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

    num_states = [4, 8, 16, 32]
    for n in num_states:
        hmm = GaussianHMM(n_state=4, x_size=1, iter=50)
        hmm.train(train_data.reshape(-1, 1))
        predictions, _ = hmm.generate_seq(len(test_data))
        plt.figure()
        plt.plot(predictions, label="predictions")
        plt.plot(test_data, label="actual values")
        plt.legend()
        plt.ylim((2500, 5000))
        plt.title("{} hidden states".format(n))
        plt.xlabel("t")
        plt.ylabel("price")
        plt.savefig("{}_hidden_states_predictions".format(n))

    # print("means:", hmm.emit_means)
    # print("covs:", hmm.emit_covars)
    # print("initial: ", hmm.emit_prob)
    # print("transition: ", hmm.transmat_prob)
