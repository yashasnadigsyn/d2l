import numpy as np

class RNN():
    def __init__(self, X_t, n_states):
        self.T = max(X_t.shape)
        self.X_t = X_t
        self.Y_hat = np.zeros((self.T, 1))
        self.n_states = n_states

        self.Wx = 0.1*np.random.randn(n_states,1)
        self.Wh = 0.1*np.random.randn(n_states, n_states)
        self.Wy = 0.1*np.random.randn(1, n_states)
        self.biases = 0.1*np.random.randn(n_states, 1)

        self.H = [np.zeros((n_states,1)) for t in range(self.T + 1)]

    def forward(self, Xt, ht_1):
        out = np.dot(self.Wx, Xt) + np.dot(self.Wh, ht_1) + self.biases
        ht = np.tanh(out)
        y_hat_t = np.dot(self.Wy, ht)

        return ht, y_hat_t, out

