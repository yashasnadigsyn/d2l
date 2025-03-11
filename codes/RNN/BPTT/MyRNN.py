import numpy as np

class RNN():
    def __init__(self, X_t, n_states, Activation):
        self.T = max(X_t.shape)
        self.X_t = X_t
        self.Y_hat = np.zeros((self.T, 1))
        self.n_states = n_states

        self.Wx = 0.1*np.random.randn(n_states,1)
        self.Wh = 0.1*np.random.randn(n_states, n_states)
        self.Wy = 0.1*np.random.randn(1, n_states)
        self.biases = 0.1*np.random.randn(n_states, 1)

        self.H = [np.zeros((n_states,1)) for t in range(self.T + 1)]
        self.Activation = Activation

    def forward(self):
        self.dWx = np.zeros((self.n_states, 1))
        self.dWh = np.zeros((self.n_states, self.n_states))
        self.dWy = np.zeros((1, self.n_states))
        self.dbiases = np.zeros((self.n_states, 1))

        X_t = self.X_t
        H = self.H
        Y_hat = self.Y_hat
        ht = H[0]

        Activation = self.Activation
        ACT = [Activation for t in range(self.T)]

        [ACT, H, Y_hat] = self.RNNCell(X_t, ht, ACT, H, Y_hat)

        self.Y_hat = Y_hat
        self.H = H
        self.ACT = ACT

    def RNNCell(self, X_t, ht, ACT, H, Y_hat):

        for t, xt in enumerate(X_t):
            xt = xt.reshape(1,1)
            out = np.dot(self.Wx, xt) + np.dot(self.Wh, ht) + self.biases

            ACT[t].forward(out)
            ht = ACT[t].output

            y_hat_t = np.dot(self.Wy, ht)

            H[t+1] = ht
            Y_hat[t] = y_hat_t

        return (ACT, H, Y_hat)
    
    def backward(self, dvalues):
        T = self.T
        H = self.H
        X_t = self.X_t
        ACT = self.ACT
        
        dWx = self.dWx
        dWy = self.dWy
        dWh = self.dWh
        dbiases = self.dbiases
        
        Wy = self.Wy
        Wh = self.Wh
        
        dht = np.dot(Wy.T, dvalues[-1].reshape(1,1))
        
        for t in reversed(range(T)):
            dy = dvalues[t].reshape(1,1)
            xt = X_t[t].reshape(1,1)
            
            ACT[t].backward(dht)
            dtanh = ACT[t].dinputs
            
            dWx += np.dot(dtanh, xt)
            dWy += np.dot(H[t+1], dy).T
            dWh += np.dot(H[t], dtanh.T)
            dbiases += dtanh
            
            dht = np.dot(Wh, dtanh) + np.dot(Wy.T, dy)
            
        self.dWx = dWx
        self.dWy = dWy
        self.dWh = dWh
        self.dbiases = dbiases

class Tanh():
    def forward(self, inputs):     
        self.output = np.tanh(inputs)
        self.inputs = inputs

    def backward(self, dvalues):
        deriv = 1 - self.output**2
        self.dinputs = np.multiply(deriv, dvalues)

        