from MyRNN import *

import numpy as np
import matplotlib.pyplot as plt

X_t = np.arange(-10, 10, 0.1)
X_t = X_t.reshape(len(X_t), 1)
Y_t = np.sin(X_t) + 0.1*np.random.randn(len(X_t), 1)

plt.plot(X_t, Y_t)
plt.show()

# Testing RNN
n_states = 500
rnn = RNN(X_t, n_states, Tanh())
T = rnn.T
n_epochs = 200
e = 1e-5

for n in range(n_epochs):
    rnn.forward()
    Y_hat = rnn.Y_hat
    
    dY = Y_hat - Y_t
    L = 0.5*np.dot(dY.T, dY)/T
    
    print(float(L))
    
    rnn.backward(dY)
    rnn.Wx -= e*rnn.dWx
    rnn.Wy -= e*rnn.dWy
    rnn.Wh -= e*rnn.dWh
    rnn.biases -= e*rnn.dbiases

    plt.plot(X_t, Y_t)
    plt.plot(X_t, Y_hat)
    plt.legend(['y - Actual', '$\hat{y}$ - Predicted'])
    plt.title('Epoch ' + str(n))
    plt.show()