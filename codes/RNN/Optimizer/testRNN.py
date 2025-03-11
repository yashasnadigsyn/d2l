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
optimizer = Optimizer_SGD(learning_rate=1e-5, momentum=0.95, decay=0.01)
T = rnn.T
n_epochs = 200

Monitor = np.zeros((n_epochs, 1))

for n in range(n_epochs):
    rnn.forward()
    Y_hat = rnn.Y_hat
    
    dY = Y_hat - Y_t
    L = 0.5*np.dot(dY.T, dY)/T
    
    Monitor[n] = L
    
    rnn.backward(dY)
    
    optimizer.pre_update_params()
    optimizer.update_params(rnn)
    optimizer.post_update_params()
    
    if not n%10:
        plt.plot(X_t, Y_t)
        plt.plot(X_t, Y_hat)
        plt.legend(['y - Actual', '$\hat{y}$ - Predicted'])
        plt.title('Epoch ' + str(n))
        plt.show()
        
plt.plot(range(n_epochs), Monitor)
plt.xlabel('Epochs')
plt.ylabel("MSE")
plt.yscale('log')
plt.show()