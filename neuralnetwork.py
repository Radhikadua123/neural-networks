#Neural network 
import numpy as np
import time

#VARIABLES
n_hidden = 10   #number of nodes in the hidden layer
n_input = 10	#number of nodes in the input layer
n_output = 10	#number of nodes in the output layer
n_samples = 300 #number of samples for training

#HYPERVARIABLES
learning_rate = 0.01   #How fast our network moves
momentum = 0.9

np.random.seed(0) #always creates the same set of random numbers


# Activation function 1
def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x)) #for getting probability or rather non linear transformation or whenever input hits a neuron, it turns the number into probability

# Activation function 2 bcz for XOR this activation function tends to create more accurate results
def tanh_prime(x):
    return  1 - np.tanh(x)**2


def train(x, t, V, W, bv, bw): # x is input matrix, t is transformation of input matrix, V is the weights of 1st layer, W is the weights of 2nd layer, 
				#bv is the bias matrix for 1st layer and bw is the weight matrix for the 2nd layer
    # forward layer1
    A = np.dot(x, V) + bv  #linear transformation i.e [(input) * (weights of 1st layer) + biases]
    Z = np.tanh(A)         #non-linear transformation using activation function or sigmoid function.
    #forward layer2
    B = np.dot(Z, W) + bw
    Y = sigmoid(B)

    # backward
    Ew = Y - t
    Ev = tanh_prime(A) * np.dot(W, Ew)

    dW = np.outer(Z, Ew)
    dV = np.outer(x, Ev)

    loss = -np.mean ( t * np.log(Y) + (1 - t) * np.log(1 - Y) ) # cross entropy function used for calculating loss in case of classificatio
    return  loss, (dV, dW, Ev, Ew)


def predict(x, V, W, bv, bw):
    A = np.dot(x, V) + bv
    B = np.dot(np.tanh(A), W) + bw
    return (sigmoid(B) > 0.5).astype(int)


#Creating the weights for the hidden layers
V = np.random.normal(scale=0.1, size=(n_input, n_hidden))
W = np.random.normal(scale=0.1, size=(n_hidden, n_output))

#Creating biases
bv = np.zeros(n_hidden)
bw = np.zeros(n_output)

args = [V,W,bv,bw]

# Generating data
X = np.random.binomial(1, 0.5, (n_samples, n_input))
T = X ^ 1

# Training the network
for epoch in range(100):
    err = []
    upd = [0]*len(args)

    t0 = time.clock()
    for i in range(X.shape[0]):
        loss, grad = train(X[i], T[i], *args)

        for j in range(len(args)):
            args[j] -= upd[j]

        for j in range(len(args)):
            upd[j] = learning_rate * grad[j] + momentum * upd[j]

        err.append( loss )

    print "Epoch: %d, Loss: %.8f, Time: %.4fs" % (
                epoch, np.mean( err ), time.clock()-t0 )

# Try to predict something

x = np.random.binomial(1, 0.5, n_input)
print "XOR prediction"
print x
print predict(x, *args)

