import numpy as np
from math import exp

# this is gonna involve some matrix multiplication stuff
# matrix of inputs * matrix of weights = activation
# problem is going to be getting the weights to align properl in the matrix


def sigmoid(x):
    return (1 / (1 + exp(-x)))

# each column of the array has the weights of a single node, and it's row represents where its pointing


class BPNetwork():
    def __init__(self, numI, numH, numO):
        self.in_nodes = numI
        self.hidden_nodes = numH
        self.out_nodes = numO
        self.weights_ih = np.random.uniform(
            low=0.001, high=1, size=(self.hidden_nodes, self.in_nodes))
        self.weights_ho = np.random.uniform(
            low=0.001, high=1, size=(self.out_nodes, self.hidden_nodes))
        self.bias_hidden = np.random.uniform(
            low=0.001, high=1, size=(self.hidden_nodes, 1))
        self.bias_out = np.random.uniform(
            low=0.001, high=1, size=(self.out_nodes, 1))

    def feed_forward(self, input_array):  # feed forward works
        # gen hidden outputs
        input_ = np.matrix(data=input_array).T  #.T cretes a column vector
        hidden_sum = np.matmul(self.weights_ih, input_)
        hidden_sum = np.add(hidden_sum, self.bias_hidden)
        vec_sigmoid = np.vectorize(sigmoid)
        # applying activation (sigmoid) function
        hidden_sum = vec_sigmoid(hidden_sum)

        # gen outputs
        output = np.matmul(self.weights_ho, hidden_sum)
        output = np.add(output, self.bias_out)
        output = vec_sigmoid(output)

        return output

    def train(self, input, answer):
        targets = np.matrix(data=answer).T # convert to column matrix
        guesses = self.feed_forward(input)
        print("Guess: ", guesses)
        err =   np.subtract(targets, guesses)
        print("error: ", err)


test = BPNetwork(2, 2, 1)
inputs = ([0, 1])
output = [1]
test.train(inputs, output)
