import numpy as np
from math import exp

# this is gonna involve some matrix multiplication stuff
# matrix of inputs * matrix of weights = activation
# problem is going to be getting the weights to align properl in the matrix


def sigmoid(x):
    return (1 / (1 + exp(-x)))

def der_sig(y):
    # y = sigmoid(x)
    return y * (1-y)

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
        self.velocity = 
        self.c = .1 # learning rate
        self.beta = .9 # momentum

    def feed_forward(self, input_array, train="False"):  # feed forward works
        # gen hidden outputs
        input_arr = np.matrix(data=input_array).T  #.T cretes a column vector
        hidden_sum = np.matmul(self.weights_ih, input_arr)
        hidden_sum = np.add(hidden_sum, self.bias_hidden)
        # creating vector of sigmoid to map outputs
        vec_sigmoid = np.vectorize(sigmoid)
        # applying activation (sigmoid) function
        hidden_sum = vec_sigmoid(hidden_sum)

        # gen outputs
        output = np.matmul(self.weights_ho, hidden_sum)
        output = np.add(output, self.bias_out)
        output = vec_sigmoid(output)
        #if used in training, return hidden layer sums and output sums
        if train:
            return input_arr, hidden_sum, output
        else:
            return output

    def train(self, input, answer):
        targets = np.matrix(data=answer).T # convert to column matrix
        input_arr, hidden_sums, guesses = self.feed_forward(input, True)
        #***** Calculate output and hidden errors *****#
        print("Guess: ", guesses)
        output_err =   np.subtract(targets, guesses)
        print("error: ", output_err)
        # can simplify the matrix multiplication by just using the weights instead of the weight/sum of weights. Maintains proportionality
        weights_ho_t = np.transpose(self.weights_ho)
        hidden_err = np.matmul(weights_ho_t, output_err)

        #***** Calculate Gradients *****#
        vec_der_sig = np.vectorize(der_sig)
        # ih
        hidden_G = vec_der_sig(hidden_sums).T
        # print("hidden_G: ", hidden_G)
        # print("hidden_err", hidden_err)
        hidden_G = np.matmul(hidden_G, hidden_err) # getting matrix size errors
        hidden_G = hidden_G * self.c
        # ho
        guesses_G = vec_der_sig(guesses).T
        guesses_G = np.matmul(output_err, guesses_G)
        guesses_G = guesses_G * self.c

        #***** Calculate Deltas *****#
        # ih
        inputs_T = np.transpose(input_arr)
        weights_ih_deltas = np.matmul(hidden_G, inputs_T)
        # ho
        hidden_T = np.transpose(hidden_sums)
        weights_ho_deltas = np.matmul(guesses_G, hidden_T)

        #***** Adjust Weights *****#
        self.weights_ih = np.add(self.weights_ih, weights_ih_deltas)
        self.weights_ho = np.add(self.weights_ho, weights_ho_deltas)




test = BPNetwork(2, 2, 1)
inputs = ([0, 1],[1,0],[2,0],[2,1])
output = [1,2,1,3]
for i, input_ in enumerate(inputs):
    print("\n ih weights: ", test.weights_ih)
    print("\n ho weights: ", test.weights_ho)
    test.train(input_, output[i])
