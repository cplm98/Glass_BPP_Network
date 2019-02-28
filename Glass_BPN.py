# 452 Assignment 2
# Written by: Connor Moore
# Student # : 20011955
# Date: Feb 26, 2019

import numpy as np
from math import exp

# this works


# activation function
def sigmoid(x):
    return (1 / (1 + exp(-x)))

# derivative of activation function


def der_sig(y):
    # y = sigmoid(x)
    return y * (1 - y)


class BPNetwork():
    def __init__(self, numI, numH, numO):
        self.in_nodes = numI
        self.hidden_nodes = numH
        self.out_nodes = numO
        # weights from input->hidden layer
        self.weights_ih = np.random.uniform(
            low=-0.5, high=.5, size=(self.hidden_nodes, self.in_nodes))
        # weights from hidden->output layer
        self.weights_ho = np.random.uniform(
            low=-0.5, high=.5, size=(self.out_nodes, self.hidden_nodes))
        # bias weight vector for input->hidden
        self.bias_hidden = np.random.uniform(
            low=-.5, high=.5, size=(self.hidden_nodes, 1))
        # bias weight vector for hidden->output
        self.bias_out = np.random.uniform(
            low=-.5, high=.5, size=(self.out_nodes, 1))
        self.c = .05  # learning rate
        self.momentum = .9  # momentum
        # last weights adjustments from each layer
        self.last_weight_adjustment_ih = np.zeros(
            (self.hidden_nodes, self.in_nodes))
        self.last_weight_adjustment_ho = np.zeros(
            (self.out_nodes, self.hidden_nodes))
        # correct counters
        self.correct_test = 0
        self.correct_train = 0

    def feed_forward(self, input_array, train=False):
        # gen hidden outputs
        input_arr = np.matrix(data=input_array).T
        hidden_sum = np.matmul(self.weights_ih, input_arr)
        hidden_sum = np.add(hidden_sum, self.bias_hidden)
        # creating vector of sigmoid to map outputs
        vec_sigmoid = np.vectorize(sigmoid)
        # applying activation (sigmoid) function
        hidden_sum = vec_sigmoid(hidden_sum)
        # gen ouput outputs
        output = np.matmul(self.weights_ho, hidden_sum)
        output = np.add(output, self.bias_out)
        output = vec_sigmoid(output)
        if train:
            return input_arr, hidden_sum, output
        else:
            # hot one encode output
            result = [0, 0, 0, 0, 0, 0, 0]
            res_i = np.argmax(output)
            result[res_i] = 1
            return result

    def train(self, input, answer):
        target_arr = [0, 0, 0, 0, 0, 0, 0]
        result = [0, 0, 0, 0, 0, 0, 0]
        target = answer  # this is just the singular result
        # hot one encode the target
        target_arr[target - 1] = 1
        input_arr, hidden_sums, guesses = self.feed_forward(input, True)
        res_i = np.argmax(guesses)  # select largest value as answer
        result[res_i] = 1  # set it to one
        if result == target_arr:
            self.correct_train += 1

        #***** Calculate output and hidden errors *****#
        output_err = np.subtract(target_arr, guesses.T)
        # can simplify the matrix multiplication by just using the weights instead of the weight/sum of weights. Maintains proportionality
        weights_ho_t = np.transpose(self.weights_ho)
        hidden_err = np.matmul(weights_ho_t, output_err.T)

        #***** Calculate Gradients *****#
        vec_der_sig = np.vectorize(der_sig)
        hidden_G = vec_der_sig(hidden_sums)
        # elementwise multiplication
        hidden_G = np.array(hidden_G) * np.array(hidden_err)
        hidden_G = hidden_G * self.c
        # ho
        guesses_G = vec_der_sig(guesses)
        # (output_err, guesses_G)
        guesses_G = np.array(guesses_G) * np.array(output_err.T)
        guesses_G = guesses_G * self.c

        #***** Calculate Deltas *****#
        # ih
        inputs_T = np.transpose(input_arr)
        weights_ih_deltas = np.matmul(
            hidden_G, inputs_T) + (self.momentum * self.last_weight_adjustment_ih)
        self.last_weight_adjustment_ih = weights_ih_deltas
        # ho
        hidden_T = np.transpose(hidden_sums)
        weights_ho_deltas = np.matmul(
            guesses_G, hidden_T) + (self.momentum * self.last_weight_adjustment_ho)
        self.last_weight_adjustment_ho = weights_ho_deltas

        #***** Adjust Weights *****#
        self.weights_ih = np.add(self.weights_ih, weights_ih_deltas)
        self.weights_ho = np.add(self.weights_ho, weights_ho_deltas)
        # adjust bias
        self.bias_hidden = np.add(self.bias_hidden, hidden_G)
        self.bias_out = np.add(self.bias_out, guesses_G)
