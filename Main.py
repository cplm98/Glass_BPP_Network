# 452 Assignment 2
# Written by: Connor Moore
# Student # : 20011955
# Date: Feb 26, 2019

from Glass_BPNV4 import BPNetwork
from DataManager import DataManager
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
# NNet = BPNetwork(9, 8, 6)

fileWrite = False
data = DataManager()
test = BPNetwork(9, 8, 7)
# initial weights
init_weights = test.weights_ih

# convert back to integer
def convert_decode(arr):
    if arr == [1, 0, 0, 0, 0, 0, 0]:
        return 1
    if arr == [0, 1, 0, 0, 0, 0, 0]:
        return 2
    if arr == [0, 0, 1, 0, 0, 0, 0]:
        return 3
    if arr == [0, 0, 0, 1, 0, 0, 0]:
        return 4
    if arr == [0, 0, 0, 0, 1, 0, 0]:
        return 5
    if arr == [0, 0, 0, 0, 0, 1, 0]:
        return 6
    if arr == [0, 0, 0, 0, 0, 0, 1]:
        return 7


# loop through epochs
for i in range(500):
    for i, td in enumerate(data.train_inputs):
        # train data
        test.train(td, data.train_targets[i])
    # reset internal correct counter
    test.correct_train = 0
    for i, td in enumerate(data.train_inputs):
        # test on training set
        test.train(td, data.train_targets[i])
    print("test: ", test.correct_train)

    # test on validation set
    # correct counter for each loop
    correct = 0
    for i, td in enumerate(data.val_inputs):
        target_arr = [0, 0, 0, 0, 0, 0, 0]
        target_arr[data.val_targets[i] - 1] = 1
        guess = test.feed_forward(td)
        if guess == target_arr:
            correct += 1
    print("validate: ", correct)
    # if score of over 70% achieved on validation, break
    if (correct / 32) > .7 or (test.correct_train/150) > .90:
        test.correct_train = 0
        break
    test.correct_train = 0
# correct counter reset
correct = 0
guess_arr = []
targets_arr = []
#***** TEST *****#
for i, td in enumerate(data.test_inputs):
    target_arr = [0, 0, 0, 0, 0, 0, 0]
    target_arr[data.test_targets[i] - 1] = 1
    guess = test.feed_forward(td)
    # print("target_Arr: ",target_arr)
    # print("Guess:      ", guess)
    if guess == target_arr:
        correct += 1
    guess_int = convert_decode(guess)
    guess_arr.append(guess_int)
    target_int = convert_decode(target_arr)
    targets_arr.append(target_int)

print("\n\n Test Accuracy: ", correct/32)
conf_matrix = confusion_matrix(guess_arr, targets_arr)
print("\n\n Confusion Matrix \n\n")
print(conf_matrix, "\n")

class_report=classification_report(guess_arr, targets_arr)
print("Precision and Recall \n")
print(class_report, "\n")

final_weights = test.weights_ih
print("\n\n Final Weights: \n", final_weights)

if fileWrite:
    f = open('452NN_A2_output.txt', 'w')
    f.write("Name: Connor Moore\nStudent #: 20011955")
    f.write("\n Program Generated Labels: \n" + str(guess_arr))
    f.write("\n Actual Labels: \n" + str(targets_arr))
    f.write("\n\n Final weights: " + np.array2string(final_weights))
    f.write("\n\n Confusion Matrix: \n" + np.array2string(conf_matrix))
    f.write("\n\n Precision and Recall:\n" + class_report)
