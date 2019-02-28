# 452 Assignment 2
# Written by: Connor Moore
# Student # : 20011955
# Date: Feb 26, 2019

import pandas
import os.path
import random


class DataManager():
    def __init__(self):
        self.data = None
        self.train_set = None
        self.test_set = None
        self.read_data()
        self.make__data_sets()

    def read_data(self):
        my_path = os.path.abspath(os.path.dirname(__file__))
        path = os.path.join(my_path, "GlassData.csv")
        cols = ['id', 'refractive_index', 'sodium', 'magnesium', 'aluminium',
                'silicon', 'potassium', 'calcium', 'barium', 'iron', 'class']
        self.data = pandas.read_csv(path, index_col=False, names=cols)
        self.data = self.data.iloc[1:]  # deletes header row from file

    def make__data_sets(self):
        # make unique copies of data
        train_set = self.data.copy()
        test_set = self.data.copy()
        val_set = self.data.copy()
        # determine test and training indeces
        shape = self.data.shape  # get size of datafram
        num_el = shape[0]  # get num rows
        train_size = int(num_el * .7)  # take 80% for training
        # select random unique data indeces
        rand_i = random.sample(range(1, num_el), train_size)
        inv_rand_i = []
        for i in range(1, num_el):
            if i not in rand_i:
                inv_rand_i.append(i)
        # basically the issue is that the inv_r_i is ordered
        print("test and val", inv_rand_i)
        test_i = random.sample(inv_rand_i, 32)
        val_i = []
        while len(val_i) < 33:
            temp = random.sample(inv_rand_i, 1)
            if temp not in test_i:
                val_i.append(temp[0])
        # val_i = inv_rand_i[~test_i]
        # test_i = inv_rand_i[:len(inv_rand_i)//2]
        # val_i = inv_rand_i[len(inv_rand_i)//2:]
        # edit sets
        test_set = test_set.iloc[test_i, :]
        val_set = val_set.iloc[val_i, :]
        # val_set = val_set.loc[~test_set.index.isin(rand_i and test_i)]
        train_set = train_set.iloc[rand_i, :] # take all indeces in rand_i
        #test_set = test_set.loc[~test_set.index.isin(rand_i)] # take all indeces not in rand_i
        test_inputs = test_set.drop(columns=['id', 'class'])
        # print(test_inputs)
        test_inputs = test_inputs.convert_objects(convert_numeric=True)
        test_inputs = self.minmax_norm(test_inputs.copy())
        self.test_inputs = test_inputs.values
        self.test_targets = test_set['class'].astype(int).tolist()
        print("test target list: ", self.test_targets)

        val_inputs = val_set.drop(columns=['id', 'class'])
        val_inputs = val_inputs.convert_objects(convert_numeric=True)
        val_inputs = self.minmax_norm(val_inputs.copy())
        self.val_inputs = val_inputs.values
        self.val_targets = val_set['class'].astype(int).tolist()

        train_inputs = train_set.drop(columns=['id', 'class'])
        train_inputs = train_inputs.convert_objects(convert_numeric=True)
        train_inputs = self.minmax_norm(train_inputs.copy())
        self.train_inputs = train_inputs.values
        self.train_targets = train_set['class'].astype(int).tolist()

    def minmax_norm(self, df):
        for column in df:
            min_ = df[column].min()
            max_ = df[column].max()
            adjusted = []
            for i, val in enumerate(df[column]):
                adjusted.append((val-min_)/(max_-min_))
            df[column] = adjusted
        return df
