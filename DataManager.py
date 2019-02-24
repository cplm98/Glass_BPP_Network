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
        cols = ['id', 'refractive_index', 'sodium', 'magnesium', 'aluminium', 'silicon', 'potassium', 'calcium', 'barium', 'iron', 'class']
        self.data = pandas.read_csv(path, index_col=False, names=cols)
        self.data = self.data.iloc[1:] # deletes header row from file
        print(self.data)

    def make__data_sets(self):
        # make unique copies of data
        train_set = self.data.copy()
        test_set = self.data.copy()
        # determine test and training indeces
        shape = self.data.shape # get size of datafram
        num_el = shape[0] #get num rows
        train_size = int(num_el * .8) # take 80% for training
        rand_i = random.sample(range(1,num_el), train_size) # select random unique data indeces
        # edit sets
        train_set = train_set.iloc[rand_i,:]
        test_set = test_set.loc[~test_set.index.isin(rand_i)]
        self.train_set = train_set
        self.test_set = test_set
        self.train_inputs = train_set.drop(columns=['id' , 'class'])
        self.train_targets = train_set['class']
