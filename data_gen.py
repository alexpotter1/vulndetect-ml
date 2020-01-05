import numpy as np
from sklearn.preprocessing import LabelEncoder
import keras
import math
import util

class DataGenerator(keras.utils.Sequence):
    def __init__(self, paths, batch_size=8, dim=(32,32,32), n_channels=1, n_classes=112, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.paths = paths
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.shuffle = shuffle

        self.label_encoder = LabelEncoder()
        print("Generator initialised.")
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.paths) / self.batch_size))
    
    def __getitem__(self, index):
        inds = self.indexes[index * self.batch_size:(index+1)*self.batch_size]
        paths_temp = [self.paths[i] for i in inds]
        x, y = self.__data_generation(paths_temp)
        return x,y
    
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)
        
    def __data_generation(self, paths_temp):
        x = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=object)

        for i, path in enumerate(paths_temp):
            class_x = None
            class_y = None
            with np.load(path) as data:
                print("Loading %s" % path)
                class_x = data['X']
                print(class_x.shape)
                class_y = data['Y']
                print(class_y)

            if class_x.shape[0] != self.dim[0]:
                continue

            x[i,] = class_x
            y[i] = class_y
        
        # keras can't one-hot encode strings, only integers, so transform first
        y = [e for e in y if e is not None]
        y = np.concatenate(y).ravel().tolist()
        label_vectors = self.label_encoder.fit_transform(y)

        samples = sum(len(s) for s in x)
        x = np.reshape(x, (samples, util.VEC_SIZE, self.n_channels))
        y = keras.utils.to_categorical(label_vectors).ravel()

        # ensure x and y first dimension are equal lengths
        if y.shape[0] < x.shape[0]:
            y = np.repeat(y, math.ceil(x.shape[0] / y.shape[0]))
            y = y[:x.shape[0]]
        elif (y.shape[0] > x.shape[0]):
            y = y[x.shape[0]]

        print('X shape=%s' % str(x.shape))
        print('Y shape=%s' % str(y.shape))

        return x,y

