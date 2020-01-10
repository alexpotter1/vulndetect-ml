import numpy as np
from sklearn.preprocessing import LabelEncoder
import keras
import math


class DataGenerator(keras.utils.Sequence):
    def __init__(self, paths, batch_size=8, dim=(32, 32, 32), n_channels=1, n_classes=112, shuffle=True):
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
        inds = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        paths_temp = [self.paths[i] for i in inds]
        x, y = self.__data_generation(paths_temp)
        return x, y
    
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)
        
    def __data_generation(self, paths_temp):
        x = np.empty((self.batch_size, *self.dim), dtype=object)
        y = np.empty((self.batch_size), dtype=object)

        for i, path in enumerate(paths_temp):
            with np.load(path, allow_pickle=True) as data:
                print("Loading %s" % path)
                class_x = data['X']
                print(class_x.shape)
                class_y = data['Y']

            if class_x.shape[0] != self.dim[0]:
                continue

            # ensure data coming from file is uniform length
            min_csr_x_subshape_ind0 = min([x.shape[0] for x in class_x])
            min_csr_y_subshape_ind0 = min([y.shape[0] for y in class_y])

            x_i_trim = np.asarray([i[:min_csr_x_subshape_ind0] for i in class_x])
            print("trim_x shape: " + str(x_i_trim.shape))
            x[i, ] = x_i_trim

            y[i] = np.asarray([i[:min_csr_y_subshape_ind0] for i in class_y])

        # ensure x and y first dimension are equal lengths
        # pylint: disable=E1136
        if y.shape[0] < x.shape[0]:
            y = np.repeat(y, math.ceil(x.shape[0] / y.shape[0]))
            y = y[:x.shape[0]]
        elif (y.shape[0] > x.shape[0]):
            y = y[x.shape[0]]

        # reshape to accommodate lstm timestep
        x = np.reshape(x, (x.shape[0], 1, x.shape[1]))

        print('X shape=%s' % str(x.shape))
        print('Y shape=%s' % str(y.shape))

        return x, y
