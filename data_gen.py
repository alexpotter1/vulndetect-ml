import numpy as np
from sklearn.preprocessing import LabelEncoder
import tensorflow.keras
import math


class DataGenerator(tensorflow.keras.utils.Sequence):
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
        x = []
        y = []

        for _, path in enumerate(paths_temp):
            with np.load(path, allow_pickle=True) as data:
                print("Loading %s" % path)
                class_x = data['X']
                print(class_x.shape)
                class_y = data['Y']

            class_x = class_x[:self.dim[0]]
            if class_x.shape[0] == 0:
                # this won't work
                continue

            # ensure data coming from file is uniform length
            min_csr_x_subshape_ind0 = min([x.shape[0] for x in class_x])
            min_csr_y_subshape_ind0 = min([y.shape[0] for y in class_y])

            x_i_trim = np.asarray([i[:min_csr_x_subshape_ind0].toarray() for i in class_x])

            '''if (x_i_trim.shape[1] < self.dim[1]):
                diff = self.dim[1] - x_i_trim.shape[1]
                npad = ((0,0), (0, diff), (0,0))
                x_i_trim = np.pad(x_i_trim, pad_width=npad, mode='constant')'''

            print("trim_x shape: " + str(x_i_trim.shape))
            x.append(x_i_trim)

            y.append(np.asarray([i[:min_csr_y_subshape_ind0].toarray() for i in class_y], dtype=np.uint8))

        if len(x) == 0:
            x = np.empty((self.batch_size, *self.dim), dtype=np.uint8)
        else:
            x = np.asarray(x)[0]
        
        if len(y) == 0:
            y = np.empty((self.batch_size,), dtype=np.uint8)
        else:
            y = np.asarray(y)[0]

        # ensure x and y first dimension are equal lengths
        # pylint: disable=E1136
        if y.shape[0] < x.shape[0]:
            y = np.repeat(y, math.ceil(x.shape[0] / y.shape[0]))
            y = y[:x.shape[0]]
        elif (y.shape[0] > x.shape[0]):
            y = y[:x.shape[0]]

        x_transform = np.zeros((self.dim[0], self.dim[1], x.shape[2]))
        x_transform[:x.shape[0], :x.shape[1], :] = x
        x = x_transform

        y_transform = np.zeros((self.dim[0], self.dim[1], y.shape[2]))
        y_transform[:y.shape[0], :y.shape[1], :] = y
        y = y_transform

        # reshape to accommodate lstm timestep
        # x = np.reshape(x, (x.shape[0], 1, x.shape[2]))
        # y = np.reshape(y, (y.shape[0], 256, -1))

        print('X shape=%s' % str(x.shape))
        print('Y shape=%s' % str(y.shape))

        return x, y
