import numpy as np
from sklearn.preprocessing import LabelEncoder
import tensorflow.keras


class DataGenerator(tensorflow.keras.utils.Sequence):
    def __init__(self, paths, batch_size=8, dim=(32, 32, 32), n_channels=1, n_classes=112, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.paths = paths
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.min_seen_data_length = None

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

        if self.min_seen_data_length is None:
            self.min_seen_data_length = 1e9
            print("Getting minimum common file count to collate across all vector bundles...")
            for _, path in enumerate(paths_temp):
                with np.load(path, allow_pickle=True) as data:
                    file_count_for_class = data['X'].item().shape[0]

                    if file_count_for_class < self.min_seen_data_length:
                        self.min_seen_data_length = file_count_for_class
        
        print("Minimum common file count: " + str(self.min_seen_data_length))
        print("Loading data from vector bundles...")
        
        # two rounds of loads is kinda silly but oh well
        for _, path in enumerate(paths_temp):
            with np.load(path, allow_pickle=True) as data:
                encoded_texts = data['X'].item()
                encoded_labels = data['Y'].item()

                if encoded_texts is None or encoded_labels is None:
                    # can't work with this
                    continue

                encoded_texts = encoded_texts.toarray()
                encoded_labels = encoded_labels.toarray()

                # check texts and labels are same dimensionality
                if encoded_texts.shape != encoded_labels.shape:
                    print("WARNING: Texts and labels are different shapes, texts=%s, labels=%s" % (str(encoded_texts.shape), str(encoded_labels.shape)))

                x.append(encoded_texts[:self.min_seen_data_length])
                y.append(encoded_labels[:self.min_seen_data_length])
                
        x = np.squeeze(np.asarray(x), axis=0)
        y = np.squeeze(np.asarray(y), axis=0)

        print('X shape=%s' % str(x.shape))
        print('Y shape=%s' % str(y.shape))

        return x, y
