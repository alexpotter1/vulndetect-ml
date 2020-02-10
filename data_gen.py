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
                    try:
                        file_count_for_class = data['X'].item().shape[0]
                        if file_count_for_class < self.min_seen_data_length:
                            self.min_seen_data_length = file_count_for_class
                    except AttributeError:
                        print("Got no X data for %s? Hmm (1/2)" % path)
                        self.min_seen_data_length = 5
        
        print("Common file count: " + str(self.min_seen_data_length))
        print("Loading data from vector bundles...")
        
        # two rounds of loads is kinda silly but oh well
        for _, path in enumerate(paths_temp):
            with np.load(path, allow_pickle=True) as data:
                encoded_texts = data['X'].item()
                label_ints = data['Y']

                if encoded_texts is None or label_ints is None:
                    # can't work with this
                    print("Got no data for npz path %s? Hmm (2/2)" % path)
                    continue

                encoded_texts = encoded_texts.toarray()

                # check texts and labels are same length
                if encoded_texts.shape[0] != label_ints.shape[0]:
                    print("WARNING: Texts and labels are different lengths, texts=%s, labels=%s" % (str(encoded_texts.shape[0]), str(label_ints.shape[0])))

                x.append(encoded_texts[:self.min_seen_data_length])
                y.append(label_ints[:self.min_seen_data_length])

        x = np.asarray(x)
        y = np.asarray(y)
        try:
            x = np.squeeze(x, axis=0)
            y = np.squeeze(y, axis=0)
        except ValueError as e:
            # x, y could be empty, return default
            print("\n\nWARNING: %s" % (str(e)))
            print("Returning placeholder zero arrays\n\n")
            x = np.zeros((self.dim), dtype=np.int32)
            y = np.zeros((self.dim), dtype=np.int32)

        print('X shape=%s' % str(x.shape))
        print('Y shape=%s' % str(y.shape))

        return x, y
