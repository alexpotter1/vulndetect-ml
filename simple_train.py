#!/usr/bin/env python3

import numpy as np
import parse_training_input
from keras.models import Sequential, Model
from keras.layers import Embedding, Dropout, Bidirectional, LSTM, Dense, Activation
from keras.layers.merge import Concatenate
import util
from data_gen import DataGenerator
import math
import time

import ray
ray.init()

np.random.seed(5)
categories = util.get_vulnerability_categories()

max_files_to_load = 2000
category_count = len(categories)

# Model hyper-params
seq_length = 2 * 2048
filter_size = (3, 9, 19)
pooling_size = (3, 9, 19)
num_filter = 128
p_dropout = (0.25, 0.5)
hidden_dims = 128
num_quantised_chars = len(util.supported_char_list)

batch_size = 8

if not util.do_saved_vectors_exist():
    print("\nParsing training data...\n")
    parse_training_input.save_vulnerable_code_samples(util.BASE_PATH)
    print("\nTraining data vectorisation complete. Starting training in 5 seconds...\n")
    time.sleep(5)

print("\nTraining...\n")
# setup data loader (generator)
labels = util.get_vulnerability_categories()
generator_params = {
    'dim': (17,102400),
    'batch_size': batch_size,
    'n_classes': len(labels),
    'n_channels': 71,
    'shuffle': True,
}

full = util.get_saved_vector_list()
# 70/30 train validate split
split = round(len(full) * 0.7)
train = full[:split]
validate = full[split:]
print((len(train), len(validate)))

training_gen = DataGenerator(train, **generator_params)
validation_gen = DataGenerator(validate, **generator_params)

'''vectors, labels = map(list, zip(*functions))

shuffle_indices = np.random.permutation(np.arange(len(labels)))
vec_shuf = vectors[shuffle_indices]
lab_shuf = labels[shuffle_indices]'''

model = Sequential()

model.add(LSTM(256, input_shape=(util.VEC_SIZE, generator_params['n_channels']), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(1, activation='softmax'))
model.compile('adam', 'sparse_categorical_crossentropy', metrics=['accuracy'])
print(model.summary())
steps_per_epoch = ((len(labels) * util.MAX_FILE_PARSE) // batch_size)
print("Steps/epoch: %s" % steps_per_epoch)
model.fit_generator(generator=training_gen, validation_data=validation_gen, steps_per_epoch=steps_per_epoch, use_multiprocessing=True, workers=12)
model.save('save_temp.h5')