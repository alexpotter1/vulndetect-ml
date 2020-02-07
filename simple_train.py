#!/usr/bin/env python3

import numpy as np
import parse_training_input
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, LSTM, Dense, Lambda, Embedding
import util
from data_gen import DataGenerator
import json
import os
import datetime
import time

import ray
ray.init()

np.random.seed(5)
categories = util.get_vulnerability_categories()

max_files_to_load = 2000
category_count = len(categories)
MAX_N_CHUNKS = 1000

# Model hyper-params
seq_length = 2 * 2048
filter_size = (3, 9, 19)
pooling_size = (3, 9, 19)
num_filter = 128
p_dropout = (0.25, 0.5)
hidden_dims = 128
num_quantised_chars = len(util.supported_char_list)

batch_size = 1

if not util.do_saved_vectors_exist():
    print("\nParsing training data...\n")
    parse_training_input.save_vulnerable_code_samples(util.BASE_PATH)
    print("\nComplete. Starting training in 5 seconds...\n")
    time.sleep(5)

print("\nTraining...\n")
# setup data loader (generator)
labels = util.get_vulnerability_categories()
generator_params = {
    'dim': (17, 4096),
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

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tensorflow.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# obtain max vocabulary onehot size for embedding layer
embed_vocab_size = 64
if os.path.exists(util.SAVE_PATH + "auto_vars.json"):
    with open(util.SAVE_PATH + 'auto_vars.json', 'r') as f:
        auto_vars = json.load(f)
    
    embed_vocab_size = auto_vars.get('onehot_range_max', 64)
    print("\n\nEmbedding layer vocabulary size set to %i\n\n" % embed_vocab_size)
else:
    print("WARNING: Autogenerated variable json doesn't exist! Neural network training may fail!")

embedding_vector_length = 32
model = Sequential()
model.add(Embedding(embed_vocab_size, embedding_vector_length, input_length=generator_params['dim'][1]))
model.add(LSTM(64, recurrent_dropout=0.2, dropout=0.2))
model.add(Dropout(0.2))
model.add(Dense(4096))
model.add(Lambda(lambda x: tensorflow.expand_dims(x, -1)))
model.add(Dense(category_count, activation='softmax'))
# model.add(Activation('softmax'))
model.compile('adam', 'sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
print(model.summary())
steps_per_epoch = ((len(labels) * util.MAX_FILE_PARSE) // batch_size)
print("Steps/epoch: %s" % steps_per_epoch)
model.fit_generator(
    generator=training_gen,
    validation_data=validation_gen,
    steps_per_epoch=steps_per_epoch,
    use_multiprocessing=False,
    workers=1,
    callbacks=[tensorboard_callback])
model.save('save_temp.h5')
