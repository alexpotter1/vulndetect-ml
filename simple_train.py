#!/usr/bin/env python3

import numpy as np
import tensorflow
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Embedding
import util
import os

try:
    import tensorflow_datasets as tfds
except ModuleNotFoundError:
    print("Tensorflow dataset module not installed. Installing now...")

    from setuptools.command.easy_install import main as install
    install(['tensorflow_datasets'])
    print("Installed!\n")

tensorflow.keras.backend.clear_session()
tensorflow.config.optimizer.set_jit(True)

np.random.seed(5)
categories = util.get_vulnerability_categories()

max_files_to_load = 2000
category_count = len(categories)
print("Loaded %i vulnerability categories from labels.txt" % category_count)
MAX_N_CHUNKS = 1000

# Model hyper-params
seq_length = 2 * 2048
filter_size = (3, 9, 19)
pooling_size = (3, 9, 19)
num_filter = 128
p_dropout = (0.25, 0.5)
hidden_dims = 128
num_quantised_chars = len(util.supported_char_list)

batch_size = 4

# get nist juliet dataset
try:
    tfds.load('nist_juliet_java')
except tfds.core.registered.DatasetNotFoundError:
    # Juliet dataset not yet registered, so explicitly register with TFDS
    util.register_custom_dataset_with_tfds()

    # re-import and try again!
    import tensorflow_datasets as tfds
    
train_dataset, train_info = tfds.load('nist_juliet_java/subwords16k', with_info=True, as_supervised=True, split='train[:80%]')
test_dataset, test_info = tfds.load('nist_juliet_java/subwords16k', with_info=True, as_supervised=True, split='train[-20%:]')

train_encoder = train_info.features['code'].encoder
print('Training vocabulary size: %i' % train_encoder.vocab_size)

print("Verifying encoder integrity...")
test_str = 'The quick brown fox jumped over the lazy dog'
encoded_decoded_str = train_encoder.decode(train_encoder.encode(test_str))

assert encoded_decoded_str == test_str
print("Verification successful!")

BUFFER_SIZE = 10000
BATCH_SIZE = 64

pad_shape = ([None], ())
train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE, padded_shapes=pad_shape)
test_dataset = test_dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE, padded_shapes=pad_shape)

if os.path.exists(os.path.join(os.getcwd(), 'save_temp.h5')):
    model = load_model('save_temp.h5')
    print(model.summary())
else:
    model = Sequential()
    model.add(Embedding(train_encoder.vocab_size, 64, name='embed'))
    model.add(LSTM(64))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(category_count, activation='softmax'))
    print(model.summary())

    model.compile(loss=tensorflow.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=tensorflow.keras.optimizers.Adam(1e-4), metrics=['sparse_categorical_accuracy'])

    history = model.fit(train_dataset, epochs=10, validation_data=test_dataset, validation_steps=30)
    model.save('save_temp.h5')

test_loss, test_acc = model.evaluate(test_dataset)
print('Test loss: {}'.format(test_loss))
print('Test accuracy: {}'.format(test_acc))

e = model.layers[0]
weights = e.get_weights()[0]
print(weights.shape)

with open('vecs.tsv', 'w', encoding='utf-8') as out_v:
    with open('meta.tsv', 'w', encoding='utf-8') as out_m:
        for num, token in enumerate(train_encoder.subwords):
            vec = weights[num + 1]
            out_m.write(token + '\n')
            out_v.write('\t'.join([str(x) for x in vec]) + '\n')