#!/usr/bin/env python3

import tensorflow
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout, Embedding
import util
import os
import datetime
from tfds_juliet.tfds_juliet import NISTJulietJava

try:
    import tensorflow_datasets as tfds
except ModuleNotFoundError:
    print("Tensorflow dataset module not installed. Installing now...")

    from setuptools.command.easy_install import main as install
    install(['tensorflow_datasets'])
    print("Installed! --- Re run this script! ---\n")

tensorflow.keras.backend.clear_session()
tensorflow.config.optimizer.set_jit(True)

categories = util.get_vulnerability_categories()
category_count = len(categories)
print("Loaded %i vulnerability categories from labels.txt" % category_count)

# get nist juliet dataset
print("\nFetching NIST Juliet Java code dataset")
print("WARNING: THIS MAY TAKE A WHILE! PLEASE DO NOT INTERRUPT\n")
train_dataset, train_info = tfds.load('nist_juliet_java/subwords16k', with_info=True, as_supervised=True, split='train[:80%]')
test_dataset, test_info = tfds.load('nist_juliet_java/subwords16k', with_info=True, as_supervised=True, split='train[-20%:]')

if os.path.exists(os.path.join(os.getcwd(), 'train_encoder.subwords')):
    print('Loading encoder from file')
    train_encoder = tfds.features.text.SubwordTextEncoder.load_from_file('train_encoder')
else:
    print('Building encoder')
    train_encoder = train_info.features['code'].encoder
    print('Training vocabulary size: %i' % train_encoder.vocab_size)

print("Verifying encoder integrity...")
test_str = 'The quick brown fox jumped over the lazy dog'
encoded_decoded_str = train_encoder.decode(train_encoder.encode(test_str))

assert encoded_decoded_str == test_str
print("Verification successful!")

print("Saving encoder to file...")
train_encoder.save_to_file('train_encoder')

BUFFER_SIZE = 10000
BATCH_SIZE = 64

pad_shape = ([None], ())
train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE, padded_shapes=pad_shape)
test_dataset = test_dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE, padded_shapes=pad_shape)

log_dir = "logs/fit" + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
tensorboard_callback = tensorflow.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
es_callback = tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-2, patience=2, verbose=1)

if os.path.exists(os.path.join(os.getcwd(), 'save_temp.h5')):
    model = load_model('save_temp.h5')
    print(model.summary())
else:
    model = Sequential()
    model.add(Embedding(train_encoder.vocab_size, 64, name='embed'))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(category_count, activation='softmax'))
    print(model.summary())

    model.compile('adam', 'sparse_categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(train_dataset, epochs=20, validation_data=test_dataset, validation_steps=30, callbacks=[tensorboard_callback, es_callback])
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
print("Saved vecs.tsv, meta.tsv for embedding visualisation")
