#!/usr/bin/env python3

import numpy as np
from tensorflow.keras.models import load_model
import tensorflow_datasets as tfds
import util
from parse_training_input import extract_bad_function_from_text

model_file = 'save_temp.h5'
encoder_file = 'train_encoder'
input_path = "./new/test-cwe253.java"

print("Loading model...\n")
model = load_model(model_file)
print(model.summary())

print("Loading encoder...")
encoder = tfds.features.text.SubwordTextEncoder.load_from_file(encoder_file)

print("Loading new code sample...")
with open(input_path, 'r') as f:
    sample_text = extract_bad_function_from_text(f.read())
    print(sample_text)

encoded = encoder.encode(sample_text)

predictions = model.predict(encoded, verbose=1)
print("Got prediction shape: ", predictions.shape)
predicted_class = np.argmax(predictions[-1])
predicted_label = util.get_label_category_from_int(predicted_class)

print("Predicted: ", str(predicted_label))
print("Probability: ", round(predictions[-1][predicted_class] * 100, 2))
