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

encoded = encoder.encode(sample_text)

# must reshape to indicate batch size of 1
# otherwise, we get a prediction for every word/token in the sequence (rather than the sequence as a whole)
encoded = np.asarray(encoded, dtype=np.int32).reshape(1, -1)

prediction = model.predict(encoded, verbose=1)[0]
predicted_class = np.argmax(prediction)
predicted_label = util.get_label_category_from_int(predicted_class)

print("\nPredicted: ", str(predicted_label))
print("Prediction confidence: {}%".format(round(prediction[predicted_class] * 100, 2)))
