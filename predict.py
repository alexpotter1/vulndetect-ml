#!/usr/bin/env python3

import numpy as np
from tensorflow.keras.models import load_model
import util
from parse_training_input import vectorise_texts, extract_bad_function

model_file = 'save_temp.h5'
input_path = "./new/test-cwe253.java"

print("Loading model...\n")
model = load_model(model_file)
print(model.summary())

print("Loading new code sample...")
code_sample = extract_bad_function(input_path)
x_test = vectorise_texts(code_sample)

print("Got x_test shape: ", x_test.shape)

predictions = model.predict(x_test.toarray(), verbose=1)
print("Got prediction shape: ", predictions.shape)
predicted_class = np.argmax(predictions[-1])
predicted_label = util.get_label_category_from_int(predicted_class)

print("Predicted: ", str(predicted_label))
print("Probability: ", round(predictions[-1][predicted_class] * 100, 2))
