from flask import Flask
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

import util
from parse_training_input import extract_bad_function_from_text


class PredictorResponseModel(object):
    def __init__(self):
        self.status = None
        self.error = None
        self.isVulnerable = None
        self.vulnerabilityCategory = None
        self.predictionConfidence = None
    
    def with_status_ok(self):
        self.status = 'OK'
        return self
    
    def with_status_fail(self):
        self.status = 'FAIL'
        return self
    
    def with_error(self, error: str):
        self.error = str(error)
        return self
    
    def with_isVulnerable(self, isVulnerable: bool):
        self.isVulnerable = bool(isVulnerable)
        return self
    
    def with_vulnerabilityCategory(self, category: str):
        self.vulnerabilityCategory = str(category)
        return self
    
    def with_predictionConfidence(self, confidence: float):
        self.predictionConfidence = float(confidence)
        return self
    
    def build(self):
        fail = (False, '')
        if self.status is 'OK':
            if self.error is not None:
                fail = (True, 'Error cannot be set if status was OK')
            if self.isVulnerable is None:
                fail = (True, 'Vulnerable flag cannot be unset if status was OK')
            if self.predictionConfidence is None:
                fail = (True, 'Confidence must be set if status was OK')
        elif self.status is 'FAIL':
            if self.error is None:
                fail = (True, 'Error must be set if status was FAIL')
            if self.isVulnerable is not None:
                fail = (True, 'Vulnerable flag cannot be set if status was FAIL')
            if self.predictionConfidence is not None:
                fail = (True, 'Confidence cannot be set if status was FAIL')
            if self.vulnerabilityCategory is not None:
                fail = (True, 'Vulnerability category cannot be set if status was FAIL')
        else:
            fail = (True, 'Status was not set')

        ret = {
            'status': self.status,
            'error': self.error,
            'isVulnerable': None,
            'vulnerabilityCategory': None,
            'predictionConfidence': None,
        }

        if fail[0]:
            # force error because construction was not correct
            ret['status'] = 'FAIL'
            ret['error'] = 'PredictorResponseModel user construction error: %s' % fail[1]
        else:
            ret['isVulnerable'] = self.isVulnerable
            ret['vulnerabilityCategory'] = self.vulnerabilityCategory
            ret['predictionConfidence'] = self.predictionConfidence

        return ret


class Predictor(object):
    def __init__(self, model_path, encoder_path):
        self.model_path = model_path
        self.encoder_path = encoder_path
    
    def initialise_predictor_engine(self):
        self.model = tf.keras.models.load_model(self.model_path)
        self.encoder = tfds.features.text.SubwordTextEncoder.load_from_file(self.encoder_path)

    def predict_from_text_sample(self, text):
        response = PredictorResponseModel()
        sample_text = extract_bad_function_from_text(text)
        if len(sample_text) == 0:
            # didn't parse correctly
            response = response.with_status_fail().with_error('Could not parse Java input')
        else:
            encoded = self.encoder.encode(sample_text)

            # must reshape to indicate batch size of 1
            # otherwise, we get a prediction for every word/token in the sequence (rather than the sequence as a whole)
            encoded = np.asarray(encoded, dtype=np.int32).reshape(1, -1)

            prediction = self.model.predict(encoded, verbose=1)[0]
            predicted_class = np.argmax(prediction)
            response = response.with_status_ok()

            predicted_label = util.get_label_category_from_int(predicted_class)
            if predicted_label is not None:
                # is vulnerable
                response = response.with_isVulnerable(True).with_vulnerabilityCategory(predicted_label)
            else:
                response = response.with_isVulnerable(False)
            
            response = response.with_predictionConfidence(prediction[predicted_class] * 100)
        
        return response.build()
            

class AppWithOptionalPredictor(Flask):
    def __init__(self, name):
        super().__init__(name)

        self.predictor = None

    def createPredictor(self, model_path, encoder_path):
        if self.predictor is not None:
            self.predictor = Predictor(model_path, encoder_path)
            self.predictor.initialise_predictor_engine()
    
    def predict_from_text_sample(self, text):
        if self.predictor is None:
            raise RuntimeError('Predictor not initialised - call createPredictor() first')
        
        return self.predictor.predict_from_text_sample(text)
