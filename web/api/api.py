#!/usr/bin/env python3
from flask import request
from flask_cors import CORS
from predictor import AppWithOptionalPredictor
from models.api_response_model import APIResponse

model_path = '../../save_temp.h5'
encoder_path = '../../train_encoder'

app = AppWithOptionalPredictor(__name__)
app.create_predictor(model_path, encoder_path)
CORS(app)


@app.route('/api/heartbeat', methods=['GET'])
def heartbeat():
    return APIResponse().with_message('Heartbeat').with_statusCode(200).build()


@app.route('/api/predict', methods=['POST'])
def predict():
    code_text = request.form.get('file', None)

    if 'file' not in request.files:
        print('No files uploaded')
    else:
        file = request.files['file']
        print(file.name)
        file.seek(0)
        code_text = file.read().decode('utf-8')
    
    predictor_response = app.predict_from_text_sample(code_text)
    api_response = APIResponse()

    if predictor_response.status == 'OK':
        api_response = api_response \
            .with_statusCode(200) \
            .with_isVulnerable(predictor_response.isVulnerable) \
            .with_vulnerabilityCategory(predictor_response.vulnerabilityCategory) \
            .with_predictionConfidence(predictor_response.predictionConfidence)
    else:
        if 'parse Java input' in predictor_response.error:
            api_response = api_response.with_statusCode(400).with_message(predictor_response.error)
        else:
            api_response = api_response.with_statusCode(500)
    
    return api_response.build()
