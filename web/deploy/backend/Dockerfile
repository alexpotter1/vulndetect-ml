FROM tensorflow/tensorflow:latest
ENV PYTHONUNBUFFERED 1
RUN python3 -m pip install gunicorn flask flask_cors tensorflow_datasets javalang
RUN mkdir api
WORKDIR /api
ADD . /api/
