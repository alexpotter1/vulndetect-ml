#!/usr/bin/env python3
from flask import Flask

app = Flask(__name__)

STATUS_CODES = {
    200: '200 OK',
    201: '201 Created',
    400: '400 Bad Request',
    404: '404 Not Found',
    500: '500 Internal Server Error',
}


class APIResponse(object):
    def __init__(self):
        self._statusCode: str = None
        self._message: str = None
    
    @property
    def statusCode(self):
        return self._statusCode
    
    @statusCode.setter
    def statusCode(self, value: int):
        if value in STATUS_CODES:
            self._statusCode = STATUS_CODES[value]
        else:
            self._statusCode = '501 Not Implemented'

    def with_statusCode(self, value: int):
        self.statusCode = value
        return self

    @property
    def message(self):
        return self._message

    @message.setter
    def message(self, value: str):
        self._message = str(value)

    def with_message(self, value: str):
        self.message = value
        return self
    
    def build(self):
        return {
            'status': self.statusCode,
            'message': self.message
        }


@app.route('/')
def default_hello():
    return APIResponse().with_message('Hello from flask').with_statusCode(200).build()
