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
        self._isVulnerable: bool = None
        self._vulnerabilityCategory: str = None
        self._predictionConfidence: float = None
    
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

    @property
    def isVulnerable(self):
        return self._isVulnerable

    @isVulnerable.setter
    def isVulnerable(self, value: bool):
        self._isVulnerable = bool(value)

    def with_isVulnerable(self, value: bool):
        self.isVulnerable = value
        return self

    @property
    def vulnerabilityCategory(self):
        return self._vulnerabilityCategory

    @vulnerabilityCategory.setter
    def vulnerabilityCategory(self, value: str):
        self._vulnerabilityCategory = str(value)

    def with_vulnerabilityCategory(self, value: str):
        self.vulnerabilityCategory = value
        return self

    @property
    def predictionConfidence(self):
        return self._predictionConfidence

    @predictionConfidence.setter
    def predictionConfidence(self, value: float):
        self._predictionConfidence = float(value)

    def with_predictionConfidence(self, value: float):
        self.predictionConfidence = value
        return self
    
    def build(self):
        return {
            'status': self.statusCode,
            'message': self.message,
            'isVulnerable': self.isVulnerable,
            'vulnerabilityCategory': self.vulnerabilityCategory,
            'predictionConfidence': self.predictionConfidence
        }
