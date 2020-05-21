class PredictorResponseModel(object):
    def __init__(self, status=None, error=None, isVulnerable=None, vulnerabilityCategory=None, predictionConfidence=None):
        self._status = status
        self._error = error
        self._isVulnerable = isVulnerable
        self._vulnerabilityCategory = vulnerabilityCategory
        self._predictionConfidence = predictionConfidence
    
    @property
    def status(self):
        return self._status
    
    @property
    def error(self):
        return self._error
    
    @property
    def isVulnerable(self):
        return self._isVulnerable
    
    @property
    def vulnerabilityCategory(self):
        return self._vulnerabilityCategory
    
    @property
    def predictionConfidence(self):
        return self._predictionConfidence
