
from abc import ABC, abstractmethod 

class DetectorBase(ABC): 
    
    @abstractmethod
    def __init__(self, model, dataset, gpu): 
        pass
    
    @abstractmethod
    def train(self):
        pass
    
    @abstractmethod
    def detect_distribution(self):
        pass





