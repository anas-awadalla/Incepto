import torch
from torch.utils.data import Dataset, DataLoader


class IllegalDataDeminsionException(Exception):
    pass


class ImageInterpreter(object):
    
    def __init__(self, model, dataset, gpu=-1):
        
        data_loader = DataLoader(dataset,batch_size=len(dataset))
        itr = iter(data_loader)
        X,y = next(itr)
        
        if len(X.shape) != 4:
            raise IllegalDataDeminsionException("Excpected data to have deminsions 4 but got deminsion ",len(X.shape))
        
        if gpu == "-1":
            device = torch.device('cpu')
        else:
            device = torch.device('cuda:'+str(gpu))

        
        
    def interpret_image(self, image):
        pass
        









