import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import ReLU
from misc_functions import (get_example_params,
                            convert_to_grayscale,
                            save_gradient_images,
                            get_positive_negative_saliency)
from captum.attr import GuidedBackprop

class IllegalDataDeminsionException(Exception):
    pass


class ImageInterpreter(object):
    
    def __init__(self, model, dataset=None, gpu=-1):
        
        self.model = model

        # Put model in evaluation mode
        self.model.eval()

        
        # data_loader = DataLoader(dataset,batch_size=len(dataset))
        # itr = iter(data_loader)
        # X,y = next(itr)
        
        # if len(X.shape) != 4:
        #     raise IllegalDataDeminsionException("Expected data to have deminsions 4 but got deminsion ",len(X.shape))
        
        if gpu == -1:
            device = torch.device('cpu')
        else:
            device = torch.device('cuda:'+str(gpu))
            
        self.algo = GuidedBackprop(model)

        
        
    def interpret_image(self, image, target_class, result_dir):
        # Get gradients
        guided_grads = self.algo.attribute(image, target_class).detach().cpu().numpy()[0]
        print(guided_grads.shape)
        # Save colored gradients
        save_gradient_images(guided_grads, result_dir + '_Guided_BP_color')
        # Convert to grayscale
        grayscale_guided_grads = convert_to_grayscale(guided_grads* image.detach().numpy()[0]
)
        # Save grayscale gradients
        save_gradient_images(grayscale_guided_grads, result_dir + '_Guided_BP_gray')
        # Positive and negative saliency maps
        pos_sal, neg_sal = get_positive_negative_saliency(guided_grads)
        save_gradient_images(pos_sal, result_dir + '_pos_sal')
        save_gradient_images(neg_sal, result_dir + '_neg_sal')
        print('Guided backprop completed')
