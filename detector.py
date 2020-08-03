from analyze import analyze
from OOD_Generate_Mahalanobis import extract_features
from OOD_Regression_Mahalanobis import train_detector

class detector():
    def __init__(self, pre_trained_net_path, in_distribution, out_of_distribution, data_labels, num_classes, transformation=None):
        self.in_distribution = in_distribution
        self.out_of_distribution = out_of_distribution
        self.model = pre_trained_net_path
        self.num_classes = num_classes
        self.in_transform = transformation
        self.data_lables = data_labels
        
    def analyze_image(self, channel_labels):
        analyze(self.in_distribution,self.out_of_distribution,channel_labels,is_image=True,data_labels=self.data_labels)
    
    def analyze_data(self, channel_labels, signal_frequency):
        analyze(self.in_distribution,self.out_of_distribution,channel_labels,is_image=False,data_labels=self.data_labels,signal_frequency=signal_frequency)
        
    def detect_ood(self, gpu, batch_size):
        extract_features(self.model, self.in_distribution, self.data_labels[0], self.data_labels[1:], self.out_of_distribution, self.in_transform, gpu=gpu, batch_size=batch_size, num_classes=self.num_classes)
        train_detector([self.data_labels[0]], self.data_labels[1:])
        
    # def automate_attack(self):
        
    # def automate_adv_training(self):
    
    # def automate_adv_defense(self):
    
    # Give options for specfic attacks/defenses/training....
    
    