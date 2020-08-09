from analyze import analyze
from OOD_Generate_Mahalanobis import extract_features
from OOD_Regression_Mahalanobis import train_detector
from ADV_Samples import generate_adv_samples
from datasets.parkinsons_dataset import parkinsonsData
from models.parkinsonsNet import Network
import os
import torch
import torchvision
from torchvision.transforms import transforms

class client():
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
    
    def attack(self, adv_type, gpu, batch_size=200, in_transform=None):
        generate_adv_samples(model=self.model,net_type="model",dataset_name="mPower",dataset = self.in_distribution,gpu=gpu,adv_type=adv_type,num_classes=self.num_classes,batch_size=batch_size,in_transform=in_transform)
        
    def feature_analyze(self,channel_labels):
        analyze(self.in_distribution,self.out_of_distribution,channel_labels,is_image=False,data_labels=self.data_labels,signal_frequency=signal_frequency)
        # Change to extract features and run dashboard
        
        
    # def adv_training(self):
    
    # def adv_defense(self):
    
# files = os.listdir("../../../data3/mPower/data")
# dataset = parkinsonsData(files, col=8)
transform = transforms.Compose([transforms.ToTensor()])
dataset = torchvision.datasets.CIFAR10(".",download=True,transform=transform)
model = torchvision.models.densenet121(pretrained=True)
x = client(model,dataset,dataset,data_labels=["mPower","mPower"],num_classes=1)
x.attack("FGSM",gpu=0)