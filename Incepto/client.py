from analyze import analyze
from OOD_Generate_Mahalanobis import extract_features
from OOD_Regression_Mahalanobis import train_detector
from ADV_Samples import generate_adv_samples
from datasets.parkinsons_dataset import parkinsonsData
from models.parkinsonsNet import Network
from Incepto.gram_ood import gram_metrics
import os
import torch
import torchvision
from torchvision.transforms import transforms
import calMetric as m
import calData as d

class client():
    def __init__(self, model, in_distribution, out_of_distribution, data_labels, num_classes, is_image=False, transformation=None):
        self.in_distribution = in_distribution
        self.out_of_distribution = out_of_distribution
        self.model = model
        self.num_classes = num_classes
        self.in_transform = transformation
        self.data_lables = data_labels
        self.is_image=is_image
        
    def analyze_data(self, channel_labels, signal_frequency=None):
        if self.is_image:
            analyze(model=self.model, in_distribution=self.in_distribution,out_of_distribution=self.out_of_distribution,channel_labels=channel_labels,is_image=True,data_labels=self.data_lables)
        else:
            analyze(model=self.model, in_distribution=self.in_distribution,out_of_distribution=self.out_of_distribution,channel_labels=channel_labels,signal_frequency=200,data_labels=self.data_lables)
        
    def detect_ood_mahalanobis(self, model, gpu, batch_size):
        extract_features(model, self.in_distribution, self.data_labels[0], self.data_labels[1:], self.out_of_distribution, self.in_transform, gpu=gpu, batch_size=batch_size, num_classes=self.num_classes)
        train_detector([self.data_labels[0]], self.data_labels[1:])
        
    def detect_ood_gram(self, model, gpu, batch_size):
        gram_metrics(model,self.data_labels[0],self.in_distribution,self.data_labels[1:], self.out_of_distribution,gpu=gpu,batch_size=batch_size)
    
    def detect_ood_odin(self, gpu, nnName="model", epsilon= 0.0014, temperature = 1000 ):
        net1 = self.model
        testloaderIn = torch.utils.data.DataLoader(self.in_distribution, batch_size=1,
                                            shuffle=False, num_workers=2)
        testloadersOut = []
        for dataset in self.out_of_distribution:
            testloadersOut.append(torch.utils.data.DataLoader(dataset, batch_size=1,
                                         shuffle=False, num_workers=2))
        net1.cuda(gpu)
        d.testData(net1, torch.nn.CrossEntropyLoss(), gpu, testloaderIn, testloadersOut, nnName, self.data_lables[0], epsilon, temperature) 
        m.metric(nnName, self.data_lables[0])
        
        
        

        
           
# files = os.listdir("../../../data3/mPower/data")
# dataset = parkinsonsData(files, col=8)
transform = transforms.Compose([transforms.Resize(32),transforms.ToTensor(),])
dataset = torchvision.datasets.CIFAR10(".",download=True,transform=transform)
model = torchvision.models.resnet.ResNet
checkpoint = torch.load("/home/anasa2/Incepto/pre_trained/kaggle90-best.pth") 
model = checkpoint.cpu()
print(model)
ben = torchvision.datasets.ImageFolder(root='/home/anasa2/Incepto/test_data/', transform=transform)
mal = torchvision.datasets.ImageFolder(root='/home/anasa2/Incepto/mal_test_data/', transform=transform)

# model = torch.load("/home/anasa2/Incepto/pre_trained/parkinsonsNet-outbound_mpower-outbound.pth")
x = client(model,mal,[dataset,ben],data_labels=["HAMmal","Cifar10","HAMBenign"],num_classes=2,is_image=True)
x.analyze_data(channel_labels=["r","g","b"])