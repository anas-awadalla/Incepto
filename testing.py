from analyze import analyze
from mHealthData import mHealthData
from MotionSenseData import MotionSenseData
from oodParkinsonsData import oodParkinsonsData
from parkinsons_dataset import parkinsonsData
import os
from sklearn.model_selection import train_test_split
import torchvision
from torchvision import transforms


files = os.listdir("../../../data3/mPower/data")
transform = transforms.Compose([transforms.ToTensor()])
analyze(torchvision.datasets.CIFAR10("./",download=True, transform=transform),[torchvision.datasets.CIFAR100("./", transform=transform, download=True)], channel_labels=["x","y","z"], signal_frequency=0.05, is_image=True,data_labels=["Cifar10","ImageNet"])