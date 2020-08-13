import cv2 as cv
import matplotlib.pyplot as plt
import torchvision.transforms as t
from extractor import Extractor
import numpy as np


def visualize_maps_features_1d(model, img):
    fig = plt.figure(figsize=(30, 30))

    extractor = Extractor(model)
    extractor.activate()

    featuremaps = [extractor.CNN_layers[0](img)]
    for x in range(1, len(extractor.CNN_layers)):
        featuremaps.append(extractor.CNN_layers[x](featuremaps[-1]))

    for x in range(len(featuremaps)):
        layers = featuremaps[x][0, :, :].detach()
        for i, filter in enumerate(layers):
            if i == 64:
                break
            index = 130 +i+2
            ax = fig.add_subplot(index)
            ax.plot(filter)
            ax.axis('off')
            ax.set_title("feature map: "+str(x)+" filter: "+str(i))

    return fig


def visualize_maps_features_2d(model, img):
    fig = plt.figure(figsize=(30, 30))

    
    extractor = Extractor(model)
    extractor.activate()

    img = cv.cvtColor(np.float32(img), cv.COLOR_BGR2RGB)
    img = t.Compose([
        t.ToPILImage(),
        t.Resize((128, 128)),
        t.ToTensor(),
        t.Normalize(0.5, 0.5)])(np.uint8(img)).unsqueeze(0)
    featuremaps = [extractor.CNN_layers[0](img)]
    for x in range(1, len(extractor.CNN_layers)):
        featuremaps.append(extractor.CNN_layers[x](featuremaps[-1]))

    for x in range(len(featuremaps)):
        layers = featuremaps[x][0, :, :, :].detach()
        for i, filter in enumerate(layers):
            if i == 64:
                break
            ax = fig.add_subplot(8, 8, i + 1)
            ax.imshow(filter)
            ax.axis('off')
            ax.set_title("feature map: "+str(x)+" filter: "+str(i))

    return fig


def visualize_maps_filters_1d(model):
    fig = plt.figure(figsize=(30, 30))

    extractor = Extractor(model)
    extractor.activate()

    for index, filter in enumerate(extractor.CNN_weights[0]):
        ax = fig.add_subplot(8, 8, index + 1)
        ax.plot(filter[0, :].cpu().detach())
        ax.axis('off')
        ax.set_title("filter: "+str(index), pad=10)
    return fig


def visualize_maps_filters_2d(model):
    fig = plt.figure(figsize=(30, 30))

    extractor = Extractor(model)
    extractor.activate()

    for index, filter in enumerate(extractor.CNN_weights[0]):
        ax = fig.add_subplot(8, 8, index + 1)
        ax.imshow(filter[0, :, :].cpu().detach())
        ax.axis('off')
        ax.set_title("filter: "+str(index), pad=10)
    return fig