import cv2 as cv
import matplotlib.pyplot as plt
import torchvision.transforms as t
from extractor import Extractor


def visualize_maps_features(model, img):
    fig = plt.figure()

    extractor = Extractor(model)
    extractor.activate()

    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = t.Compose([
        t.ToPILImage(),
        t.Resize((128, 128)),
        t.ToTensor(),
        t.Normalize(0.5, 0.5)])(img).unsqueeze(0)

    featuremaps = [extractor.CNN_layers[0](img)]
    for x in range(1, len(extractor.CNN_layers)):
        featuremaps.append(extractor.CNN_layers[x](featuremaps[-1]))

    for x in range(len(featuremaps)):
        plt.figure(figsize=(30, 30))
        layers = featuremaps[x][0, :, :, :].detach()
        for i, filter in enumerate(layers):
            if i == 64:
                break
            plt.subplot(8, 8, i + 1)
            plt.imshow(filter)
            plt.axis('off')
            plt.title("feature map: "+str(x)+" filter: "+str(i))

    return fig

def visualize_maps_filters(model):
    fig = plt.figure()
    extractor = Extractor(model)
    extractor.activate()

    plt.figure(figsize=(35, 35))
    for index, filter in enumerate(extractor.CNN_weights[0]):
        plt.subplot(8, 8, index + 1)
        plt.imshow(filter[0, :, :].detach())
        plt.axis('off')
        plt.title("filter: "+str(index))
    return fig