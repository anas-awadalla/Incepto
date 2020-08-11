import cv2 as cv
import matplotlib.pyplot as plt
import torchvision.transforms as t
from extractor import Extractor


def visualize_maps_features(model, image_path):

    extractor = Extractor(model)
    extractor.activate()

    # Filter Map
    img = cv.cvtColor(cv.imread(image_path), cv.COLOR_BGR2RGB)
    img = t.Compose([
        t.ToPILImage(),
        t.Resize((128, 128)),
        # t.Grayscale(),
        t.ToTensor(),
        t.Normalize(0.5, 0.5)])(img).unsqueeze(0)

    featuremaps = [extractor.CNN_layers[0](img)]
    for x in range(1, len(extractor.CNN_layers)):
        featuremaps.append(extractor.CNN_layers[x](featuremaps[-1]))

    # Visualising the featuremaps
    for x in range(len(featuremaps)):
        plt.figure(figsize=(30, 30))
        layers = featuremaps[x][0, :, :, :].detach()
        for i, filter in enumerate(layers):
            if i == 64:
                break
            plt.subplot(8, 8, i + 1)
            plt.imshow(filter, cmap='gray')
            plt.axis('off')

        # plt.savefig('featuremap%s.png'%(x))

    plt.show()

def visualize_maps_filters(model, image_path):
    
    extractor = Extractor(model)
    extractor.activate()

    # Filter Map
    img = cv.cvtColor(cv.imread(image_path), cv.COLOR_BGR2RGB)
    img = t.Compose([
        t.ToPILImage(),
        t.Resize((128, 128)),
        # t.Grayscale(),
        t.ToTensor(),
        t.Normalize(0.5, 0.5)])(img).unsqueeze(0)

    # Visualising the filters
    plt.figure(figsize=(35, 35))
    for index, filter in enumerate(extractor.CNN_weights[0]):
        plt.subplot(8, 8, index + 1)
        plt.imshow(filter[0, :, :].detach(), cmap='gray')
        plt.axis('off')
    plt.show()