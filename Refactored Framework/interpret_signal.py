from __future__ import print_function

from matplotlib._color_data import CSS4_COLORS
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from captum.attr import GuidedBackprop
from misc_functions import (get_example_params,
                            convert_to_grayscale,
                            save_gradient_images,
                            get_positive_negative_saliency)

class IllegalDataDimensionException(Exception):
    pass


class SignalInterpreter(object):
    """Summary of class here.

        Longer class information....
        Longer class information....

        Attributes:
            likes_spam: A boolean indicating if we like SPAM or not.
            eggs: An integer count of the eggs we have laid.
        """

    def __init__(self, model, X, y, gpu, channel_labels):
        """

        Args:
            model:
            dataset:
            gpu:
            channel_labels:
        """
        # data_loader = DataLoader(dataset, batch_size=len(dataset))
        # itr = iter(data_loader)
        self.X, self.y = X, y
        
        self.algo = GuidedBackprop(model)
        
        
        self.model = model.eval()

        if len(self.X.shape) != 3:
            raise IllegalDataDimensionException("Expected data to have dimensions 3 but got dimension ",
                                                str(len(self.X.shape)))

        if len(self.X.shape) != len(channel_labels):
            raise IllegalDataDimensionException("Expected channel_labels to contain ", str(len(self.X.shape)),
                                                " labels but got ", str(len(channel_labels)), " labels instead.")

        self.channel_labels = channel_labels

        if gpu == -1:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:' + str(gpu))
            
        model.to(self.device)

        counter = 0
        self.modules = {}
        for top, module in model.named_children():
            if type(module) == torch.nn.Sequential:
                self.modules.update({"" + top: module})
                counter += 1

        print(f"Total Interpretable layers: {counter}")

        self.X.to(self.device)

        layer_map = {0: self.X.flatten(start_dim=1)}
        index = 1
        output = self.X

        for i in self.modules:
            layer = self.modules[i]
            layer.eval().to(self.device)
            output = layer(output)
            layer_map.update({index: output.flatten(start_dim=1)})
            index += 1

        assert counter == len(layer_map) - 1
        print("Intermediate Forward Pass Through " + str(len(layer_map) - 1) + " Layers")

        print("Generating PCA...")
        self.pca_layer_result = []

        self.pca = PCA(n_components=3)
        self.pca_result = self.pca.fit_transform(layer_map[len(layer_map) - 1].detach().numpy())
        for i in range(len(channel_labels)):
            self.pca_layer_result.append(self.pca_result[:,i])
        
        print("Generation Complete")

    def interpret_signal(self, signal):
        """

        Args:
            signal:

        Returns:

        """
        
        output = signal.unsqueeze(0).to(self.device)

        for i in self.modules:
            layer = self.modules[i]
            layer.eval().to(self.device)
            output = layer(output)
            
        output = torch.flatten(output,start_dim=1)
        
        reduced_signal = self.pca.transform(output.detach())
        example_index = self.__closest_point(reduced_signal)
        example_signal = self.X[example_index]
        
        signal_attribution = self.algo.attribute(signal.unsqueeze(0), target=0).detach().cpu().numpy()[0]
        example_signal_attribution = self.algo.attribute(example_signal.unsqueeze(0), target=0).detach().cpu().numpy()[0]
        
        signal_attribution = convert_to_grayscale(signal_attribution* signal.detach().numpy())
        example_signal_attribution = convert_to_grayscale(example_signal_attribution* example_signal.detach().numpy())
        
        # save_gradient_images(grayscale_guided_grads, result_dir + '_Guided_BP_gray')
        
        return self.__plot_signals([signal, example_signal], [signal_attribution, example_signal_attribution]
                                   , int(torch.round(torch.sigmoid(self.model(signal.unsqueeze(0).to(self.device)))).item())
                                   , self.y[example_index].item(), self.channel_labels)

    def __closest_point(self, point):
        """

        Args:
            point:
            points:

        Returns:

        """
        points = np.asarray(self.pca_result)
        deltas = points - point
        dist_2 = np.einsum('ij,ij->i', deltas, deltas)
        return np.argmin(dist_2)

    def __plot_signals(self, signals, attribution, output_test, output_training, channel_labels):
        """

        Args:
            signals:
            channel_labels:

        Returns:
            object:

        """
        colors = list(CSS4_COLORS.keys())
        i = 0
        fig = plt.figure(figsize=(40, 40))
        ax = fig.add_subplot(4, 4, i + 1)
        i += 1

        color_index = 0
        ax.set_title("Interpreted Signal with Output Class "+str(output_test))
        # for channel, label in zip(signals[0], channel_labels):
        ax.plot(sum(signals[0]))
        ax.plot(sum(attribution[0].detach().numpy()))
        
            # color_index += 1
            
        plt.legend()

        ax = fig.add_subplot(4, 4, i + 1)
        i += 1

        color_index = 0
        ax.set_title("Example Signal with Output Class "+str(output_training))
        # for channel, label in zip(signals[1], channel_labels):
        #     ax.plot(channel, color=colors[color_index + 20], label=label)
        #     color_index += 1
        ax.plot(sum(signals[1].detach().numpy()))
        ax.plot(sum(attribution[1].detach().numpy()))

        plt.legend()
        plt.show()
        return plt