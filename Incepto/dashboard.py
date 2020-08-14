
from collections.abc import Iterable
from tqdm import tqdm
import torch
from matplotlib import pyplot as plt
import cv2
from PIL import Image
import numpy as np
import param
import pandas as pd 
import numpy as np
import seaborn as sns
import panel as pn
from visualize_filters_maps import visualize_maps_features_2d, visualize_maps_features_1d
from visualize_filters_maps import visualize_maps_filters_2d, visualize_maps_filters_1d
from class_activation_mapping import get_cam
from guided_backprop import generate_gb
from torchvision import transforms

class DashboardDataElements(param.Parameterized):
    
        def filter_map(self, model):
            return visualize_maps_filters_2d(list(model.children()))
        
        def feature_map(self, model, img):
            return visualize_maps_features_2d(list(model.children()), img)
        
        def pixel_dist_img(self, image):
            print("Generating Pixel Distribution Histogram...")
            fig = plt.figure()
            # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

            # print("Calculating Variance of Laplacian Operators...")
            # plt.title("Grayscale Histogram")
            plt.ylabel("Count")
            plt.xlabel("Intensity Value")
            # plt.plot(hist)
            # plt.xlim([0, 256])
            ax = plt.hist(image.ravel(), bins = 256)
            vlo = cv2.Laplacian(image, cv2.CV_32F).var()
            plt.text(0.5, 1, ('Variance of Laplacian: '+str(vlo)), style='italic', bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10})
        
            return fig
        
        
        def color_dist_img(self, image):
            print("Generating Color Distribution Histogram...")
            # chans = cv2.split(image)
            # colors = ("b", "g", "r")
            fig = plt.figure()
            _ = plt.hist(image.ravel(), bins = 256, color = 'orange', )
            _ = plt.hist(image[:, :, 0].ravel(), bins = 256, color = 'red', alpha = 0.5)
            _ = plt.hist(image[:, :, 1].ravel(), bins = 256, color = 'Green', alpha = 0.5)
            _ = plt.hist(image[:, :, 2].ravel(), bins = 256, color = 'Blue', alpha = 0.5)
            _ = plt.xlabel('Intensity Value')
            _ = plt.ylabel('Count')
            _ = plt.legend(['Total', 'Red_Channel', 'Green_Channel', 'Blue_Channel'])
            
            
            
            # plt.title("'Flattened' Color Histogram")
            # plt.xlabel("Bins")
            # plt.ylabel("# of Pixels")
            # features = []
            # # loop over the image channels
            # for (chan, color) in zip(chans, colors):
            #     # create a histogram for the current channel and
            #     # concatenate the resulting histograms for each
            #     # channel
            #     hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
            #     features.extend(hist)
            #     # plot the histogram
            #     plt.plot(hist, color = color)
            #     plt.xlim([0, 256])
           
            return fig
            
        
        def multi_dem_color_hist(self, image):
            print("Generating Multi-deminsional Color Histograms...")
            chans = cv2.split(image)
            fig = plt.figure()
            fig.tight_layout(pad=3.0)

            # plot a 2D color histogram for green and blue
            ax = fig.add_subplot(131)
            hist = cv2.calcHist([chans[1], chans[0]], [0, 1], None,
                [0, 1], [0, 1, 0, 1])
            p = ax.imshow(hist, interpolation = "nearest")
            ax.set_title("2D Color Histogram for Green and Blue")
            plt.colorbar(p)
            # plot a 2D color histogram for green and red
            ax = fig.add_subplot(132)
            hist = cv2.calcHist([chans[1], chans[2]], [0, 1], None,
                [0, 1], [0, 1, 0, 1])
            p = ax.imshow(hist, interpolation = "nearest")
            ax.set_title("2D Color Histogram for Green and Red")
            plt.colorbar(p)
            # plot a 2D color histogram for blue and red
            ax = fig.add_subplot(133)
            hist = cv2.calcHist([chans[0], chans[2]], [0, 1], None,
                [0, 1], [0, 1, 0, 1])
            p = ax.imshow(hist, interpolation = "nearest")
            ax.set_title("2D Color Histogram for Blue and Red")
            plt.colorbar(p) 
            plt.close()
            return fig
        
        def plt_energy_spec(self, image, signal_frequency, channel_labels):
            print("Generating Energy Spectrum...")
            chans = cv2.split(image)
            # print(chans)
            plot = plt.figure()
            plt.title("Energy")
            for i in range(len(chans[0][0])):
                plot.add_subplot(111).magnitude_spectrum(chans[0][0][i], Fs=signal_frequency, scale='dB', color=('C'+str(i)))           
            if channel_labels is not None:
                plot.legend(channel_labels)
            plt.close()
            return plot

        
        def plt_power_spec(self, image, signal_frequency, channel_labels):
            print("Generating Power Spectrum...")
            chans = cv2.split(image)
            # print(chans)
            plot = plt.figure()
            plt.title("Power")
        
            for i in range(len(chans[0][0])):
                data = chans[0][0][i]
                ps = np.abs(np.fft.fft(data))**2

                freqs = np.fft.fftfreq(data.size, signal_frequency)
                idx = np.argsort(freqs)

                plot.add_subplot(111).plot(freqs[idx], ps[idx], color=('C'+str(i)))
            
            plt.xlabel("Frequency")
            plt.ylabel("Power")
            if channel_labels is not None:
                plot.legend(channel_labels)
            plt.close()
            return plot
                
        def mean_plot(self, image, channel_labels):
            print("Calculating Min/Max/Mean/StDev for Channels...")
            chans = cv2.split(image)
            mean = []
            for i in zip(chans[0][0]):
                mean.append(np.mean(np.asarray(i)))
            df = pd.DataFrame({"Channels":channel_labels,"Mean":mean})
            return df
        
        def min_plot(self, image, channel_labels):
            chans = cv2.split(image)
            min = []
            for i in zip(chans[0][0]):
                min.append(np.min(np.asarray(i)))
            df = pd.DataFrame({"Channels":channel_labels,"Min":min})
            return df
                
        def max_plot(self, image, channel_labels):
            chans = cv2.split(image)
            max = []
            for i in zip(chans[0][0]):
                max.append(np.max(np.asarray(i)))
            df = pd.DataFrame({"Channels":channel_labels,"Max":max})
            return df
                
        def stdev_plot(self, image, channel_labels):
            chans = cv2.split(image)
            stdev = []
            for i in zip(chans[0][0]):
                stdev.append(np.std(np.asarray(i)))
            df = pd.DataFrame({"Channels":channel_labels,"Standard Deviation":stdev})
            return df

        def gb_plot(self, image, model):
            fig = plt.figure(figsize=(40,40))
            index=1
            for num, module_pos in zip(enumerate(model.features),model.features._modules.items()):
                num = num[0]
                module_pos = module_pos[0]
                guided_grads_img,grayscale_guided_grads,pos_sal,neg_sal = generate_gb(model,image,0,num,1)
            
                ax = fig.add_subplot(8,8,index)
                ax.imshow(np.asarray(guided_grads_img))
                ax.set_title("Guided Gradiant - "+str(module_pos))
                index +=1
                    
                ax = fig.add_subplot(8,8,index)
                ax.imshow(np.asarray(grayscale_guided_grads))
                ax.set_title("Gradient Activation - "+str(module_pos))
                index +=1
                    
                ax = fig.add_subplot(8,8,index)
                ax.imshow(np.asarray(pos_sal))
                ax.set_title("Postive Salency - "+str(module_pos))
                index +=1
                
                ax = fig.add_subplot(8,8,index)
                ax.imshow(np.asarray(neg_sal))
                ax.set_title("Negative Salency - "+str(module_pos))
                index +=1

                return fig
        
        def cam_plot(self, image, model, target_class=0):
            fig = plt.figure(figsize=(40,40))
            index=1
            for module_pos, module in model.features._modules.items():

                cam, heatmap, heatmap_on_image = get_cam(model,image,target_class,module)

                ax = fig.add_subplot(8,8,index)
                ax.imshow(np.asarray(heatmap))
                ax.set_title("Heat Map - "+str(module_pos))

                index +=1
                ax = fig.add_subplot(8,8,index)
                ax.imshow(np.asarray(cam))
                ax.set_title("Class Activation Map - "+str(module_pos))
                index +=1
                ax = fig.add_subplot(8,8,index)
                ax.imshow(np.asarray(heatmap_on_image))
                ax.set_title("Heat Map on Image - "+str(module_pos))
                index +=1

            return fig
        
        def plot_img(self, img, target=0):
            fig = plt.figure()
            ax = fig.add_subplot(131)
            ax.imshow(img)
            ax.set_title("Target: "+str(target))

            return fig