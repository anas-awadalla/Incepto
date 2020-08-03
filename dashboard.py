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

class DashboardDataElements(param.Parameterized):
        def pixel_dist_img(self, image):
            print("Generating Pixel Distribution Histogram...")
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cv2.imshow("gray", gray)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            fig = plt.figure()
            print("Calculating Variance of Laplacian Operators...")
            vlo = cv2.Laplacian(gray, cv2.CV_64F).var()
            plt.text(3, 8, ('Variance of Laplacian: '+str(vlo)), style='italic', bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
            plt.title("Grayscale Histogram")
            plt.xlabel("Bins")
            plt.ylabel("# of Pixels")
            plt.plot(hist)
            plt.xlim([0, 256])
            return fig
        
        
        def color_dist_img(self, image):
            print("Generating Color Distribution Histogram...")
            chans = cv2.split(image)
            colors = ("b", "g", "r")
            fig = plt.figure()
            plt.title("'Flattened' Color Histogram")
            plt.xlabel("Bins")
            plt.ylabel("# of Pixels")
            features = []
            # loop over the image channels
            for (chan, color) in zip(chans, colors):
                # create a histogram for the current channel and
                # concatenate the resulting histograms for each
                # channel
                hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
                features.extend(hist)
                # plot the histogram
                plt.plot(hist, color = color)
                plt.xlim([0, 256])
           
            return fig
            
        
        def multi_dem_color_hist(self, image):
            print("Generating Multi-deminsional Color Histograms...")
            chans = cv2.split(image)
            fig = plt.figure()
            # plot a 2D color histogram for green and blue
            ax = fig.add_subplot(131)
            hist = cv2.calcHist([chans[1], chans[0]], [0, 1], None,
                [32, 32], [0, 256, 0, 256])
            p = ax.imshow(hist, interpolation = "nearest")
            ax.set_title("2D Color Histogram for Green and Blue")
            plt.colorbar(p)
            # plot a 2D color histogram for green and red
            ax = fig.add_subplot(132)
            hist = cv2.calcHist([chans[1], chans[2]], [0, 1], None,
                [32, 32], [0, 256, 0, 256])
            p = ax.imshow(hist, interpolation = "nearest")
            ax.set_title("2D Color Histogram for Green and Red")
            plt.colorbar(p)
            # plot a 2D color histogram for blue and red
            ax = fig.add_subplot(133)
            hist = cv2.calcHist([chans[0], chans[2]], [0, 1], None,
                [32, 32], [0, 256, 0, 256])
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

