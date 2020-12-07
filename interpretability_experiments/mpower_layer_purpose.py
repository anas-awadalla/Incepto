# %%
from scipy import fft, arange, signal
import numpy as np
from numpy.fft import rfft
from numpy import argmax, mean, diff, log, nonzero
from scipy.signal import blackmanharris, correlate, find_peaks
# from parabolic import parabolic

# %%
def find_perodic(s, threshold):
    f, Pxx = signal.periodogram(s, fs = 200, window='hanning', scaling='spectrum')
    time = []
    for amp_arg in np.argsort(np.abs(Pxx))[::-1][1:20]:
        time.append(1 / f[amp_arg])
    
    return sum(time)/len(time)


# %%

def classify_frequency(signal, fs, threshold):
    """
    Estimate frequency from peak of FFT
    """
    # Compute Fourier transform of windowed signal
    windowed = signal * blackmanharris(len(signal))
    f = rfft(windowed)

    # Find the peak and interpolate to get a more accurate peak
    i = argmax(abs(f))  # Just use this for less-accurate, naive version
    true_i = parabolic(log(abs(f)), i)[0]

    # Convert to equivalent frequency
    return fs * true_i / len(windowed) > threshold

# %%
def get_peaks(signal, threshold):
    peaks, _ = find_peaks(signal)
    return len(peaks) > threshold

# %%
def __z_score(signal):
    mean_int = np.mean(signal)
    std_int = np.std(signal)
    z_scores = (signal - mean_int) / std_int
    return z_scores
    
# %%
def get_spikes(signal):
    spikes = abs(np.array(__z_score(signal))) > 3.5
    return np.sum(spikes)

# %%
def classify_amplitude(signal, threshold):
    return np.max(signal) > threshold

# %%
def classify_mean(signal, threshold):
    print(np.average(signal,axis=0))
    return np.average(signal,axis=0) > threshold


# %%
def classify_stdev(signal, threshold):
    return np.std(signal,axis=0) > threshold


# %%
def classify_zero_crossing(signal, threshold):
    return np.nonzero(np.diff(signal > 0))[0].size > threshold

# %%
from scipy.stats import skew 

def classify_skewness(signal):
    return skew(signal) > 0

# %%
from scipy.stats import kurtosis

def classify_kurtosis(signal):
    return kurtosis(signal) < 0

# %%
from scipy.signal.spectral import periodogram, welch

def spectral_entropy(x, sf, method='fft', nperseg=None, normalize=False):
    x = np.array(x)
    # Compute and normalize power spectrum
    if method == 'fft':
        _, psd = periodogram(x, sf)
    elif method == 'welch':
        _, psd = welch(x, sf, nperseg=nperseg)
    psd_norm = np.divide(psd, psd.sum())
    se = -np.multiply(psd_norm, np.log2(psd_norm)).sum()
    if normalize:
        se /= np.log2(psd_norm.size)
    return se

def classify_entropy(signal, threshold=0):
    print(spectral_entropy(signal,200))
    return spectral_entropy(signal,200) > threshold

# %%
