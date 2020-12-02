# %%
from scipy import fft, arange, signal
import numpy as np
from numpy.fft import rfft
from numpy import argmax, mean, diff, log, nonzero
from scipy.signal import blackmanharris, correlate, find_peaks
from parabolic import parabolic

# %%
def find_perodic(signal, threshold):
    f, Pxx = signal.periodogram(signal, fs = 200, window='hanning', scaling='spectrum')
    for amp_arg in np.argsort(np.abs(Pxx))[::-1][1:20]:
        time = 1 / f[amp_arg]
        if time > threshold:
            return True
        
    return False

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
    peaks, _ = find_peaks(signal, distance=150)
    return len(peaks) > 4

# %%
def __z_score(signal):
    mean_int = np.mean(signal)
    std_int = np.std(signal)
    z_scores = (signal — mean_int) / std_int
    return z_scores
    
# %%
def get_spikes(signal):
    spikes = abs(np.array(__z_score(signal))) > 3.5
    return np.sum(spikes)

# %%
def classify_amplitude(signal, threshold):
    return np.max(signal) > threshold

# %%
def classify_power(signal, threshold):
    pass

# %%
def classify_energy(signal, threshold):
    pass

# %%
def classify_ramp(signal, threshold):
    pass

# %%
def classify_step(signal, threshold):
    pass

# %%
def classify_pulse(signal, threshold):
    pass
