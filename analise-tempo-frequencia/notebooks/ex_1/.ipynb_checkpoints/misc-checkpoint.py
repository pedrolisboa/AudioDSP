from __future__ import division

import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from scipy.signal import decimate, hanning, convolve, spectrogram
import numpy as np

def read_audio(filepath):
    return wav.read(filepath)

def plot_spec(Sxx, f, t,audio_name,ax):
    h = ax.matshow(Sxx, 
               interpolation="nearest",
               extent=[t.min(), t.max(), f.min(), f.max()],
               aspect="auto",
               origin = 'lower',
               cmap="jet")
    ax.xaxis.tick_bottom()
    cbar = plt.colorbar(h, ax = ax)
    cbar.ax.set_ylabel("Magnitude (dB)")
    ax.set_title(audio_name[:-4].encode(encoding='UTF-8'))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    


def tpsw(x, npts=None, n=None, p=None, a=None):
    if x.ndim == 1:
        x = x[:, np.newaxis]
    if npts is None:
        npts = x.shape[0]
    if n is None:
        n=int(round(npts*.04/2.0+1))
    if p is None:
        p =int(round(n / 8.0 + 1))
    if a is None:
        a = 2.0
    if p>0:
        h = np.concatenate((np.ones((n-p+1)), np.zeros(2 * p-1), np.ones((n-p+1))), axis=None)
    else:
        h = np.ones((1, 2*n+1))
        p = 1
    h /= np.linalg.norm(h, 1)

    def apply_on_spectre(xs):
        return convolve(xs, h, mode='same')

    mx = np.apply_along_axis(apply_on_spectre, arr=x, axis=0)
    indl = (x-a*mx) > 0
    x = np.where(indl, mx, x)
    mx = np.apply_along_axis(apply_on_spectre, arr=x, axis=0)
    return mx
