from __future__ import division

import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from scipy.signal import decimate, hanning, convolve, spectrogram
import numpy as np

def discrete_derivative(x, fs):
    y = np.zeros_like(x);
    for i in range(1,len(x)-1):
            y[i] = (x[i] - x[i-1])*fs
    return y


def read_audio(filepath):
    return wav.read(filepath)

def plot_spec(Sxx, f, t,audio_name,ax):
    h = ax.matshow(Sxx, 
               interpolation="nearest",
               extent=[f.min(), f.max(), t.min(), t.max()],
               aspect="auto",
               origin = 'lower',
               cmap="jet")
    ax.xaxis.tick_bottom()
    cbar = plt.colorbar(h, ax = ax)
    cbar.ax.set_ylabel("Magnitude (dB)")
    ax.set_title(audio_name[:-4].encode(encoding='UTF-8'))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    
    
def processSpec(Sxx, win, wlen, thres):
    C = sum(win)/wlen;
    power = np.absolute(Sxx)/wlen/C
    power = power / tpsw(power)
    power = 20*np.log10(power + 1e-6)
    power[power < thres] = 0
    
    return power


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

    
    
class Track:
    def __init__(self, initial_frame, first_freq, first_amplitude):
        self.initial_frame = initial_frame
        self.frequency = [first_freq]
        self.amplitude = [first_amplitude]
        self.final_frame = -1
        
    def getFinalFrame(self) :
        return self.final_frame
    
    def getInitialFrame(self) :
        return self.initial_frame
    
    def setFinalFrame(self,value) :
        self.final_frame = value
                
    def getFrequency(self) :
        return self.frequency
    
    def matchFrequency(self, peak_freq, diff):
        if abs((self.frequency[len(self.frequency) - 1]) - peak_freq) < diff:
            return True
        return False
    
    def append_frame(self,freq, ampl, i_frame):
        last_frame = len(self.frequency) + self.initial_frame - 1
        if last_frame < i_frame:
            self.frequency = self._zero_pad(self.frequency, i_frame - last_frame)
            self.amplitude = self._zero_pad(self.amplitude, i_frame - last_frame)
        np.append(self.frequency, freq)
        np.append(self.amplitude, ampl)
        
    def _setFinalFrame(self):
        self.final_frame = len(self.frequency) + self.initial_frame
    
    def closeTrack(self):
        self._setFinalFrame()
        
    def _zero_pad(self, array, n_frames):
        return np.concatenate([array,
                          np.zeros(n_frames)])
    
def evaluate_closest_track(frequency,listOfTracks,freq_dist_thresh,frame,freq_list) :
    winner = -1
    dif = 28000
    for track in range(0,len(listOfTracks)-1):
        Track_analyzed = listOfTracks[track]
        if ((Track_analyzed.getFinalFrame() < 0) and (Track_analyzed.getInitialFrame() < frame) and (len(Track_analyzed.getFrequency()) + Track_analyzed.getInitialFrame() - 1 != frame)):
            n_dif = abs(freq_list[frequency] - (listOfTracks[track].getFrequency())[frame-1-listOfTracks[track].getInitialFrame()])         
            dif = min(n_dif,dif)
            if (dif == n_dif) and (dif/(listOfTracks[track].getFrequency())[frame-1-listOfTracks[track].getInitialFrame()] <= freq_dist_thresh):
                winner = track
    return (winner,dif)
    
    
    
    
    
    
    
    
    
    
    
    
    
    