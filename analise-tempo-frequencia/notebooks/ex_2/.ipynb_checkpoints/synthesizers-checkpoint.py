import numpy as np

class Sinusoidal:
    def __init__():
        pass
    
    def __init__(self, track_array, buffer_size, n_frames):
        self.track_array = track_array
        self.n_frames = n_frames
        self.buffer_size = buffer_size
    
    def run(self):
        signal = np.zeros((self.n_frames,self.buffer_size))
        for i, track in enumerate(self.track_array):
            start_frame = track.initial_frame
            
            amplitudes = self._pad_array(start_frame, track.amplitude, self.n_frames)
            ang_frequencies = 2*np.pi*self._pad_array(start_frame, track.frequency, self.n_frames)
            
#             amplitudes = self.linear_interpolate(amplitudes)
#             phases = self.cubic_interpolate(ang_frequencies)
            
            frame_windows = np.tile(range(self.buffer_size), (self.n_frames, 1))
#             frame_windows = np.array(range(self.n_frames*self.buffer_size)) % self.buffer_size
#             frame_windows = frame_windows.reshape((self.n_frames, self.buffer_size))
            frame_windows = np.transpose(frame_windows)
            phases_matrix = [ang_frequencies*window for window in frame_windows]
#             phases_matrix = [phases*window for window in frame_windows]
            amplitudes_matrix = [amplitudes*window for window in frame_windows]

            
            oscilators = [amplitudes*np.sin(phases) 
                          for (amplitudes, phases) in zip(amplitudes_matrix, phases_matrix)]
            
            oscilators = np.transpose(oscilators)
            signal += oscilators
        return signal
        return np.concatenate(signal, axis=1)
        
    def linear_interpolate(self, array):
        lagged_array = np.roll(array, 1)
        lagged_array[0] = lagged_array[1]
        
        diff = (array - lagged_array)/self.buffer_size
        
        return lagged_array + diff
        
    def cubic_interpolate():
        theta_array = np.zeros(self.n_frames)
        raise NotImplementedError  
#     def cubic_interpolate:
#         def alpha(theta, theta_lag, omega, omega_lag):
#             return (3/(self.buffer_size**2))*(theta - theta_lag)
            
    def _pad_array(self, start_frame, array, n_frames):
        return np.concatenate([np.zeros(start_frame),
                          array,
                          np.zeros(n_frames - (start_frame + len(array)))])
                    
    
class spectral:
    def __init__():
        pass
    
    