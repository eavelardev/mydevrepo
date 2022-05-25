"""Play a sine signal."""
import numpy as np
import sounddevice as sd

frequency = 500
amplitude = 0.2

start_idx = 0

samplerate = sd.query_devices(None, 'output')['default_samplerate']

def callback(outdata, frames, time, status):
    global start_idx
    t = (start_idx + np.arange(frames)) / samplerate
    t = t.reshape(-1, 1)
    outdata[:] = amplitude * np.sin(2 * np.pi * frequency * t)
    start_idx += frames

with sd.OutputStream(device=None, channels=1, callback=callback,
                        samplerate=samplerate):
    print('#' * 80)
    print('press Return to quit')
    print('#' * 80)
    input()
