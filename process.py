import os
import matplotlib.pyplot as plt
import numpy as np

#for loading and visualizing audio files
import librosa
import librosa.display


audio_fpath = os.path.join("data", "SeaOtter", "68039001.wav")

signal, sr = librosa.load(audio_fpath, sr=16000)

import numpy as np
import matplotlib.pyplot as plt

n_fft = 2048
D = np.abs(librosa.stft(signal[:n_fft], n_fft=n_fft, hop_length=n_fft+1))
plt.plot(D)
plt.show()

hop_length = 512
D = np.abs(librosa.stft(signal, n_fft=n_fft,  hop_length=hop_length))
librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='linear')
plt.colorbar()
plt.show()


DB = librosa.amplitude_to_db(D, ref=np.max)
librosa.display.specshow(DB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.show()
