# import keras
from keras.models import load_model
import librosa
import librosa.display
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2

n_fft = 2048
hop_length = 512
plt.axis('off')
plt.tight_layout()
# model = load_model("model.h5")
audio_fpath = os.path.join("data", "SeaOtter", "68039001.wav")
signal, sr = librosa.load(audio_fpath, sr=16000)
D = np.abs(librosa.stft(signal, n_fft=n_fft, hop_length=hop_length))
DB = librosa.amplitude_to_db(D, ref=np.max)
librosa.display.specshow(DB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log')
plt.savefig("test.jpg", dpi=100, bbox_inches=0, pad_inches=0)

print(DB.shape)
img = cv2.imread("test.jpg")
img = cv2.resize(img, (224, 224, 3))
print(img)