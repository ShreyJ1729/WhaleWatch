import os
import matplotlib.pyplot as plt
import numpy as np
import sys
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

#for loading and visualizing audio files
import librosa
import librosa.display
from tqdm import tqdm

# making it faster
import multiprocessing


# some constants
n_fft = 2048
hop_length = 512


def process_class(animal, indx, audio_clips):
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    tqdm.write(f"processing {animal} {indx}/{len(classes)} \n")
    for idx, clip in enumerate(audio_clips):
        audio_fpath = os.path.join("data", animal, clip)
        tqdm.write(f"{animal} {idx}/{len(audio_clips)}\n")

        signal, sr = librosa.load(audio_fpath, sr=16000)

        D = np.abs(librosa.stft(signal, n_fft=n_fft, hop_length=hop_length))
        DB = librosa.amplitude_to_db(D, ref=np.max)
            
        plt.axis('off')
        plt.tight_layout()
        librosa.display.specshow(DB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log')
        # plt.show()
            
        if not os.path.isdir("processed"):
            os.mkdir("processed")
        if not os.path.isdir(os.path.join("processed", animal)):
            os.mkdir(os.path.join("processed", animal))
        savefile = "".join(clip.split(".")[:-1])+".png"
        savefile = os.path.join("processed", animal, savefile)
        if not os.path.isfile(savefile):
            plt.savefig(savefile, dpi=300, bbox_inches=0, pad_inches=0)

max_num_simultaneous_processes = 5

processes = []
classes = os.listdir("data")

class_dict = {}
for animal in classes:
    class_dict[animal] = len(os.listdir(f"data/{animal}"))

class_dict = {i: class_dict[i] for i in class_dict if class_dict[i]>200}
for i in class_dict:
    print(f"{i}: {class_dict[i]}")

classes = list(class_dict.keys())

for indx in range(0, len(classes), max_num_simultaneous_processes):
    for animal in classes[indx:min(indx+max_num_simultaneous_processes, len(classes)-1)]:
        audio_clips = os.listdir(os.path.join("data", animal))
        p = multiprocessing.Process(target=process_class, args=(animal, indx, audio_clips, ))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()