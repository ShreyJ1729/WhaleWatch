import os
import numpy as np
import random

animals = os.listdir("processed")
if not os.path.isdir("train"):
    os.mkdir("train")
if not os.path.isdir("test"):
    os.mkdir("test")
for animal in animals:
    if not os.path.isdir("train/"+animal):
        os.mkdir("train/"+animal)
    if not os.path.isdir("test/"+animal):
        os.mkdir("test/"+animal)

split=0.8
from tqdm import tqdm
for animal in tqdm(animals):
    paths = ["processed/"+animal+"/"+i for i in os.listdir("processed/"+animal)]
    for path in paths:
        # nasty solution to bash script problem
        path2=path.replace("'", "").replace("(", "").replace(")", "")
        tqdm.write(f"executing command [cp \"{path}\" 'folder/{animal}/{path2.split('/')[-1]}']")
        if random.randint(0,100)/100>0.8:
            os.system(f"cp \"{path}\" \"test/{animal}/{path2.split('/')[-1]}\"")
        else:
            os.system(f"cp \"{path}\" \"train/{animal}/{path2.split('/')[-1]}\"")