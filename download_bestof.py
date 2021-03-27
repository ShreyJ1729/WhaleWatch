import requests
import re
import json
from tqdm import tqdm
import os

r = requests.get("https://cis.whoi.edu/science/B/whalesounds/index.cfm")
urls = re.findall("code=[a-zA-Z0-9]+", r.text)
urls = list(set(["https://cis.whoi.edu/science/B/whalesounds/bestOf.cfm?"+i for i in urls]))

pages=[]    

print("Retrieving URL information...")
for url in tqdm(urls):
    pages.append(requests.get(url).text)

final_urls = {}
for page in pages:
    wav_urls = ["https://whoicf2.whoi.edu"+i.strip("\"") for i in re.findall("\"[/a-zA-Z0-9\.]+\.wav\"", page)]
    animal_name = re.search("<h3>Best[a-zA-Z0-9\s,\-()']+", page)
    animal_name = animal_name[0].lstrip("<h3>").rstrip("\n(").strip(" ")
    animal_name = animal_name[8:]
    print(animal_name)
    final_urls[animal_name] = wav_urls

print(json.dumps(final_urls, indent=1), file=open("final_urls.txt", "w+"))

if not os.path.isdir("data"):
    os.mkdir("data")

print("Downloading files...")

for i, name in enumerate(final_urls):
    tqdm.write(f"{i}/{len(final_urls)} Downloading {name} files...")
    url_list = list(set(final_urls[name]))
    class_dir = os.path.join("data", name)

    if not os.path.isdir(class_dir):
        os.mkdir(class_dir)

    for idx, url in enumerate(url_list):
        filepath = os.path.join(class_dir, f"audio{idx}.wav")
        if not os.path.isfile(filepath):
            r = requests.get(url)
            with open(filepath, "wb+") as f:
                f.write(r.content)