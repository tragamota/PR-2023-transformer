import os

import numpy as np
import pandas as pd

import requests

from urllib.parse import urlparse

DOWNLOAD_COUNT = 20

dataset_mapping = pd.DataFrame(np.loadtxt('SIDD_mapping.txt', dtype=str))
dataset_urls = pd.DataFrame(np.loadtxt('SIDD_URLs.txt', dtype=str))

file_names = []

for index, row in dataset_mapping.iterrows():
    url = urlparse(row[0])
    file_names.append(str(url.path.split("/")[3]))


dataset_urls = pd.concat([dataset_urls, pd.Series(file_names)], axis=1)
dataset_urls.columns = ['url', 'name']

if not os.path.exists('SIDD'):
    os.mkdir('SIDD')

if not os.path.exists('SIDD_DOWNLOAD'):
    os.mkdir('SIDD_DOWNLOAD')

srgb_dataset = dataset_urls[dataset_urls['name'].str.contains("NOISY_SRGB")].iloc[:70, :]
srgb_dataset = pd.concat([srgb_dataset, dataset_urls[dataset_urls['name'].str.contains("GT_SRGB")].iloc[:70, :]])

imageIndex = 0

for i, data in srgb_dataset.iterrows():
    print(f"Downloading subset of dataset with index {imageIndex} / {len(srgb_dataset)} with filename {data[1]}")

    if not os.path.exists(f"SIDD_DOWNLOAD/{data[1]}"):

        try:
            zip_request = requests.get(data[0], allow_redirects=True)
            open(f"SIDD_DOWNLOAD/{data[1]}", 'wb').write(zip_request.content)
        except:
            print("Assets is not found on server")

    imageIndex += 1




