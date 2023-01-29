import cv2
import numpy as np
import os 

from os import walk

root_dir = 'datasets/SIDD'
root_dir_new = 'SIDD_NEW_512'

for (dirpath, dirnames, filenames) in walk(os.path.join(root_dir, 'train/input')):
    for filename in filenames:
        image = cv2.imread(os.path.join(dirpath, filename))

        resized_image = cv2.resize(image, (256, 256))

        cv2.imwrite(os.path.join(root_dir_new, f'train/input/{filename}'), resized_image)

for (dirpath, dirnames, filenames) in walk(os.path.join(root_dir, 'train/target')):
    for filename in filenames:
        image = cv2.imread(os.path.join(dirpath, filename))

        resized_image = cv2.resize(image, (256, 256))

        cv2.imwrite(os.path.join(root_dir_new, f'train/target/{filename}'), resized_image)

for (dirpath, dirnames, filenames) in walk(os.path.join(root_dir, 'test/input')):
    for filename in filenames:
        image = cv2.imread(os.path.join(dirpath, filename))

        resized_image = cv2.resize(image, (256, 256))

        cv2.imwrite(os.path.join(root_dir_new, f'test/input/{filename}'), resized_image)        

for (dirpath, dirnames, filenames) in walk(os.path.join(root_dir, 'test/target')):
    for filename in filenames:
        image = cv2.imread(os.path.join(dirpath, filename))

        resized_image = cv2.resize(image, (256, 256))

        cv2.imwrite(os.path.join(root_dir_new, f'test/target/{filename}'), resized_image)