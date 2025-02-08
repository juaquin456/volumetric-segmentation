import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
folder = "Train"

for file in os.listdir(folder):
    folder_obj = os.path.join(folder, file)
    for img_name in os.listdir(folder_obj):
        img_path = os.path.join(folder_obj, img_name)
        img = nib.load(img_path).get_fdata()
        print(img.shape)
    break