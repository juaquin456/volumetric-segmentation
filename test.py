import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from torch.utils.data import DataLoader
from dataset import BrainLoader

ds = BrainLoader("Train")
dl = DataLoader(ds, 4, shuffle=True)
sum = torch.zeros(4) 
sum_sqr = torch.zeros(4) 
n = 0
for x, y in dl:
    tmp = torch.sum(x, axis=(0, 2, 3, 4))
    sum += tmp
    sum_sqr += torch.sum(x**2, axis=(0, 2, 3, 4))
    n += x.shape[0] 

mean = sum/n 
std = torch.sqrt((sum_sqr/ n) - mean**2)

print("Mean:", mean)
print("Std:", std)