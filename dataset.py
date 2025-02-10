import os
import torch
import numpy as np
from torch.utils.data import Dataset
import nibabel as nib

class BrainLoader(Dataset):
    def __init__(self, foldername, one_hot = True):
        self.foldername = foldername
        self.items = list(os.listdir(self.foldername))
        self.suffixes = ["_flair.nii.gz", "_t1.nii.gz", "_t1ce.nii.gz", "_t2.nii.gz"]
        self.ysuffix = "_seg.nii.gz"
        self.one_hot = one_hot
    def __len__(self):
        return len(self.items)
    def __getitem__(self, index):
        name = self.items[index]
        item_folder = os.path.join(self.foldername, name)

        xnames = [name + sfx for sfx in self.suffixes]
        yname = name + self.ysuffix
        x = []
        for xname in xnames:
            x.append(nib.load(os.path.join(item_folder, xname)).get_fdata())
        yimg = nib.load(os.path.join(item_folder, yname)).get_fdata()
        ytensor = torch.IntTensor(yimg)
        ytensor[ytensor == 4] = 3
        if self.one_hot:
            ytensor = torch.nn.functional.one_hot(ytensor, 4).permute((3, 0, 1, 2))
        return torch.FloatTensor(np.array(x)), ytensor 


