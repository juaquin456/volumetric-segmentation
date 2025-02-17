import os
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

class BrainDataset(Dataset):
    def __init__(self, rootpath, transform=None):
        self.transform = transform
        self.samples = [
            os.path.join(rootpath, r)
            for r in os.listdir(rootpath)
            if os.path.isdir(os.path.join(rootpath, r))
        ]

    def __len__(self):
        return len(self.samples)

    def _loadTensor(self, filepath):
        data = nib.load(filepath).get_fdata().astype(np.float32)
        tensor = torch.tensor(data).permute(2, 0, 1)  # [H, W, D] -> [D, H, W]
        return tensor

    def __getitem__(self, idx):
        folder = self.samples[idx]

        flair = seg = t1 = t1ce = t2 = None
        for file in os.listdir(folder):
            mode = file.split("_")[-1]
            filepath = os.path.join(folder, file)
            if mode == "flair.nii.gz":
                flair = self._loadTensor(filepath)
            elif mode == "seg.nii.gz":
                seg = self._loadTensor(filepath)
            elif mode == "t1.nii.gz":
                t1 = self._loadTensor(filepath)
            elif mode == "t1ce.nii.gz":
                t1ce = self._loadTensor(filepath)
            elif mode == "t2.nii.gz":
                t2 = self._loadTensor(filepath)
        
        voxel = torch.stack([flair, t1, t1ce, t2], dim=0)  # [chan, D, H, W]
        
        if seg is not None:
            seg[seg == 4] = 3
        
        return {'voxel': voxel, 'mask': seg}