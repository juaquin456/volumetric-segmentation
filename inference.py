import torch
from torch.utils.data import DataLoader
from model import model
from dataset import BrainDataset
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

testDataset = BrainDataset(rootpath='./Test_', transform=None)
testLoader  = DataLoader(testDataset, batch_size=1, shuffle=False)

model.to(device)
model.load_state_dict(torch.load("86.pt", weights_only=True, map_location=device))
model.eval() 
predictions = {}

with torch.no_grad():
    for i, data in enumerate(testLoader):
        voxel = data['voxel'].to(device)  # [B, chan, D, H, W]
        pred = model(voxel)
        pred = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)  # [D, H, W]
        predictions[f'sample_{i}'] = pred
torch.save(predictions, 'prediction.pt')