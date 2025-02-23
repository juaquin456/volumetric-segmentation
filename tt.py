import torch

tt = torch.load("prediction.pt", weights_only=False)
print(tt.values())