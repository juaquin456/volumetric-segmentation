from dataset import BrainLoader
import torch
from model import model, criterion
# from a import model, criterion
from torch.utils.data import DataLoader
from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device, flush=True)

model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=1e-5)

ds = BrainLoader("./Train", False)
N = len(ds)
train_len = int(0.9 * N)
val_len = N - train_len
train_set, val_set = torch.utils.data.random_split(ds, [train_len, val_len])

train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
val_loader = DataLoader(val_set, batch_size=4)


best_loss = 1000000

epochs = 40 
print("Starting...", flush=True)
for epoch in range(epochs):
    model.train()
    n_train = 0 
    loss_train = 0.
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        loss_train += loss.item()
        n_train += 1
    
    loss_train /= n_train    

    scheduler.step()
    model.eval()
    loss_val = 0
    with torch.no_grad():
        n_val = 0
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = criterion(output, y)
            loss_val += loss.item() 
            n_val += 1
        loss_val /= n_val 
 
    print(f"Epoch {epoch} / {epochs}: loss_train: {loss_train}, loss_val: {loss_val}", flush=True)
    if loss_val < best_loss:
        best_loss = loss_val
        model_path = f"models/{epoch}.pt"
        torch.save(model.state_dict(), model_path)