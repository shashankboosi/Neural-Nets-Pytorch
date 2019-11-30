# convert.py
import torch
from src.stage2.BiLSTM import Network

device = torch.device('cpu')
net = Network().to(device)
print(f'Loading model..')
net.load_state_dict(torch.load('model.pth', map_location=torch.device(device)))
print(f'Saving model as {device}.')
torch.save(net.state_dict(), './model.pth')