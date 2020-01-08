import torch.nn.functional as F
import torch
from torch_geometric.data import Data, DataLoader
from risk_gnn import Net

import progressbar

import pickle
with open("dataset2.p", "rb") as f:
        data_list = pickle.load(f)
        
print("loaded dataset")

loader = DataLoader(data_list, batch_size=128, shuffle=True)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#model = Net()
model = torch.load('model.pt')
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(200):
    print("epoch {}".format(epoch))
    
    for data in progressbar.progressbar(loader):
        data = data.to(device)
        
        optimizer.zero_grad()
        out = model(data)
        out = out.exp()
        #loss = F.nll_loss(out, data.y)
        loss = F.mse_loss(out, data.y)
        loss.backward()
        optimizer.step()
        
        
    torch.save(model, 'model2.pt')
    
    
model.eval()

for data in loader:
    
    data = data.to(device)
    out = model(data)
    out = out.cpu()
    
    
    for p,y in zip(out, data.y):
        #print(list(one_hot_encoding(y, 3)))
        print(y.cpu().detach().numpy().ravel())
        print(list(p.exp().detach().numpy().ravel()))
        print()
    
    break









    



