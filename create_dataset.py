import torch.nn.functional as F
import torch
from torch_geometric.data import Data, DataLoader
from risk_gnn import Net

data_list = board_list
loader = DataLoader(data_list, batch_size=32, shuffle=True)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(20):
    
    for data in loader:
        data = data.to(device)
        
        optimizer.zero_grad()
        out = model(data)
        out = out.exp()
        #loss = F.nll_loss(out, data.y)
        loss = F.mse_loss(out, data.y)
        loss.backward()
        optimizer.step()
        
        
    
    
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









    



