from torch_geometric.nn import global_add_pool
import torch.nn.functional as F
import torch
from torch.nn import Sequential as Seq, Linear as Lin, ReLU

from torch_geometric.nn import GCNConv


num_global_features = 6
num_node_features = 6
num_players = 3


def MLP(a):
    F_in, F_out = a
    return Seq(Lin(F_in, F_out), ReLU() )

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.mlp1 = Seq(Lin(num_node_features + num_global_features, 32), ReLU() )
        #self.mlp1 = Seq( MLP([num_node_features + num_global_features, 32]), MLP([32, 64]), MLP([64, 32]))
        #self.mlp2 = Seq( MLP([32, 32]), MLP([32, 64]), MLP([64, 32]))
        #self.mlpout = Seq( MLP([32, 64]), MLP([64, 32]), Lin(32, num_players))
        
        self.conv1 = GCNConv(32, 32)
        self.conv2 = GCNConv(32, 32)
        self.conv3 = GCNConv(32, num_players)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch = data.batch
        
        #batch = torch.tensor([0 for _ in range(len(x))])
    
        
        #u = global feature vector
        #x = torch.cat([x, u], dim=1)
        
        x = self.mlp1(x)
        #u = global_mean_pool(x, batch)
        
        #tmp = torch.ones((48,1))
        #u = torch.mm(tmp, u)
        #x = torch.cat([x, u], dim=1)

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        
        #x = self.conv2(x, edge_index)
        #x = F.relu(x)
        
        #x = self.mlp2(x)
        
        x = self.conv3(x, edge_index)
        
        out = global_add_pool(x, batch)
        #out = self.mlpout(out)
        
        return F.log_softmax(out, dim=1)




Net()















