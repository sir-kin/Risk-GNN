import numpy as np
import networkx as nx
import copy

from torch_geometric.data import Data
import torch

from Risk_board_game import Risk

import glob
import pickle

import game_map


edge_list = list(Risk().map.edges())
# include both directions of edges
edge_list += [tuple(reversed(i)) for i in edge_list]

edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

def encode_loaded(d,y):
    x = torch.tensor(d, dtype=torch.float)
    
    d = Data(x=x, edge_index=edge_index)
    d.y = torch.tensor([y]).float()
    
    return d



file_list = glob.glob('saved_games/*')

data_list = []

for fn in file_list:
    with open(fn, "rb") as f:
        t = pickle.load(f, encoding='bytes') 
    
    board_list, winner_prediction_gt = t
    
    for i,j in zip(board_list, winner_prediction_gt):
        d = encode_loaded(i,j)
        data_list.append(d)
    
    
with open("dataset.p", "wb") as f:
        pickle.dump(data_list, f)







