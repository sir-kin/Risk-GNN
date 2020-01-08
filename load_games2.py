import numpy as np
import networkx as nx
import copy
import progressbar

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

file_list = glob.glob('saved_games2/*')

data_list = []

for fn in progressbar.progressbar(file_list):
    with open(fn, "rb") as f:
        t = pickle.load(f, encoding='bytes') 
    
    data_list += t
    
    
with open("dataset2.p", "wb") as f:
        pickle.dump(data_list, f)







