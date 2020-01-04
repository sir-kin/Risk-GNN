import numpy as np
import pickle

from Risk_board_game import Risk, get_move_list, one_hot_encoding


def step(r, player):
    

    results, flag = get_move_list(r)
    
    print(list(np.roll(r.get_num_territories_owned(), player)))

    
    if flag:
        print("END PLAYER {} TURN".format(player))
        player = (player + 1) % 3
        

    """
    choose a move at random
    """
    i = np.random.choice(range(len(results)))
  
    selected_move = results[i]
    boards = [i[0] for i in selected_move]
    probs = [i[1] for i in selected_move]
    
    r = np.random.choice(boards, 1, p=probs)
    r = r[0]
    
    return r, player
    

edge_list = list(Risk().map.edges())
# include both directions of edges
edge_list += [tuple(reversed(i)) for i in edge_list]

#edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

def encode(r):
    a,b = r.encode()
    d = [np.concatenate([a,i]) for i in b]
    
    return d
    #x = torch.tensor(d, dtype=torch.float)
    
    #return Data(x=x, edge_index=edge_index)



def play_game(_=None):
    player = 0
    
    r = Risk()
    
    board_list = []
    player_turn_list = []
    while r.check_winner() is None:
        board_list.append(encode(r))
        player_turn_list.append(player)
        
        r, player = step(r, player)
        

    
    
    print("PLAYER {} WINS!".format(player))
    
    
    winner = one_hot_encoding(player, 3)
    
    # add the correct outcome to the data object
    winner_prediction_gt = [np.roll(winner, -i) for i in player_turn_list]
    
    return board_list, winner_prediction_gt




board_list, winner_prediction_gt = play_game()

game_idx = 0
with open("saved_games/game_{}".format(game_idx), "wb") as f:
    t = (board_list, winner_prediction_gt)
    pickle.dump(t, f)
    
    





