import numpy as np
import game_map
import copy
import networkx as nx
from matplotlib import pyplot as plt

def one_hot_encoding(i, N):
    """
    vector of size N,
    where i'th entry is 1 and rest are 0
    """
    r = np.zeros(N)
    r[i] = 1
    return r


class Territory:
    def __init__(self, idx, owner, num_troops):
        self.idx = idx
        self.owner = owner
        self.num_troops = num_troops
    
    """
    encode as a vector ingestible by a NN
    """
    def encode(self):
        # owner (one-hot encoding)
        # just_finished_invasion_src (blank at first)
        # just_finished_invasion_dst (blank at first)
        # num_troops
        return np.hstack([one_hot_encoding(self.owner, 3), [0], [0], [self.num_troops]])
        
    def add_troop(self):
        self.num_troops += 1
    

class Risk:
    def __init__(self):
        # it's always player 0's turn =P
        #self.player_turn = 0
        """
        0) place troops
        1) attack
        2) reinforce territory
        3) end turn; a NOP
        """
        self.turn_stage = 0
        self.remaining_troops_to_place = 0
        
        """
        after a successful invasion,
        you get to reinforce troops from the src to the dst territory
        """
        self.just_finished_invasion = 0
        self.just_finished_invasion_src = None
        self.just_finished_invasion_dst = None
        
        self.map = game_map.make_map()
        self.num_territories = self.map.number_of_nodes()
        self.num_players = 3
        
        if self.num_territories % self.num_players != 0:
            raise ValueError("Number of territories ({}) must be divisible by number of players ({}) for a fair game".format(self.num_territories,  self.num_players))
            
        self.territories = [None for i in range(self.num_territories)]
        self.initialize_board()
        
        self.remaining_troops_to_place = self.get_number_of_troops_to_place(0)
        
    
    def initialize_board(self):
        """
        randomly divide the territories into equal parts among all the players,
        and place 3 troops on each territory
        """
        perm = np.random.permutation(np.arange(self.num_territories))
        n = self.num_territories // self.num_players

        for i in range(self.num_players):
            for j in perm[n * i : n * (i+1)]:
                self.territories[j] = Territory(j, i, 3)
    
    
    def encode(self):
        """
        return encoding of global features
        and a list of each territory's encoded features
        """
        # GLOBAL features:
        # turn_stage (one-hot encoding)
        # remaining_troops_to_place (int)
        # just_finished_invasion (bool)
        global_features = np.hstack([one_hot_encoding(self.turn_stage, 4), [self.remaining_troops_to_place], [self.just_finished_invasion]])
        
        territory_features = [t.encode() for t in self.territories]
        
        if self.just_finished_invasion:
            territory_features[self.just_finished_invasion_src][4] = 1
            territory_features[self.just_finished_invasion_dst][5] = 1
            
        return global_features, territory_features


    def end_attack_stage(self):
        
        self.turn_stage = 2
        
        self.just_finished_invasion = 0
        self.just_finished_invasion_src = None
        self.just_finished_invasion_dst = None
        
        
    def end_reinforce_stage(self):
        
        self.turn_stage = 3
        
        self.just_finished_invasion = 0
        self.just_finished_invasion_src = None
        self.just_finished_invasion_dst = None
        
        
    def start_next_players_turn(self):
        """
        change the ownership of territories by rotating CCW by 1.
        Prepare for the `place troops` stage of the next player's turn.
        if the next player has already lost, jump them forwards to the `end turn` stage.
        """
        
        # rotate the owners by 1
        for t in self.territories:
            t.owner = (t.owner - 1) % self.num_players
        
        self.turn_stage = 0
        self.remaining_troops_to_place = self.get_number_of_troops_to_place(0)
        
        # this only happens if you've already lost all your territories
        if self.remaining_troops_to_place == 0:
            self.turn_stage = 3


    def place_troop(self, idx):
        """
        automatically goes to attack stage (2) after placing last troop
        """
        self.territories[idx].add_troop()
        self.remaining_troops_to_place -= 1
        
        # if you've placed all your troops,
        # then move on to the attack stage of your turn.
        if self.remaining_troops_to_place == 0:
            self.turn_stage = 1
            
    
    def get_my_territories(self, owner=0):
        """
        returns list of indices of all territories owned by `owner`
        """
        return [i for i,t in enumerate(self.territories) if t.owner == owner]
        
    
    
    def reinforce_territory_post_invasion(self, src, dst, num):
        self.reinforce_territory(src, dst, num)
        
        self.just_finished_invasion = 0
        self.just_finished_invasion_src = None
        self.just_finished_invasion_dst = None
        
        
    def reinforce_territory(self, src, dst, num):
        """
        move `num` troops from the `src` territory to the `dst` territory.
        At least 1 troop must remain in the `src` territory
        """
        t_src = self.territories[src]
        t_dst = self.territories[dst]
        
        if t_src.owner != t_dst.owner:
            raise ValueError("Source and Destination Territories are owned by different players")

        if t_src.num_troops - num < 1:
            raise ValueError("Moving ({}) troops would abandon the source territory ({})".format(num, src))
        
        t_src.num_troops -= num
        t_dst.num_troops += num
        
    
    def get_number_of_troops_to_place(self, owner):
        """
        returns the number of troops that player `owner`
        gets to place at the start of their turn.
        
        Currently, this is `number of territories` // 3
        """
        num_territories_owned = 0
        
        for i in self.territories:
            if i.owner == owner:
                num_territories_owned += 1
                
        # if you're out of the game, tough shit
        if num_territories_owned == 0:
            return 0
        
        a = num_territories_owned // 3
        
        # always get at least 3 new troops
        return max(3, a)
    
    
    def check_winner(self):
        """
        simplified win condition:
            first to own 38 territories
            (~80% of map)
            is declared the victor
        this is b/c the untrained network will
            suck at the game, and take a v long
            time to accidentally win
            
        returns 0, 1, 2 as the index of the winner,
        returns None if win condition not yet reached.
        """
        
        num_territories_owned = [0 for i in range(self.num_players)]
        
        for i in self.territories:
            num_territories_owned[i.owner] += 1
            
        for i in range(self.num_players):
            if num_territories_owned[i] >= self.num_territories * 0.7:
                return i
        
        return None
    
    def get_num_territories_owned(self):
        
        num_territories_owned = [0 for i in range(self.num_players)]
        
        for i in self.territories:
            num_territories_owned[i.owner] += 1
            
        return num_territories_owned
        
    
    def get_attack_moves_from_territory(self, idx, owner):
        """
        get all the moves associated with launching an attack from territory `idx`
        
        moves are specified by (src, dst, num_troops_atk, num_troops_def) tuple
        returns a list of moves
        """
        
        my_territory = self.territories[idx]
        
        src_num_troops = my_territory.num_troops
        if src_num_troops <= 1:
            return []
        
        # cannot attack with more troops than present at `src` territory
        # cannot abandon `src` territory; must leave at least 1 troops behind
        a = src_num_troops - 1
        
        # can't attack with more than 3 troops
        max_attacking_troops = min(3, a)
        
        # neighbors of the territory on the map graph
        neighboring_idx = list(self.map.neighbors(idx))
        # neighboring territories owned by a different player
        neighboring_idx = [i for i in neighboring_idx if self.territories[i].owner != owner]
        
        res = []
        
        for dst in neighboring_idx:
            
            num_troops_def = min(2, self.territories[dst].num_troops)
            
            for num_troops_atk in range(1, max_attacking_troops + 1):
                tmp = (idx, dst, num_troops_atk, num_troops_def)
                res.append(tmp)
                
        return res
        
    
    def attack(self, src, dst, num_troops_atk, atk_casualties, def_casualties):
        t_src = self.territories[src]
        t_dst = self.territories[dst]
        
        t_src.num_troops -= atk_casualties
        t_dst.num_troops -= def_casualties
        
        # if all the defenders were killed
        if t_dst.num_troops <= 0:
            # then change the ownership of the territory
            # and move in all of the attackers who survived the attack
            t_dst.num_troops = (num_troops_atk - atk_casualties)
            t_dst.owner = t_src.owner
            
            # and set the flags so you have the option to move 
            # additional troops to the new territory
            self.just_finished_invasion = 1
            self.just_finished_invasion_src = src
            self.just_finished_invasion_dst = dst

            
        
        
    def get_attack_moves(self, owner=0):
        """
        get all the moves associated with launching an attack from territory `idx`
        
        moves are specified by (src, dst, num_troops_atk, num_troops_def) tuple
        returns a list of moves
        """
        
        my_territories_idx = [i.idx for i in self.territories if i.owner == owner]
        
        results = []
        for idx in my_territories_idx:
            results += self.get_attack_moves_from_territory(idx, owner)
        
        return results
        
        
    def get_reinforce_moves(self, owner=0):
        """
        get all the moves associated with reinforcing a territory
        
        moves are specified by (src, dst, num_troops) tuple
        returns a list of moves
        """
        my_territories = [i for i in self.territories if i.owner == owner]
        
        my_territories_idx = [t.idx for t in my_territories]
        
        results = []
        
        for t in my_territories:
            src = t.idx
            
            for dst in my_territories_idx:
                if src == dst:
                    continue
                
                # number of troops that can be
                # moved from each territory
                # = (number of troops) - 1
                for n in range(1, t.num_troops):
                    tmp = (src, dst, n)
                    results.append(tmp)
                    
        return results
    
    def get_reinforce_moves_post_invasion(self):
        """
        get all the moves associated with reinforcing a territory
        where src = just_finished_invasion_src
              dst = just_finished_invasion_src
        
        moves are specified by (src, dst, num_troops) tuple
        returns a list of moves
        """
        src = self.just_finished_invasion_src
        dst = self.just_finished_invasion_dst
        
        N = self.territories[src].num_troops
        
        results = []
        
        # number of troops that can be
        # moved from each territory
        # = (number of troops) - 1
        # NOTE: have option of moving 0 troops, ie a NOP
        for n in range(0, N):
            tmp = (src, dst, n)
            results.append(tmp)
                    
        return results
        
        
    def draw_map(self, player=0):
        g = self.map
        color_list = ["red", "green", "yellow"]
        node_colors = [color_list[(self.territories[i].owner + player) % 3] for i in g.nodes()]
        
        labels={i : str(self.territories[i].num_troops) for i in g.nodes() }
            
        nx.draw_kamada_kawai(g, labels=labels, nodelist=g.nodes(), node_color=node_colors,font_size=56, node_size=4000)


def get_attack_outcomes(r, src, dst, num_troops_atk, num_troops_def):
    #print(src, dst, num_troops_atk, num_troops_def)
    outcomes = []
    
    # src, dst, num_troops_atk, atk_casualties, def_casualties
        
    if num_troops_atk == 3 and num_troops_def == 2:
        # attacker wins 2
        prob = 2890 / 7776
        R = copy.deepcopy(r)
        R.attack(src, dst, num_troops_atk, 0, 2)
        outcomes.append((R, prob))
        
        # defender wins 2
        prob = 2275 / 7776
        R = copy.deepcopy(r)
        R.attack(src, dst, num_troops_atk, 2, 0)
        outcomes.append((R, prob))
        
        # both lose 1
        prob = 2611 / 7776
        R = copy.deepcopy(r)
        R.attack(src, dst, num_troops_atk, 1, 1)
        outcomes.append((R, prob))
        
        return outcomes
    
    if num_troops_atk == 2 and num_troops_def == 2:
        # attacker wins 2
        prob = 295 / 1296
        R = copy.deepcopy(r)
        R.attack(src, dst, num_troops_atk, 0, 2)
        outcomes.append((R, prob))
        
        # defender wins 2
        prob = 581 / 1296
        R = copy.deepcopy(r)
        R.attack(src, dst, num_troops_atk, 2, 0)
        outcomes.append((R, prob))
        
        # both lose 1
        prob = 420 / 1296
        R = copy.deepcopy(r)
        R.attack(src, dst, num_troops_atk, 1, 1)
        outcomes.append((R, prob))
        
        return outcomes
    
    if num_troops_atk == 1 and num_troops_def == 2:
        # attacker wins 1
        prob = 55 / 216
        R = copy.deepcopy(r)
        R.attack(src, dst, num_troops_atk, 0, 1)
        outcomes.append((R, prob))
        
        # defender wins 1
        prob = 161 / 216
        R = copy.deepcopy(r)
        R.attack(src, dst, num_troops_atk, 1, 0)
        outcomes.append((R, prob))
        
        return outcomes
    
    if num_troops_atk == 3 and num_troops_def == 1:
        # attacker wins 1
        prob = 855 / 1296
        R = copy.deepcopy(r)
        R.attack(src, dst, num_troops_atk, 0, 1)
        outcomes.append((R, prob))
        
        # defender wins 1
        prob = 441 / 1296
        R = copy.deepcopy(r)
        R.attack(src, dst, num_troops_atk, 1, 0)
        outcomes.append((R, prob))
        
        return outcomes
    
    if num_troops_atk == 2 and num_troops_def == 1:
        # attacker wins 1
        prob = 125 / 216
        R = copy.deepcopy(r)
        R.attack(src, dst, num_troops_atk, 0, 1)
        outcomes.append((R, prob))
        
        # defender wins 1
        prob = 91 / 216
        R = copy.deepcopy(r)
        R.attack(src, dst, num_troops_atk, 1, 0)
        outcomes.append((R, prob))
        
        return outcomes
    
    if num_troops_atk == 1 and num_troops_def == 1:
        # attacker wins 1
        prob = 15 / 36
        R = copy.deepcopy(r)
        R.attack(src, dst, num_troops_atk, 0, 1)
        outcomes.append((R, prob))
        
        # defender wins 1
        prob = 21 / 36
        R = copy.deepcopy(r)
        R.attack(src, dst, num_troops_atk, 1, 0)
        outcomes.append((R, prob))
        
        return outcomes



"""
returns list of 
    markov vector of 
        (`Risk` objects, `probability of outcome`) tuple
after making each move


also returns flag 
    = 1 if this move ends your turn
    = 0 otherwise
"""
def get_move_list(r):
    # end of turn stage
    if r.turn_stage == 3:
        R = copy.deepcopy(r)
        R.start_next_players_turn()
        return [[(R, 1.0)]], True
    
    results = []
    
    # place troops stage
    if r.turn_stage == 0:
        territories_idx = r.get_my_territories()
        
        for i in territories_idx:
            R = copy.deepcopy(r)
            R.place_troop(i)
            
            results.append([(R, 1.0)])

        return results, False

    
    # reinforce territory stage
    if r.turn_stage == 2:
        
        # option to not reinforce anywhere
        R = copy.deepcopy(r)
        R.end_reinforce_stage()
        results.append([(R, 1.0)])
        
        moves = r.get_reinforce_moves()
        
        for t in moves:
            src, dst, num = t
            
            R = copy.deepcopy(r)
            
            R.reinforce_territory(src, dst, num)
            R.end_reinforce_stage()
            results.append([(R, 1.0)])

        return results, False
    
    
    # attack stage
    if r.turn_stage == 1:
        

        # if just successfully took over a territory,
        # then can move additional troops from src to dst
        if r.just_finished_invasion:
                    
            moves = r.get_reinforce_moves_post_invasion()
            
            # one of the moves is a NOP of moving 0 troops
            for t in moves:
                src, dst, num = t
                
                R = copy.deepcopy(r)
                
                R.reinforce_territory_post_invasion(src, dst, num)
                results.append([(R, 1.0)])
    
            return results, False

        # this is the hard-part
        # actually attacking neighboring territories
        # saved the hardest part for last...
        # this is the only stage with stochastic outcomes
        else:
            # option to stop attacking
            R = copy.deepcopy(r)
            R.end_attack_stage()
            results.append([(R, 1.0)])
            
            moves = r.get_attack_moves()

            for t in moves:
                (src, dst, num_troops_atk, num_troops_def) = t
                
                # all the outcomes associated with making a move
                # and their probability of occuring
                outcomes = get_attack_outcomes(r, src, dst, num_troops_atk, num_troops_def)
                
                results.append(outcomes)
            
            return results, False





def step(r, player, prob_random_move = 1.0, model=None, animate=True):
    
    if model is None and prob_random_move != 1.0:
        raise ValueError("Without a model, can only make 100% random moves.")
    
    results, flag = get_move_list(r)
    
    print(list(np.roll(r.get_num_territories_owned(), player)))
    

    if model is not None:
        print(list(np.roll(get_winning_chances(r, model), player)))
    
    if animate:
        plt.clf()
        r.draw_map(player)
        plt.pause(0.001)
    
    if flag:
        print("END PLAYER {} TURN".format(player))
        player = (player + 1) % 3
        
        
    if np.random.rand() < prob_random_move:
        """
        choose a move at random
        """
        i = np.random.choice(range(len(results)))
    
    
    else:
        """
        choose the move with the highest associated winning odds
        """
        winning_odds = [get_winning_chances_stochastic(a, model) for a in results]
        my_winning_odds = [i[0] for i in winning_odds]
        #print(my_winning_odds)
        i = np.argmax(my_winning_odds)
    
    selected_move = results[i]
    boards = [i[0] for i in selected_move]
    probs = [i[1] for i in selected_move]
    
    r = np.random.choice(boards, 1, p=probs)
    r = r[0]
    
    return r, player
    

from torch_geometric.data import Data
import torch

edge_list = list(Risk().map.edges())
# include both directions of edges
edge_list += [tuple(reversed(i)) for i in edge_list]

edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

def encode(r):
    a,b = r.encode()
    d = [np.concatenate([a,i]) for i in b]
    x = torch.tensor(d, dtype=torch.float)
    
    return Data(x=x, edge_index=edge_index)


#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#model = Net().to(device)

def get_winning_chances(r, model):
    d = encode(r)
    d.batch = torch.tensor([0 for _ in d.x])
    
    res = model(d).exp().detach()
    res = res.numpy().ravel()
    
    return res

"""
for when the outcome of your move is stochastic.
this averages your expected return over the possible outcomes
"""
def get_winning_chances_stochastic(outcomes, model):
    res = 0
    
    for tmp in outcomes:
        r, prob = tmp
        res += get_winning_chances(r, model) * prob
        
    return res
    


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
    
    winner_prediction_gt = [np.roll(winner, -i) for i in player_turn_list]
    
    # add the correct outcome to the data object
    for d,y in zip(board_list, winner_prediction_gt):
        #d.y = torch.LongTensor([np.nonzero(y)[0][0]])
        d.y = torch.tensor([y]).float()

    return board_list




#board_list = play_game()

"""
from multiprocessing import Pool
p = Pool(1)

if __name__ == '__main__':

    board_list = []
    for i in p.map(play_game, range(1)):
        board_list += i
"""

board_list = []
for i in range(1):
    board_list += play_game()







