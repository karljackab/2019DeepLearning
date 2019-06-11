#!/usr/bin/env python3

"""
Basic framework for developing 2048 programs in Python

Author: Hung Guei (moporgic)
        Computer Games and Intelligence (CGI) Lab, NCTU, Taiwan
        http://www.aigames.nctu.edu.tw
Modifier: Kuo-Hao Ho (lukewayne123)
"""

from board import board
from action import action
from weight import weight
from array import array
from episode import episode
import random
import sys
import copy
import math

class agent:
    """ base agent """
    
    def __init__(self, options = ""):
        self.info = {}
        options = "name=unknown role=unknown " + options
        for option in options.split():
            data = option.split("=", 1) + [True]
            self.info[data[0]] = data[1]
        return
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        return
    
    def open_episode(self, flag = ""):
        return
    
    def close_episode(self, flag = ""):
        return
    
    def take_action(self, state):
        return action()
    
    def check_for_win(self, state):
        return False
    
    def property(self, key):
        return self.info[key] if key in self.info else None
    
    def notify(self, message):
        data = message.split("=", 1) + [True]
        self.info[data[0]] = data[1]
        return
    
    def name(self):
        return self.property("name")
    
    def role(self):
        return self.property("role")


class random_agent(agent):
    """ base agent for agents with random behavior """
    
    def __init__(self, options = ""):
        super().__init__(options)
        seed = self.property("seed")
        if seed is not None:
            random.seed(int(seed))
        return
    
    def choice(self, seq):
        target = random.choice(seq)
        return target
    
    def shuffle(self, seq):
        random.shuffle(seq)
        return

    def close_episode(self, ep, flag = ""):
        return 


class weight_agent(agent):
    """ base agent for agents with weight tables """
    
    def __init__(self, options = ""):
        super().__init__(options)
        self.episode = episode()
        self.net = []
        self.tuples = []
        init = self.property("init")
        if init is not None:
            self.init_weights()
        load = self.property("load")
        if load is not None:
            self.load_weights(load)
        self.alpha = 0.1
        alpha = self.property("alpha")
        if alpha is not None:
            self.alpha = alpha
        return
    
    def __exit__(self, exc_type, exc_value, traceback):
        save = self.property("save")
        if save is not None:
            self.save_weights(save)
        return

    def init_weights(self):
        self.net += [weight(16777216)]*4

        self.tuples.append([0,1,2,3,4,5])
        self.tuples.append([4,5,6,7,8,9])
        self.tuples.append([0,1,2,4,5,6])
        self.tuples.append([4,5,6,8,9,10])
        self.len_tuples = len(self.tuples)
        return
    
    def load_weights(self, path):
        input = open(path, 'rb')
        size = array('L')
        size.fromfile(input, 1)
        size = size[0]
        for i in range(size):
            self.net += [weight()]
            self.net[-1].load(input)
        return

    def save_weights(self, path):
        output = open(path, 'wb')
        array('L', [len(self.net)]).tofile(output)
        for w in self.net:
            w.save(output)
        return

    def open_episode(self, flag = ""):
        self.episode.clear()
        return

    def close_episode(self, ep, flag = ""):
        total_reward = 0
        episode = ep[2:].copy()
        # backward
        ## episode element => (state, action, reward, time usage)
        episode.reverse()
        exact = 0
        for i in range(3, len(episode), 2):
            reward = episode[i-2][2]
            total_reward += reward
            value = self.get_value(episode[i][0])
            error = exact - value
            v = self.alpha * error / 32
            exact = reward + self.update_value(episode[i][0], v)
        return total_reward

    def get_index(self, board_state):
        idx_list = [0]*4
        for idx in range(4):
            temp = 0
            for i in self.tuples[idx]:
                temp = temp*16 + board_state[i]
            idx_list[idx] = temp
        return idx_list

    def get_value(self, board_state):
        value = 0.0
        for i in range(8):
            new_board = board(board_state)
            if (i >= 4):
                new_board.transpose()
            new_board.rotate(i)
            idx_list = self.get_index(new_board)
            for weight_idx, idx in enumerate(idx_list):
                value += self.net[weight_idx][idx]
        return value

    def update_value(self, board_state, value):
        total = 0.0
        for i in range(8):
            new_board = board(board_state)
            if (i >= 4):
                new_board.transpose()
            new_board.rotate(i)
            idx_list = self.get_index(new_board)
            for weight_idx, idx in enumerate(idx_list):
                self.net[weight_idx][idx] += value
                total += self.net[weight_idx][idx]
        return total

class learning_agent(agent):
    """ base agent for agents with a learning rate """
    
    def __init__(self, options = ""):
        super().__init__(options)
        self.alpha = 0.1
        alpha = self.property("alpha")
        if alpha is not None:
            self.alpha = float(alpha)
        return


class rndenv(random_agent):
    """
    random environment
    add a new random tile to an empty cell
    2-tile: 90%
    4-tile: 10%
    """
    
    def __init__(self, options = ""):
        super().__init__("name=random role=environment " + options)
        return
    
    def take_action(self, state):
        empty = [pos for pos, tile in enumerate(state.state) if not tile]
        if empty:
            pos = self.choice(empty)
            tile = random.choice([1] * 9 + [2])
            return action.place(pos, tile)
        else:
            return action()

class player(weight_agent):
    """
    dummy player
    select a legal action randomly
    """
    
    def __init__(self, options = ""):
        super().__init__("name=dummy role=player init=True" + options)
        return

    def take_action(self, state):
        max_value = -1
        max_op = -1
        for op in range(4):
            new_board = board(state)
            reward = new_board.slide(op)
            if reward == -1:
                continue
            else:
                expect = 0
                expect = self.get_value(new_board)
                expect += reward
                if expect > max_value:
                    max_value = expect
                    max_op = op
        if max_op == -1:
            return action()
        else:
            return action.slide(max_op)
    
if __name__ == '__main__':
    print('2048 Demo: agent.py\n')
    pass
