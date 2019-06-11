#!/usr/bin/env python3

"""
Framework for 2048 & 2048-like Games (Python 3)

Author: Hung Guei (moporgic)
        Computer Games and Intelligence (CGI) Lab, NCTU, Taiwan
        http://www.aigames.nctu.edu.tw
Modifier: Kuo-Hao Ho (lukewayne123)
"""

from board import board
from action import action
from episode import episode
from statistic import statistic
from agent import player
from agent import rndenv
import sys


if __name__ == '__main__':
    print('2048 Demo: ' + " ".join(sys.argv))
    print()
    
    total, block, limit = 100000, 1000, 0
    play_args, evil_args = "", ""
    load, save = "", ""
    summary = False
    for para in sys.argv[1:]:
        if "--total=" in para:
            total = int(para[(para.index("=") + 1):])
        elif "--block=" in para:
            block = int(para[(para.index("=") + 1):])
        elif "--limit=" in para:
            limit = int(para[(para.index("=") + 1):])
        elif "--play=" in para:
            play_args = para[(para.index("=") + 1):]
        elif "--evil=" in para:
            evil_args = para[(para.index("=") + 1):]
        elif "--load=" in para:
            load = para[(para.index("=") + 1):]
        elif "--save=" in para:
            save = para[(para.index("=") + 1):]
        elif "--summary" in para:
            summary = True
    
    stat = statistic(total, block, limit)
    
    if load:
        input = open(load, "r")
        stat.load(input)
        input.close()
        summary |= stat.is_finished()
    
    reward_list = []
    with player(play_args) as play, rndenv(evil_args) as evil:
        while not stat.is_finished():
            play.open_episode("~:" + evil.name())
            evil.open_episode(play.name() + ":~")
            # type
            ## game => last episode in stat
            ## play => player class (inherit from weight_agent)
            ## evil => rndenv class (inherit from random_agent)
            stat.open_episode(play.name() + ":" + evil.name())
            game = stat.back()
            while True:
                # Play and environment plays in turns
                who = game.take_turns(play, evil)   ## choose who to take action
                move = who.take_action(game.state()) ## input a board as state, return action class
                if not game.apply_action(move) or who.check_for_win(game.state()):
                    break
            win = game.last_turns(play, evil)
            save_weight = stat.close_episode(win.name())
            total_reward = play.close_episode(stat.back().ep_moves, win.name())
            evil.close_episode(win.name())
            reward_list.append(total_reward)
            if len(reward_list) > 1000:
                with open('reward', 'a') as f:
                    print('write reward record')
                    for i in reward_list:
                        f.write(str(i)+',')
                    del reward_list
                    reward_list = []
                    
                print('write player weight')
                play.save_weights('./player_weight')
                
    if summary:
        stat.summary()
    
    if save:
        output = open(save, "w")
        stat.save(output)
        output.close()