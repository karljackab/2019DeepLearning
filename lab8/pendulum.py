import argparse
import pickle
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

device = torch.device('cuda:2')
parser = argparse.ArgumentParser(description='Solve the Pendulum-v0 with DDPG')
parser.add_argument(
    '--gamma', type=float, default=0.99, metavar='G', help='discount factor (default: 0.9)')

parser.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default: 0)')
parser.add_argument('--render', action='store_true', help='render the environment')
parser.add_argument(
    '--log-interval',
    type=int,
    default=10,
    metavar='N',
    help='interval between training status logs (default: 10)')
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

TrainingRecord = namedtuple('TrainingRecord', ['ep', 'reward'])
Transition = namedtuple('Transition', ['s', 'a', 'r', 's_'])


class ActorNet(nn.Module):
    def __init__(self):
        super(ActorNet, self).__init__()
        self.fc = nn.Linear(3, 400)
        self.fc2 = nn.Linear(400, 300)
        self.mu_head = nn.Linear(300, 1)

    def forward(self, s):
        x = torch.relu(self.fc(s))
        x = torch.relu(self.fc2(x))
        u = torch.tanh(self.mu_head(x))
        return u


class CriticNet(nn.Module):
    def __init__(self):
        super(CriticNet, self).__init__()
        self.fc = nn.Linear(3, 400)
        self.fc2 = nn.Linear(401, 300)
        self.v_head = nn.Linear(300, 1)

    def forward(self, s, a):
        x = F.relu(self.fc(s))
        x = F.relu(self.fc2(torch.cat([x, a], dim=1)))
        state_value = self.v_head(x)
        return state_value


class Memory():
    data_pointer = 0
    isfull = False

    def __init__(self, capacity):
        self.memory = np.empty(capacity, dtype=object)
        self.capacity = capacity

    def update(self, transition):
        self.memory[self.data_pointer] = transition
        self.data_pointer += 1
        if self.data_pointer == self.capacity:
            self.data_pointer = 0
            self.isfull = True

    def sample(self, batch_size):
        return np.random.choice(self.memory, batch_size)


class Agent():
    max_grad_norm = 0.5

    def __init__(self):
        self.training_step = 0
        self.var = 1.
        self.eval_cnet, self.target_cnet = CriticNet().float().to(device), CriticNet().float().to(device)
        self.eval_anet, self.target_anet = ActorNet().float().to(device), ActorNet().float().to(device)
        self.memory = Memory(10000)
        self.optimizer_c = optim.Adam(self.eval_cnet.parameters(), lr=0.001)
        self.optimizer_a = optim.Adam(self.eval_anet.parameters(), lr=0.0001)
        self.Loss = nn.MSELoss()

    def select_action(self, state):
        state = torch.from_numpy(state).float().to(device).unsqueeze(0)
        mu = self.eval_anet(state)
        dist = Normal(mu, torch.tensor(self.var, dtype=torch.float).to(device))
        action = dist.sample()
        action.clamp(-2.0, 2.0)
        return (action.item(),)

    def save_param(self):
        torch.save(self.eval_anet.state_dict(), 'ddpg_anet_params.pkl')
        torch.save(self.eval_cnet.state_dict(), 'ddpg_cnet_params.pkl')

    def store_transition(self, transition):
        self.memory.update(transition)

    def update(self):
        self.training_step += 1

        transitions = self.memory.sample(32)
        s = torch.tensor([t.s for t in transitions], dtype=torch.float).to(device)
        a = torch.tensor([t.a for t in transitions], dtype=torch.float).view(-1, 1).to(device)
        r = torch.tensor([t.r for t in transitions], dtype=torch.float).view(-1, 1).to(device)
        s_ = torch.tensor([t.s_ for t in transitions], dtype=torch.float).to(device)

        with torch.no_grad():
            q_target = r + args.gamma * self.target_cnet(s_, self.target_anet(s_))
        q_eval = self.eval_cnet(s, a)

        # update critic net
        self.optimizer_c.zero_grad()
        c_loss = self.Loss(q_eval, q_target).mean()
        c_loss.backward()
        nn.utils.clip_grad_norm_(self.eval_cnet.parameters(), self.max_grad_norm)
        self.optimizer_c.step()

        # update actor net
        self.optimizer_a.zero_grad()
        a_loss = -self.eval_cnet(s, self.eval_anet(s)).mean()
        a_loss.backward()
        nn.utils.clip_grad_norm_(self.eval_anet.parameters(), self.max_grad_norm)
        self.optimizer_a.step()

        if self.training_step % 300 == 0:
            self.target_cnet.load_state_dict(self.eval_cnet.state_dict())
        if self.training_step % 301 == 0:
            self.target_anet.load_state_dict(self.eval_anet.state_dict())

        self.var = max(self.var * 0.999, 0.01)

        return q_eval.mean().item()


def main():
    env = gym.make('Pendulum-v0')
    env.seed(args.seed)

    agent = Agent()

    training_records = []
    running_reward, running_q = -1000, 0
    for i_ep in range(10000):
        score = 0
        state = env.reset()

        for t in range(200):
            action = agent.select_action(state)
            state_, reward, done, _ = env.step(action)
            score += reward
            if args.render:
                env.render()
            agent.store_transition(Transition(state, action, (reward + 8) / 8, state_))
            state = state_
            if agent.memory.isfull:
                q = agent.update()
                running_q = 0.99 * running_q + 0.01 * q

        running_reward = running_reward * 0.9 + score * 0.1
        training_records.append(TrainingRecord(i_ep, running_reward))

        if i_ep % args.log_interval == 0:
            print('Step {}\tAverage score: {:.2f}\tAverage Q: {:.2f}'.format(
                i_ep, running_reward, running_q))
        

        
        x = [r.ep for r in training_records]
        y = [r.reward for r in training_records]
        plt.plot(x, y)

        max_y = max(y)
        max_y_idx = y.index(max_y)
        plt.text(max_y_idx, max_y, str(max_y))
        
        plt.title('DDPG')
        plt.xlabel('Episode')
        plt.ylabel('Moving averaged episode reward')
        plt.savefig("./ddpg.png")
        plt.close()
        
    print("Solved! Running reward is now {}!".format(running_reward))
    env.close()
    agent.save_param()
    with open('ddpg_training_records.pkl', 'wb') as f:
        pickle.dump(training_records, f)
    

if __name__ == '__main__':
    main()
