import gym
import math
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device('cuda:2')
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

class Agent(object):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.eval_net = Net(self.state_space_dim, 32, self.action_space_dim).to(device)
        self.target_net = Net(self.state_space_dim, 32, self.action_space_dim).to(device)
        self.target_net.load_state_dict(self.eval_net.state_dict())

        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=self.lr)
        self.buffer = []
        self.steps = 0
        self.epsi = self.epsi_high
        self.training = True

    def save_weight(self):
        torch.save(self.eval_net.state_dict(), 'DQN.pkl')

    def load_weight(self):
        self.eval_net.load_state_dict(torch.load('DQN.pkl'))
        self.training = False
        self.epsi = self.epsi_low
        print('load successfully')

    def update_iter(self):
        self.steps += 1
        self.epsi *= self.decay
        if self.steps % self.copy_weight == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())

    def act(self, s0):
        if random.random() < self.epsi:
            a0 = random.randrange(self.action_space_dim)
        else:
            s0 =  torch.tensor(s0, dtype=torch.float).view(1,-1).to(device)
            a0 = torch.argmax(self.eval_net(s0)).item()
        return a0

    def put(self, *transition):
        if len( self.buffer)==self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)
        
    def learn(self):
        if (len(self.buffer)) < self.batch_size:
            return
        
        if not self.training:
            return

        samples = random.sample(self.buffer, self.batch_size)
        s0, a0, r1, s1 = zip(*samples)
        s0 = torch.tensor(s0, dtype=torch.float).to(device)
        a0 = torch.tensor(a0, dtype=torch.long).view(self.batch_size, -1).to(device)
        r1 = torch.tensor(r1, dtype=torch.float).view(self.batch_size, -1).to(device)
        s1 = torch.tensor(s1, dtype=torch.float).to(device)
        
        y_true = r1 + self.gamma * torch.max(self.target_net(s1).detach(), dim=1)[0].view(self.batch_size, -1)
        y_pred = self.eval_net(s0).gather(1, a0)
        
        loss_fn = nn.MSELoss()
        loss = loss_fn(y_pred, y_true)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def plot(self, score, mean):
        plt.close()
        plt.figure(figsize=(20, 10))
    
        plt.title('Cartpole')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(score)
        plt.plot(mean)

        max_score = max(score)
        score_idx = score.index(max_score)
        max_mean = max(mean)
        max_idx = mean.index(max_mean)
        plt.text(len(score)-1, score[-1], str(score[-1]))
        plt.text(len(mean)-1, mean[-1], str(mean[-1]))
        plt.text(score_idx, max_score, str(max_score))
        plt.text(max_idx, max_mean, str(max_mean))
        plt.savefig('./result.png')

env = gym.make('CartPole-v0')
params = {
    'gamma': 0.95,
    'epsi_high': 1,
    'epsi_low': 0.001,
    'decay': 0.995,
    'lr': 0.0005,
    'capacity': 10000,
    'batch_size': 128,
    'copy_weight': 50,
    'state_space_dim': env.observation_space.shape[0],
    'action_space_dim': env.action_space.n
}
agent = Agent(**params)
agent.load_weight()
score = []
mean = []
finish = False
for episode in range(100):
    s0 = env.reset()
    total_reward = 1
    while True:
        # env.render()
        a0 = agent.act(s0)
        s1, r1, done, _ = env.step(a0)

        if done:
            r1 = -1

        agent.put(s0, a0, r1, s1)

        if done:
            break

        total_reward += r1
        s0 = s1
        agent.learn()
    agent.update_iter()
    # if total_reward==200 and not finish:
    #     agent.save_weight()
    #     print('save')
    # if sum(score[-100:])/100 == 200:
    #     finish = True
    #     print('avg score reach 200')

    score.append(total_reward)
    print(f'Episode {episode}: score {total_reward}, avg {sum(score[-100:])/100}')
    mean.append(sum(score[-100:])/100)
agent.plot(score, mean)