import gym
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt

env = gym.make("CartPole-v0")
env = env.unwrapped
N_STATES = env.observation_space.shape[0]   # How many States
N_ACTIONS = env.action_space.n              # How many Actions
EPSILON = 0.9

BATCH_SIZE = 64
GAMMA = 0.9
MEMORY_CAPACITY = 400
LR = 0.01
NET_UPDATE = 100                   # update net
TRAIN_TIME = 30000000

class ReplayMemory(object):
    def __init__(self):
        self.capacity = 500
        self.memory = []
        self.position = 0

    def push(self, data):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = data
        self.position = (self.position + 1) % self.capacity

memory = ReplayMemory()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.func = nn.Linear(N_STATES, 10)
        self.func.weight.data.normal_(0, 0.1)  # initialization
        self.out = nn.Linear(10, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.func(x)
        x = F.relu(x)
        action_value = self.out(x)
        return action_value

class DQN(object):
    def __init__(self):
        self.eval_net = Net()
        self.target_net = Net()

        self.learn_counter = 0
        self.memory_counter = 0
        self.memory = numpy.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))      # state, action, reward, next_state
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        if numpy.random.uniform() < EPSILON:
            action_value = self.eval_net.forward(x)     # get 2 Actions
            action = torch.argmax(action_value)
            action = action.data.numpy()
        else:
            action = numpy.random.randint(0, N_ACTIONS)
        return action

    def store_memory(self, s, a, r, s_):
        memory = numpy.hstack((s, a, r, s_))
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = memory
        self.memory_counter += 1

    def learn(self):
        if self.learn_counter % NET_UPDATE == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())     # update the net
        self.learn_counter += 1

        index = numpy.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES: N_STATES + 1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES + 1:N_STATES + 2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        loss = self.loss_func(q_eval, q_target)

        memory.push(loss.data.numpy())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

start_time = time.time()  # start time
loss_data = []
loss_data_x = []
episode_data = [0]
episode_data_x = []
count = 1

def loss_(episode):
    global count
    global episode_data_x
    if len(memory.memory) == 500:
        av_loss = numpy.average(memory.memory)
        loss_data_x.append(count)
        loss_data.append(av_loss)
        count += 1
        print(av_loss)
        if av_loss <= 0.01:
            end_time = time.time()  # end time
            print("cost time:  ", end_time - start_time, "s", "Episode:", episode)
            plt.figure(figsize=(11, 7))
            plt.title("Loss")
            plt.xlabel("Learning time")
            plt.ylabel("Loss")
            plt.plot(loss_data_x, loss_data, color="blue", label="loss")
            plt.legend()
            plt.show()
            time_()
            print("Max Time:", max_time)
            exit()

def time_():
    global episode_data_x
    plt.figure(figsize=(11, 7))
    plt.title("Result")
    plt.xlabel("Episode")
    plt.ylabel("Time")
    # episode_data_x = data_x_update(loss_data_x, episode_data_x)
    plt.plot(episode_data_x, episode_data, color='red', label='Time')
    plt.legend()
    plt.show()

def time_of_episode(episode_time):
    global x
    episode_data.append(episode_time)


max_time = 0
if __name__ == '__main__':
    # Train
    dqn = DQN()
    for episode in range(TRAIN_TIME):
        s = env.reset()
        episode_max_start = time.time()
        episode_data_x.append(episode)
        while True:
            env.render()
            a = dqn.choose_action(s)
            s_, r, done, info = env.step(a)

            if done:
                r = -100
            dqn.store_memory(s, a, r, s_)

            if dqn.memory_counter > MEMORY_CAPACITY:
                dqn.learn()
            loss_(episode)
            if done:
                episode_max_stop = time.time()
                episode_time = episode_max_stop - episode_max_start
                time_of_episode(episode_time)
                if episode_time > max_time:
                    max_time = episode_time
                break
            s = s_
        print("Episode:", episode)
    env.close()