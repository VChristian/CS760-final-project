from read_data import get_data
from tree_environment import decision_tree_env, tree
from rnn_implementation import rnn

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

import numpy as np
import matplotlib.pyplot as plt

import csv
import math

# if __name__ == "__main__":
#     dataset = np.asarray(get_data("data/D3leaves.txt"))
#     tree_env = decision_tree_env(3,dataset,[0,0], train_split=1)
#     tree_env.feature_to_split = [0,0,1,0,0,0,0]
#     head = tree_env.make_tree(0,dataset)
#     tree_env.set_tree(head)
#     print(tree_env)

# data, train_indices, test_indices, heart_categorical = get_data("heart.csv")
# test_set = data[test_indices,:]
# train_set = data[train_indices,:]

def get_data_1(filename):
    '''
    Given a filename, get data

    Data format is is

    x11 x21 y1
    x12 x22 y2
    .
    .
    .
    x1n x2n yn

    :param filename: data file name
    :return: list of lists [x1i, x2i, yi]
    '''
    data = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            datum = line.strip('\n').split(' ')
            datum = [float(d) for d in datum]
            data.append(datum)
    return data

train_set = np.asarray(get_data_1("data/D1.txt"))

num_episodes = 30
epochs = 1000
feature_dimension = train_set.shape[1]-1
hidden_dim = 2**int(math.log2(train_set.shape[1]-1))
tree_depth = 2
alpha = .5
#random.seed(1)


tree_env = decision_tree_env(tree_depth,train_set,[0,0])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
rnn_controller = rnn(feature_dimension, hidden_dim).to(device) # (feature_dim, hidden_size)

rnn_optimizer = optim.Adam(rnn_controller.parameters(), lr=0.0005)

def get_action(prob):
    m = Categorical(prob)
    action = m.sample()
    return action, m

def discounted_rewards(rewards):
    rewards = np.array(rewards)
    gamma = .99
    def calc_rewards(r):
        discounted_reward = [0]*len(r)
        for i,_ in enumerate(r):
            for j in range(i,len(r)):
                discounted_reward[i] += (gamma**(j-i))*r[j]
        return discounted_reward
    start = 0
    end = tree_depth
    while start < len(rewards):
        rewards[start:end+1] = calc_rewards(rewards[start:end+1])
        start = end+1
        end = start+tree_depth
    return rewards

def compute_loss(obs, rewards, actions):
    loss = 0
    i = 0
    for input,h0 in obs:
        prob,_=rnn_controller(input,h0)
        _,dist = get_action(prob.squeeze())
        logp = dist.log_prob(actions[i])
        loss += logp*rewards[i]
        i+=1
    
    return -loss/i

all_rewards = []
all_mean_rewards = []
all_mean_auc = []
auc_test_set = []
b = .6

for i in range(epochs):

    # our trajectory buffer
    batch_obs = []
    batch_rewards = []
    batch_actions = []
    batch_auc = []

    for j in range(num_episodes):

        # reset env at every episode
        tree_env.reset()
        done = False

        # initial input to rnn controller
        input = None
        h0 = torch.randn(1,1,hidden_dim)
        episode_length = 0
        while not done:

            batch_obs.append((input, h0))

            # step in the environment
            input, h0 = rnn_controller(input, h0)
            action,_ = get_action(input.squeeze())
            _, done = tree_env.step(action)
            
            #update my buffer
            batch_actions.append(action)

            input = np.zeros((1,1,feature_dimension), dtype = np.float32)
            input[0,0,int(action)] = 1.0
            input = torch.tensor(input)

            episode_length += 1

        # build and evaluate tree
        tree_head = tree_env.make_tree(0)
        rnn_constructed_tree = tree(tree_head)
        auc_score = tree_env.evaluate_tree(rnn_constructed_tree)
        #auc_test_set.append(tree_env.evaluate_tree(rnn_constructed_tree, dataset=test_set))

        # should i create another baseline
        baseline_auc = auc_score-b

        # reward calculations - should this be cumulative?
        batch_rewards += [baseline_auc]*episode_length
        batch_auc += [auc_score]*episode_length
    
    rnn_optimizer.zero_grad()
    loss = compute_loss(batch_obs, discounted_rewards(batch_rewards), batch_actions)
    loss.backward()
    rnn_optimizer.step()

    # print data on the episode
    mean_reward = sum(batch_rewards)/len(batch_rewards)
    mean_auc = sum(batch_auc)/len(batch_auc)

    all_mean_rewards.append(mean_reward)
    all_mean_auc.append(mean_auc)
    print("epoch: {}, loss: {}, mean_reward: {}, auc: {}".format(i+1, float(loss.detach()), mean_reward, mean_auc))

    if i+1 % 10 == 0:
        torch.save(rnn_controller.state_dict, 'models/rnn_controller'+str(i+1)+'.pth')

torch.save(rnn_controller.state_dict(), 'models/rnn_controller-final.pth')

# plot auc for each epoch
plt.figure(1)
plt.plot(list(range(epochs)), all_mean_rewards)

plt.figure(2)
plt.plot(list(range(epochs)), all_mean_auc)

plt.show()

with open("mean_auc.csv", "w") as f:
    csvWriter = csv.writer(f)
    csvWriter.writerow(all_mean_rewards)