from __future__ import print_function

import sys

from torch.utils.data import DataLoader, Dataset

sys.path.append("../")

import pickle
import numpy as np
import os
import gzip
import matplotlib.pyplot as plt
import torch

from utils import *
from agent.bc_agent import BCAgent
from tensorboard_evaluation import Evaluation


class MyDataset(Dataset):
    def __init__(self, x, y):
        super(MyDataset, self).__init__()
        assert x.shape[0] == y.shape[0]  # assuming shape[0] = dataset size
        self.x = x
        self.y = y

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]


def read_data(datasets_dir="./data", frac=0.1):
    """
    This method reads the states and actions recorded in drive_manually.py 
    and splits it into training/ validation set.
    """
    print("... read data")
    data_file = os.path.join(datasets_dir, 'data.pkl.gzip')

    f = gzip.open(data_file, 'rb')
    data = pickle.load(f)

    # get images as features and actions as targets
    X = np.array(data["state"]).astype('float32')
    y = np.array(data["action"]).astype('float32')

    # split data into training and validation set
    n_samples = len(data["state"])
    X_train, y_train = X[:int((1 - frac) * n_samples)], y[:int((1 - frac) * n_samples)]
    X_valid, y_valid = X[int((1 - frac) * n_samples):], y[int((1 - frac) * n_samples):]
    return X_train, y_train, X_valid, y_valid


def preprocessing(X_train, y_train, X_valid, y_valid, history_length=1):
    # TODO: preprocess your data here.
    # 1. convert the images in X_train/X_valid to gray scale. If you use rgb2gray() from utils.py, the output shape (96, 96, 1)
    # 2. you can train your model with discrete actions (as you get them from read_data) by discretizing the action space 
    #    using action_to_id() from utils.py.

    # History:
    # At first you should only use the current image as input to your network to learn the next action. Then the input states
    # have shape (96, 96, 1). Later, add a history of the last N images to your state so that a state has shape (96, 96, N).

    a, b, c, d = X_train.shape
    X_train = np.array(list(map(rgb2gray, X_train)))
    # X_train = X_train.reshape(a, b, c, 1)
    X_train[:, 83:, :] = 255

    a, b, c, d = X_valid.shape
    X_valid = np.array(list(map(rgb2gray, X_valid)))
    # X_valid = X_valid.reshape(a, b, c, 1)
    X_valid[:, 83:, :] = 255
    # X_valid = rgb2gray(X_valid)
    # print(X_valid.shape)
    # for i in range(100, 1000):
    #     plt.imshow(X_train[i], cmap='gray')
    #     plt.show(block=False)
    #     plt.pause(.1)
    #     plt.close()

    # 2. you can train your model with discrete actions (as you get them from read_data) by discretizing the action space
    #    using action_to_id() from utils.py.
    ##################### y_train #########################
    steps = list(map(action_to_id, y_train))
    print(steps)
    a, b = y_train.shape
    y_train = np.array(steps, dtype=np.int64)
    y_train = y_train.reshape(a)
    ##################### y_valid #########################
    steps = list(map(action_to_id, y_valid))
    a, b = y_valid.shape
    y_valid = np.array(steps, dtype=np.int64)
    y_valid = y_valid.reshape(a)

    return X_train, y_train, X_valid, y_valid


def train_model(X_train, y_train, X_valid, n_minibatches, batch_size, lr, model_dir="./models",
                tensorboard_dir="./tensorboard"):
    # create result and model folders
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    print("... train model")
    a, b, c = X_train.shape
    X_train = X_train.reshape(a, 1, b, c)
    a, b, c = X_valid.shape
    X_valid = X_valid.reshape(a, 1, b, c)

    # TODO: specify your agent with the neural network in agents/bc_agent.py
    agent = BCAgent()
    stats = ["loss"]

    tensorboard_eval = Evaluation(tensorboard_dir, "car", stats)

    # TODO: implement the training
    # 
    # 1. write a method sample_minibatch and perform an update step
    # 2. compute training/ validation accuracy and loss for the batch and visualize them with tensorboard. You can watch the progress of
    #    your training *during* the training in your web browser
    # 
    # training loop
    # for i in range(n_minibatches):
    #     ...
    #     for i % 10 == 0:
    #         # compute training/ validation accuracy and write it to tensorboard
    #         tensorboard_eval.write_episode_data(...)

    ######################### Extra Method ####################
    # training_data = MyDataset(X_train, y_train)
    # val_data = MyDataset(X_valid, y_valid)
    # train_data_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    # val_data_loader = DataLoader(val_data, batch_size=batch_size)
    #
    # # calculate steps per epoch for training and validation set
    # train_steps = len(train_data_loader.dataset) // batch_size
    # val_steps = len(val_data_loader.dataset) // batch_size
    # total_train_loss = 0
    # total_val_loss = 0
    # for i, (train_features, train_labels) in enumerate(train_data_loader):
    #     batch_train_loss = agent.update(train_features, train_labels)
    #     print('[mini-batch #%3d] loss: %.6f' % (i + 1, batch_train_loss))
    #     total_train_loss += batch_train_loss
    #     dict={"loss":batch_train_loss}
    #     tensorboard_eval.write_episode_data(i,dict)

    batch_size = 64
    n_minibatches = len(X_train) // batch_size
    train_idx = np.arange(len(X_train))
    loss = 0
    complete_loss = 0
    for batch_num in range(n_minibatches + 1):
        minibatch_start = batch_num * batch_size
        minibatch_end = (batch_num + 1) * batch_size
        x_batch = X_train[minibatch_start:minibatch_end]
        y_batch = y_train[minibatch_start:minibatch_end]
        x_batch = torch.from_numpy(x_batch)
        y_batch = torch.from_numpy(y_batch)
        loss = agent.update(x_batch, y_batch)
        complete_loss += loss
        # if batch_num % 10 == 0:
        #     # compute training/ validation accuracy and write it to tensorboard
        #     print('[mini-batch #%3d] loss: %.6f' % (batch_num + 1, loss))
        print('[mini-batch #%3d] loss: %.6f' % (batch_num + 1, loss))

    # TODO: save your agent
    model_dir = agent.save(os.path.join(model_dir, "agent.pt"))
    print("Model saved in file: %s" % model_dir)


if __name__ == "__main__":
    # read data
    X_train, y_train, X_valid, y_valid = read_data("./data")

    # preprocess data
    X_train, y_train, X_valid, y_valid = preprocessing(X_train, y_train, X_valid, y_valid, history_length=1)

    # train model (you can change the parameters!)
    train_model(X_train, y_train, X_valid, n_minibatches=1000, batch_size=64, lr=1e-4)
