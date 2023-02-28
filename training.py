import pdb

import numpy as np
import torch as tc
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from dataset import ImageDataset, get_dataloader
from utils import *


class TrainingAlgorithm:
    def __init__(self, args):

        self.args = args
        self.model = get_model(args)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.learning_rate)
        self.loss_func = nn.CrossEntropyLoss()

        reload_data = args.reload_data
        # image data is loaded from collection of .jpg files, preprocessed and
        # saved as ImageDataset objects using pickle. If reload_data is False
        # and .pkl files already exist, load those instead.
        if reload_data or not datasets_exist():
            print("Loading image dataset...")
            data, labels = load_data_and_labels(args.data_path)

            data_train, data_test, labels_train, labels_test = train_test_split(
                data, labels, test_size=args.test_size
            )

            dataset_train = ImageDataset(data_train, labels_train)
            dataset_test = ImageDataset(data_test, labels_test)
            save_pickle(dataset_train, "dataset_train.pkl")
            save_pickle(dataset_test, "dataset_test.pkl")
        else:
            dataset_train = load_pickle("dataset_train.pkl")
            dataset_test = load_pickle("dataset_test.pkl")

        self.dataset_train = dataset_train
        self.dataset_test = dataset_test
        # create dataloader
        self.train_loader = get_dataloader(dataset_train, batch_size=args.batch_size)

    def train(self):

        num_params = self.model.get_num_params()
        print("\nStarting training...")
        print(f"Model: {self.model.name} with {num_params} trainable parameters")

        if tc.cuda.is_available():
            device_name = "cuda"
            print("Training on GPU\n")
        else:
            device_name = "cpu"
            print("Training on CPU\n")

        device = tc.device(device_name)
        self.model.to(device)

        self.model.train()
        n = self.args.epochs

        for i in range(n):
            for batch_idx, (data, targets) in enumerate(self.train_loader):

                # add data and targets to device
                data, targets = data.to(device), targets.to(device)

                self.model.train()
                self.optimizer.zero_grad()

                preds = self.model(data)
                loss = self.loss_func(preds, targets)

                loss.backward()
                self.optimizer.step()

                # as a first check, print loss and accuracy on training set
                if batch_idx % 10 == 0:

                    print(f"epoch {i+1}, batch {batch_idx+1} | loss: {loss}")

                    with tc.no_grad():

                        self.model.eval()
                        # compute accuracy on subset of training and test set
                        # using a sample size of n
                        n = 200
                        acc_train = compute_accuracy(
                            self.model,
                            self.dataset_train.data,
                            self.dataset_train.labels,
                            n=n,
                        )
                        acc_test = compute_accuracy(
                            self.model,
                            self.dataset_test.data,
                            self.dataset_test.labels,
                            n=n,
                        )

                    print(f"train accuracy: {acc_train}, test accuracy: {acc_test}")
