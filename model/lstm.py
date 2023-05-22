# coding=utf-8
"""
@file    lstm
@date    2023/5/22 13:35
@author  zlf
"""

import torch
import torch.nn as nn
import torch.optim as optim
import random

import data
from model.model import SequenceTaggingModel


class LSTMModel(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(LSTMModel, self).__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        tag_space = self.fc(lstm_out)
        tag_scores = nn.functional.log_softmax(tag_space, dim=2)
        return tag_scores


class LSTMModelWrapper(SequenceTaggingModel):

    def __init__(self,
                 epoch_nums: int,
                 batch_size: int,
                 lr: float,
                 hidden_dim=128):
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.batch_size = batch_size
        self.epoch_nums = epoch_nums
        self._model = None
        self._word2id = {}
        self._tag2id = {}
        self.embedding = None

    def train(self, **kwargs):
        """
        训练LSTM代码
        :param kwargs: corpus: [["a", "b", "c"],["b", "a", "c"]]
                       labels: [["B-LOC", "O", "O"], ["O", "B-PER", "B-ORG"]]
        :return:
        """
        corpus_list = kwargs["corpus"]
        if corpus_list is None or len(corpus_list) == 0:
            raise ValueError("Empty input.")
        labels_list = kwargs["labels"]
        if labels_list is None or len(labels_list) == 0:
            raise ValueError("Empty label.")
        self._word2id = kwargs["word2id"] if kwargs.__contains__("word2id") else data.build_map(corpus_list)
        self._tag2id = kwargs["tag2id"] if kwargs.__contains__("tag2id") else data.build_map(labels_list)
        self._word2id["<pad>"] = len(self._word2id)
        self._word2id["<unk>"] = len(self._word2id)
        self._tag2id["<pad>"] = len(self._tag2id)
        self._tag2id["<unk>"] = len(self._tag2id)
        input_dim = len(self._word2id)
        output_dim = len(self._tag2id)
        numerical_data = []
        for index, corpus in enumerate(corpus_list):
            numerical_data.append((corpus, labels_list[index]))

        # Create the LSTM model
        self._model = LSTMModel(input_dim=input_dim, hidden_dim=self.hidden_dim, output_dim=output_dim)

        # Define the loss function and optimizer
        loss_function = nn.NLLLoss()
        optimizer = optim.SGD(self._model.parameters(), lr=self.lr)

        # Training loop
        for epoch in range(self.epoch_nums):

            # Shuffle the training data
            random.shuffle(numerical_data)

            # Mini-batch training
            for i in range(0, len(numerical_data), self.batch_size):
                batch_data = numerical_data[i:i + self.batch_size]
                batch_inputs = []
                batch_targets = []

                for inputs, targets in batch_data:
                    batch_inputs.append(inputs)
                    batch_targets.append(targets)
                # 将batch_inputs 的size 统一为input的size

                # Clear gradients
                self._model.zero_grad()

                # Convert inputs to tensors
                # batch_inputs = torch.tensor(batch_inputs)
                # batch_targets = torch.tensor(batch_targets)

                batch_inputs = self.format(batch_inputs, self._word2id)
                batch_targets = self.format(batch_targets, self._tag2id)

                # Forward pass
                outputs = self._model(batch_inputs)

                # Compute loss
                loss = loss_function(outputs.view(-1, output_dim), batch_targets.view(-1))

                # Backward pass
                loss.backward()
                optimizer.step()

    def test(self, **kwargs) -> list:
        predicate_res = []
        test_list = kwargs["test"]
        with torch.no_grad():
            self._model.eval()
            for test_sentence in test_list:
                numerical_test_sentence = [self._word2id[word] for word in test_sentence]
                inputs = torch.tensor(numerical_test_sentence).unsqueeze(0)
                outputs = self._model(inputs)
                predicted_labels = torch.argmax(outputs, dim=2).squeeze(0).tolist()

                predicted_tags = [list(self._tag2id.keys())[idx] for idx in predicted_labels]
                predicate_res.append(predicted_tags)
        return predicate_res

    def format(self, inputs: list, maps: {}) -> list:
        max_length = len(inputs[0])
        for item in inputs:
            if len(item) > max_length:
                max_length = len(item)
        batch_tensor = torch.ones(len(inputs), max_length).long() * (maps.get("<pad>"))
        for i, l in enumerate(inputs):
            for j, e in enumerate(l):
                batch_tensor[i][j] = maps.get(e, maps.get("<unk>"))
        return batch_tensor
