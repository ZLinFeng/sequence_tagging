# coding=utf-8
"""
@file    train_test_lstm
@date    2023/5/22 15:44
@author  zlf
"""
from model.lstm import LSTMModelWrapper

if __name__ == '__main__':
    lstm_sequence_model = LSTMModelWrapper(epoch_nums=3, batch_size=2, lr=0.001, hidden_dim=10)
    training_data = [
        ("The company is located in New York City", ["O", "O", "O", "O", "O", "B-LOC", "I-LOC", "I-LOC"]),
        ("She visited Paris last summer", ["O", "O", "B-LOC", "B-TIME", "I-TIME"]),
    ]
    corpus = [a[0].split(" ") for a in training_data]
    labels = [a[1] for a in training_data]
    lstm_sequence_model.train(corpus=corpus, labels=labels)
    res = lstm_sequence_model.test(test=[["New", "York", "City"]])
    print(res)
