# coding=utf-8
"""
@file    train_crf
@date    2023/5/19 16:09
@author  zlf
"""
import data
from evaluate import Metrics
from model.crf import CRFModel
import pickle


def train_and_save():
    train_word_lists, train_tag_list, word2id, tag2id = data.data_loader("./data/train.txt")
    test_word_lists, test_tag_list, word2id, tag2id = data.data_loader("./data/test.txt")
    model = CRFModel()
    model.train(train_word_lists, train_tag_list)
    predicate_list = model.test(test_word_lists)
    print(predicate_list)
    metrics = Metrics(test_tag_list, predicate_list)
    metrics.report_scores()
    with open("./resources/crf.pkl", "wb") as writer:
        pickle.dump(model, writer)


def load_and_test():
    test_word_lists, test_tag_list, word2id, tag2id = data.data_loader("./data/tw_test.txt")
    with open("./resources/crf.pkl", "rb") as reader:
        model = pickle.load(reader)
    predicate_list = model.test(test_word_lists)
    metrics = Metrics(test_tag_list, predicate_list)
    metrics.report_scores()


if __name__ == '__main__':
    train_and_save()
    # load_and_test()
