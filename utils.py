# coding=utf-8
"""
@file    utils
@date    2023/5/19 15:41
@author  zlf
"""


def flat(inner_list: list) -> list:
    """
    列表展开
    :param inner_list:
    :return:
    """
    flatten_list = []
    for item in inner_list:
        if isinstance(item, list):
            flatten_list.extend(item)
        else:
            flatten_list.append(item)
    return flatten_list


def sent2features(sent: list) -> list:
    """
    单词组成的句子实现特征转换
    :param sent: 句子的单词列表
    :return: features
    """
    return [word2features(sent, i) for i in range(len(sent))]


def word2features(sent: list, index: int) -> {}:
    """
    根据句子单词生成词特征
    :param sent: 句子单词列表
    :param index: 单词索引
    :return: 单词特征
    """
    word = sent[index]
    pre_word = "<s>" if index == 0 else sent[index - 1]
    next_word = "</s>" if index == len(sent) - 1 else sent[index + 1]

    word_features = {
        "w": word,
        "w-1": pre_word,
        "w+1": next_word,
        "w-1:w": pre_word + word,
        "w:w+1": word + next_word,
        "bias": 1
    }
    word_features = {
        "w": word
    }
    return word_features
