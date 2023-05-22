# coding=utf-8
"""
@file    model
@date    2023/5/22 14:41
@author  zlf
"""
import abc


class SequenceTaggingModel:

    @abc.abstractmethod
    def train(self, **kwargs):
        pass

    @abc.abstractmethod
    def test(self, **kwargs) -> list:
        pass
