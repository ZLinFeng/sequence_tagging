# coding=utf-8
"""
@file    CRFModel
@date    2023/5/19 15:23
@author  zlf
"""
from sklearn_crfsuite import CRF

import model.model
from utils import sent2features


class CRFModel(model.model.SequenceTaggingModel):

    def __init__(self):
        self._crf = CRF(algorithm="lbfgs",
                        c1=0.1,
                        c2=0.1,
                        max_iterations=100,
                        all_possible_transitions=False)

    def train(self, sentences: list, tag_lists: list):
        features = [sent2features(s) for s in sentences]
        self._crf.fit(features, tag_lists)

    def test(self, sentences: list):
        features = [sent2features(s) for s in sentences]
        pred_tag_lists = self._crf.predict(features)
        return pred_tag_lists
