# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 10:58:08 2017

@author: Kester
"""

import numpy as np

def entropy(labels):
    """
    Takes a list of labels, and calculates the entropy. Currently assumes binary
    labels of 0 and 1 - this would need to be altered if multiclass data has
    to be used.
    """
    #Check there are any labels to work with
    if len(labels)==0:
        return 0
    prob = sum(labels)/len(labels)
    #Deal with the case where one class isn't present (would return nan without
    #this exception)
    if prob in [0,1]:
        ent = 0
    else:
        ent = -prob*np.log2(prob) - (1-prob)*np.log2(1-prob)
    return ent


def info_gain(feature, threshold, samples, labels):
    """
    Takes a feature and a threshold, examples and their
    labels, and find the best feature and breakpoint to split on to maximise
    information gain.
    Assumes only two classes. Would need to be altered if more are required.
    """
    #Get initial entropy
    origent = entropy(labels)
    #Get two halves of threshold
    split1 = samples[:, feature]>=threshold
    split2 = np.invert(split1)
    #Get entropy after split (remembering to weight by no of examples in each
    #half of split)
    afterent = (entropy(labels[split1])*(sum(split1)/len(labels)) + 
                entropy(labels[split2])*(sum(split2)/len(labels)))
    gain = origent - afterent
    return gain

test1 = np.array([[1, 2], [2, 2], [1, 5], [3, 1]])
labels1 = np.array([0, 1, 0, 1])
feat1 = 0
thresh1 = 1.5
feat2 = 1
thresh2 = 2.5

test2 = np.random.rand(10, 2)
labels2 = np.array([0, 1, 1, 1, 0, 1, 0, 0, 1, 1])

testans1 = info_gain(feat1, thresh1, test1, labels1)
print(testans1)

testans2 = info_gain(feat2, thresh2, test1, labels1)
print(testans2)

testans3 = info_gain(feat1, 0.3, test2, labels2)
print(testans3)