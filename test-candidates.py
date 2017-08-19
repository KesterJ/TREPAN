# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 20:56:32 2017

@author: Kester
"""

import numpy as np

###Making M OF N tests
def make_candidate_tests(samples, labels):
    """
    A function that should take all features, all samples, and return the
    the possible breakpoints for each feature. These are the midpoints between
    any two samples that do not have the same label.
    """
    #Create empty dictionary to store features and their breakpoints
    bpdict = {}
    #Loop over each feature (assumes features are columns and samples are rows)
    for feature in range(samples.shape[1]):
        #Get unique values for feature
        values = np.unique(samples[:,feature])
        breakpoints = []
        #Loop over values and check if diff classes between values
        for value in range(len(values)-1):
            #Check if different classes in associated labels, find midpoint if so
            labels1 = labels[samples[:,feature]==values[value]]
            labels2 = labels[samples[:,feature]==values[value+1]]
            if list(np.unique(labels1))!=list(np.unique(labels2)):
                midpoint = (values[value]+values[value+1])/2
                breakpoints.append(midpoint)
        #Add list of breakpoints to feature dict
        bpdict[feature] = breakpoints
    return bpdict

test1 = np.array([[1, 1, 1, 1], [2, 5, 2, 2], [3, 3, 6, 0], [4, 4, 4, 100],
                  [5, 5, 2, 5]])
labels1 = np.array([0, 0, 1, 1, 1])

test1dict = make_candidate_tests(test1, labels1)

test2 = np.array([[1, 1], [1, 2], [1,3], [1,3], [1,5]])
labels2 = np.array([0,0,0,1,1])

test2dict = make_candidate_tests(test2, labels2)

test3 = np.array([[1, 1, 1, 1], [2, 5, 2, 2], [3, 3, 6, 0], [4, 4, 4, 100],
                  [5, 5, 2, 5]])
labels3 = np.array([0, 0, 0, 0, 0])

test3dict = make_candidate_tests(test3, labels3)