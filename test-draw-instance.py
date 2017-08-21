# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 14:07:20 2017

@author: Kester
"""
import numpy as np
from scipy import stats


def passes_test(resample, test):
    """
    Take a value and a simple greater/less than test, and chceks if the test is
    passed. Returns boolean.
    """
    passes = False
    if (test[1] and resample >= test[0]) or (not test[1] and resample < test[0]):
        passes = True
    return passes


def draw_instance(kernels, condslist, feattests):
    """
    Needs to take a list of kernels and conditional probabilities for m-of-n
    tests, and generate an instance drawn from the kernels that fulfils the
    tests. Feattests is a dictionary of tests keyed by feature.
    """
    featnum = len(kernels)
    instance = np.zeros(featnum)
    constrainedfeatures = []
    for conds in condslist:
        #Choose weighted set of features from m-of-n test
        choices = np.random.choice(conds[1], p=conds[2], size=conds[0], replace=False)
        #Add those chosen to constraints
        constrainedfeatures = np.r_[constrainedfeatures,choices]
    for feature in range(featnum):
        if feature not in constrainedfeatures:
            instance[feature] = kernels[feature].resample(size=1)[0][0]
        else:
            found = False
            while not found:
                resample = kernels[feature].resample(size=1)[0][0]
                if passes_test(resample, feattests[feature]):
                    found = True
            instance[feature] = resample
    return instance

testdata = np.array([[0,1,2,3,4], [1,1,2,1,1], [0,2,4,6,8], [5,5,1,10,4],
                     [10,11,12,13,14]]).transpose()
testkerns = [stats.gaussian_kde(testdata[:,i]) for i in range(5)]

condslist = [(2, [2,3], [0.3,0.7])]
feattests = {1: (3, False), 2: (5, True), 3: (7, False)}

testinst = draw_instance(testkerns, condslist, feattests)
