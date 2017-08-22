# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 19:19:55 2017

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

###Draw instances
def draw_instances(number, kernels, constraints):
    """
    Takes a number of samples to draw, a list of constraints (which consists
    of a list of m-of-n tests that should have been satisfied to reach this
    point, and a boolean variable saying whether the test should have been
    passed or failed) and a set of kernels to draw from, and produces a set
    of samples of that number using the kernels and constraints.
    """
    probslist = []
    feattests = {}
    #Loop over tests for each node so far
    for test in constraints:
        probs = np.array([])
        probfeats = np.array([])
        m = test[0]
        n = len(test[1])
        #Check whether the test should have been passed
        if test[2]:
        #Loop over separate ns in the test
            for feattest in test[1]:
                feature = feattest[0]
                threshold = feattest[1]
                greater = feattest[2]
                #Get conditional probability of passing test by integrating over kernel
                if greater:
                    conditional_prob = kernels[feature].integrate_box_1d(threshold, np.inf)
                else:
                    conditional_prob = kernels[feature].integrate_box_1d(-np.inf, threshold)
                probs = np.append(probs, conditional_prob)
                probfeats = np.append(probfeats,feature)
                feattests[feature] = (threshold, greater)
            probs = probs/sum(probs)
            testprobs = (m, probfeats, probs)
            probslist.append(testprobs)
        #In case the test should be failed, reverse the criteria
        else:
            for feattest in test[1]:
                feature = feattest[0]
                threshold = feattest[1]
                greater = not feattest[2]
                #Get conditional probability of passing test by integrating over kernel
                if greater:
                    conditional_prob = kernels[feature].integrate_box_1d(threshold, np.inf)
                else:
                    conditional_prob = kernels[feature].integrate_box_1d(-np.inf, threshold)
                probs = np.append(probs, conditional_prob)
                probfeats = np.append(probfeats,feature)
                feattests[feature] = (threshold, greater)
            probs = probs/sum(probs)
            #We're doing reverse test, so need one more than n-m tests to be
            #passed to make sure m-of-n test is failed
            testprobs = (n+1-m, probfeats, probs)
            probslist.append(testprobs)
    instances = np.array([draw_instance(kernels, probslist, feattests) for i in range(number)])
    return instances


testdata = np.array([[0,1,2,3,4], [1,1,2,1,1], [0,2,4,6,8], [5,5,1,10,4],
                     [10,11,12,13,14]]).transpose()
testkerns = [stats.gaussian_kde(testdata[:,i]) for i in range(5)]

constraints = [(1, [(3, 5, True), (1, 1.5, False)], True), (1, [(0, 2, True), (4, 12, False)], False)]

testinstances = draw_instances(5000, testkerns, constraints)