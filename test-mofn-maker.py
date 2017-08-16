# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 23:46:34 2017

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


def binary_info_gain(feature, threshold, samples, labels):
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


def mofn_info_gain(mofntest, samples, labels):
    """
    Takes an m-of-n test with a set of samples and labels, and calculates the
    information gain provided by that test.
    
    Structure of an m-of-n test object:
    (m, [(feat_1, thresh_1, greater_1)...(feat_n, thresh_n, greater_n)])
    Where m = number of tests that must be passed (i.e. m in m-of-n)
    feat_i = the feature of the ith test
    thresh_i = the threshold of the ith test
    greater_i = a boolean: if true, value must be >= than threshold to pass
              the test. If false, it must be < threshold.
    """
    #Unpack the tests structure
    m = mofntest[0]
    septests = mofntest[1]    
    #List comprehension to generate a boolean index that tells us which samples
    #passed the test.
    splittest = np.array([samples[:,septest[0]]>=septest[1] if septest[2] else 
                 samples[:,septest[0]]<septest[1] for septest in septests])
    #Now check whether the number of tests passed per sample is higher than m
    split1 = sum(splittest)>=m
    split2 = np.invert(split1)
    #Calculate original entropy
    origent = entropy(labels)
    #Get entropy of split
    afterent = (entropy(labels[split1])*(sum(split1)/len(labels)) + 
                entropy(labels[split2])*(sum(split2)/len(labels)))
    gain = origent - afterent
    return gain

def expand_mofn_test(test, feature, threshold, greater, incrementm):
    """
    Constructs and returns a new m-of-n test using the passed test and 
    other parameters.
    """
    #Check for feature redundancy
    for feat in test[1]:
        if feature==feat[0]:
            if greater==feat[2]:
                #Just return the unmodified existing test if we'd add a threshold
                #with same feature and sign
                return test
            else:
                #Also just return the same if the two tests would overlap
                if (greater and feature <= feat[1]) or (not greater and feature >= feat[1]):
                    return test
    #If we didn't find redundancy, actually create the test
    if incrementm:
        newm = test[0]+1
    else:
        newm = test[0]
    newfeats = list(test[1])
    newfeats.append((feature, threshold, greater))
    newtest = (newm, newfeats)
    return newtest

    
def make_mofn_tests(besttest, tests, samples, labels):
    """
    Finds the best m-of-n test, using a beam width of 2.
    
    NOTES:
    -NEEDS TO KNOW HOW TO COLLAPSE TESTS WHEN TWO REDUNDANT THINGS ARE
    PRESENT e.g. 2-of {y, z, x, ¬x} -> 1-of {y, z}
    -NEEDS TO KNOW WHICH TESTS WERE ALREADY USED ON THIS BRANCH, AND NOT USE
    THOSE FEATURES AGAIN - CAN BE DONE OUTSIDE FUNCTION BY PASSING SUBSET 
    OF SAMPLES 
    -NEEDS TO AVOID USING TWO TESTS ON THE SAME LITERAL e.g. x > 0.5 and x > 0.7
    """
    #Initialise beam with best test and its negation
    initgain = binary_info_gain(besttest[0], besttest[1], samples, labels)
    beam = [(1,[(besttest[0], besttest[1], False)]), (1,[(besttest[0], besttest[1], True)])]
    beamgains = [initgain, initgain]
    #Initialise current beam (which will be modified within the loops)
    currentbeam = list(beam)
    currentgains = list(beamgains)
    beamchanged = True
    #Set up loop to repeat until beam isn't changed
    while beamchanged:
        beamchanged = False
        #Loop over the current best m-of-n tests in beam
        for test in beam:
            #Loop over the single-features in candidate tests dict
            for feature in tests:
                #Loop over the thresholds for the feature
                for threshold in tests[feature]:
                    #Loop over greater than/lesser than tests
                    for greater in [True, False]:
                        #Loop over m+1-of-n+1 and m-of-n+1 tests
                        for incrementm in [True, False]:
                            #Add selected feature+threshold to to current test
                            newtest = expand_mofn_test(test, feature, threshold, greater, incrementm)
                            #Get info gain and compare it
                            gain = mofn_info_gain(newtest, samples, labels)
                            #Compare gains
                            if gain > min(currentgains):
                                #Replace worst in beam if gain better than worst in beam
                                currentbeam[np.argmin(currentgains)] = newtest
                                currentgains[np.argmin(currentgains)] = gain
                                beamchanged = True
        #Set new tests in beam and associated gains
        beam = list(currentbeam)
        beamgains = list(currentgains)
    #Return the best test in beam
    return beam[np.argmax(beamgains)]

testarr = np.array([[1,1,1], [1,1,2], [1,2,1], [2,1,1], [1,2,2], [2,1,2], [2,2,1], [2,2,2]])
testlabels = np.array([0,1,0,0,1,0,0,0])

besttest = [2, 1.5]
tests = {0: [0.5, 1.5, 2.5], 1: [0.5, 1.5, 2.5], 2: [0.5, 1.5, 2.5]}

mofn = make_mofn_tests(besttest, tests, testarr, testlabels)