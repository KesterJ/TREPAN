# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 11:12:52 2017

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

testarr = np.array([[0,1,0], [0,2,0], [0,3,0], [0,4,0]])
testlabels = np.array([0,0,1,1])

mn1 = (1, [(1, 3, True)])

gain = mofn_info_gain(mn1, testarr, testlabels)

testarr2 = np.array([[1,1,1], [2,2,2], [3,3,3], [4,4,4]])

mn2 = (1, [(1, 3, True), (2, 5, True)])

gain2 = mofn_info_gain(mn2, testarr2, testlabels)
