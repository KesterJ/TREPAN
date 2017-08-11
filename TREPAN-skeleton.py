# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 14:52:59 2017

@author: Kester
"""
###Draw instances
def draw_instance():



def draw_sample(features, samples):
    """
    A function that takes a set of examples, and draws samples if there are
    fewer than the allowed minimum size for splitting on at a node. (e.g. if
    we want 10,000 examples, and have 9,100, we will draw 900)
    """    
    
###Making M OF N tests
def make_candidate_tests(samples, labels):
    """
    A function that should take all features, all samples, and return the
    the possible breakpoints for each feature. These are the midpoints between
    any two samples that do not have the same label.
    """
    #Combine samples and labels
    combined = np.c_[samples, labels]
    #Create empty dictionary to store features and their breakpoints
    bpdict = {}
    #Loop over each feature (assumes features are columns and samples are rows)
    for feature in range(samples.shape[1]):
        #Sort everything on the current feature
        sortedcombined = combined[combined[:,feature].argsort()]
        breakpoints = []
        sortfeat = sortedcombined[:, feature]
        sortlabel = sortedcombined[:, -1]
        for point in range(sortfeat-1):
            #Check if different classes, find midpoint if so
            if sortlabel[point] != sortlabel[point+1]:
                midpoint = (sortfeat[point]+sortfeat[point+1])/2
                #Check if already in list and add it if not
                if midpoint not in breakpoints:
                    breakpoints.append(midpoint)
        #Add list of breakpoints to feature dict
        bpdict[feature] = breakpoints
    return bpdict
    

def entropy(labels):
    """
    Takes a list of labels, and calculates the entropy. Currently assumes binary
    labels of 0 and 1 - this would need to be altered if multiclass data has
    to be used.
    """
    prob = sum(labels)/len(labels)
    ent = -prob*log(prob) - (1-prob)*log(1-prob)
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

    
def make_mofn_tests(besttest, C, samples):
    """
    Finds the best m-of-n test, using a beam width of 2.
    
    NOTES:
    -NEEDS TO KNOW HOW TO COLLAPSE TESTS WHEN TWO REDUNDANT THINGS ARE
    PRESENT e.g. 2-of {y, z, x, ¬x} -> 1-of {y, z}
    -NEEDS TO KNOW WHICH TESTS WERE ALREADY USED ON THIS BRANCH, AND NOT USE
    THOSE FEATURES AGAIN
    -NEEDS TO AVOID USING TWO TESTS ON THE SAME LITERAL e.g. x > 0.5 and x > 0.7
    -NEEDS TO TEST USING BOTH x and ¬x AS THE SEED FOR BEAM SEARCH
    """

def construct_test(features, samples):
    """
    Takes features and samples, and find the best m-of-n test to split on.
    """
    #make_candidate_tests(features, samples)
    #bestgain = 0;
    #for c in C:
        #if gain(c) > bestgain:
            #besttest = c
    #make_mofn_tests(besttest, C, samples)