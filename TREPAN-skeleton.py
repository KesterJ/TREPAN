# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 14:52:59 2017

@author: Kester
"""


###Making M OF N tests
def make_candidate_tests(features, samples):
    """
    A function that should take all features, all samples, and return a test
    for each feature.
    HOW TO DETERMINE THE THRESHOLD?
    """
    
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