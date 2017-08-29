# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 21:48:34 2017

@author: Kester
"""


import numpy as np
import pandas as pd
from scipy import stats


###SAMPLES DRAWING CODE BEGINS

def passes_test(resample, test):
    """
    Take a value and a simple greater/less than test, and chceks if the test is
    passed. Returns boolean.
    """
    passes = False
    if (test[1] and resample >= test[0]) or (not test[1] and resample < test[0]):
        passes = True
    return passes

def draw_instance(kernels, condslist, feattests, singlevalsdict):
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
        if feature in singlevalsdict:
            instance[feature] = singlevalsdict[feature]
        elif feature not in constrainedfeatures:
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
def draw_instances(number, kernels, constraints, singlevalsdict):
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
                if feature in singlevalsdict:
                    if (greater and singlevalsdict[feature] >=threshold) or ((not greater) and singlevalsdict[feature] < threshold):
                        #If it's guaranteed to be passed in test, reduce m for other values
                        #(Implied don't reduce m if not passed in test)
                        m -= 1
                else:    
                    #Get conditional probability of passing test by integrating over kernel
                    if greater:
                        conditional_prob = kernels[feature].integrate_box_1d(threshold, np.inf)
                    else:
                        conditional_prob = kernels[feature].integrate_box_1d(-np.inf, threshold)
                    probs = np.append(probs, conditional_prob)
                    probfeats = np.append(probfeats,feature)
                    feattests[feature] = (threshold, greater)
            #Equalise probabilites to sum to 1, alter m in case of single values
            #and add test to list
            probs = probs/sum(probs)
            testprobs = (m, probfeats, probs)
            probslist.append(testprobs)
        #In case the test should be failed, reverse the criteria
        else:
            for feattest in test[1]:
                feature = feattest[0]
                threshold = feattest[1]
                greater = not feattest[2]
                if feature in singlevalsdict:
                    if (greater and singlevalsdict[feature] >=threshold) or ((not greater) and singlevalsdict[feature] < threshold):
                        #If it's guaranteed to be passed in test, reduce m for other values
                        #(Implied don't reduce m if not passed in test)
                        m -= 1
                else:
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
    instances = np.array([draw_instance(kernels, probslist, feattests, singlevalsdict) for i in range(number)])
    return instances
    
def draw_sample(samples, total, significance, constraints, parentsamples):
    """
    A function that takes a set of examples, and draws samples if there are
    fewer than the allowed minimum size for splitting on at a node. (e.g. if
    we want 10,000 examples, and have 9,100, we will draw 900)
    """
    singlevalsdict = {}
    samplesneeded = total - samples.shape[0]
    numfeats = samples.shape[1]
    bandwidth = 1/np.sqrt(samples.shape[0])
    print('Making kernels...')
    if samplesneeded > 0:
        kernels = [None for i in range(numfeats)]
        for feat in range(numfeats):
            #Check if distribution for feature is diff from parent node
            #Including Bonferroni correction
            if stats.ks_2samp(samples[:,feat], parentsamples[:,feat])[1] <= significance/numfeats:
                #Check for single values
                uniques = np.unique(samples[:,feat])
                if len(uniques)==1:
                    singlevalsdict[feat] = uniques[0]
                else:
                    kernels[feat] = stats.gaussian_kde(samples[:,feat], bw_method=bandwidth)
            else:
                #Same as above, but for parent node instead
                uniques = np.unique(parentsamples[:,feat])
                if len(uniques)==1:
                    singlevalsdict[feat] = uniques[0]
                else:
                    kernels[feat] = stats.gaussian_kde(parentsamples[:,feat], bw_method=bandwidth)
        print('Drawing instances...')
        newsamples = draw_instances(samplesneeded, kernels, constraints, singlevalsdict)
        allsamples = np.r_['0,2',samples,newsamples]
    else:
        allsamples = samples
    return allsamples

###SAMPLE DRAWING CODE ENDS

###TREE BUILDING CODE BEGINS

def passes_mn_test(example, test):
    """
    Checks if a particular example passes a particular m-of-n test.
    """
    testpassed = False
    counter = 0
    featurespassed = 0
    m = test[0]
    n = len(test[1])
    while (not testpassed) and counter<n:
        feature = test[1][counter][0]
        threshold = test[1][counter][1]
        greater = test[1][counter][2]
        if (greater and example[feature] >= threshold) or ((not greater) and example[feature] < threshold):
            featurespassed += 1
        if featurespassed >= m:
            testpassed = True
        counter += 1
    return testpassed

def passes_mn_tests(example, constraints):
    """
    Loops over a list of m-of-n tests and checks if all are passed.
    """
    allpassed = True
    counter = 0
    while allpassed and counter<len(constraints):
        passed = passes_mn_test(example, (constraints[counter][0], constraints[counter][1]))
        if passed != constraints[counter][2]:
            allpassed = False
        counter += 1
    return allpassed

def create_node(constraints, mntest, passed, parent, samples, oracle):
    """
    Creates a basic node.
    """
    nodedict = {}
    #Get all constraints
    allconstraints = list(constraints)
    allconstraints.append((mntest[0], mntest[1], passed))
    nodedict['constraints'] = allconstraints
    #Calculate reach
    reaches = [passes_mn_tests(samples[i,:], allconstraints) for i in range(samples.shape[0])]
    nodedict['reaches'] = reaches
    reach = sum(reaches)/samples.shape[0]
    nodedict['reach'] = reach
    #Calculate fidelity
    localsamples = samples[reaches]
    labels = [float(oracle.predict(localsamples[i,:].reshape(1,-1))) for i in range(sum(reaches))]
    if len(stats.mode(labels).mode)>0:
        predictedclass = stats.mode(labels).mode[0]
    else:
        predictedclass = None
    nodedict['predictedclass'] = predictedclass
    fidelity = sum(labels==predictedclass)/sum(reaches)
    nodedict['fidelity'] = fidelity
    #Mark as leaf if all labels same
    if len(np.unique(labels))==1:
        nodedict['finished'] = True
    else:
        nodedict['finished'] = False
    #Add parent
    nodedict['parent'] = parent
    return nodedict
    
def expand_node(constraints, samples, nodename, tree, oracle):
    """
    Expands a node (which should have been created earlier as a shell) by
    generating samples, testing on the oracle and constructing an m-of-n
    test.
    """
    localsamples = samples[tree[nodename]['reaches']]
    total = 1000
    significance = 0.05
    print('Drawing samples...')
    newsamples = draw_sample(localsamples, total, significance, constraints, samples[tree[tree[nodename]['parent']]['reaches']])
    print('Getting labels...')
    newlabels = np.array([float(oracle.predict(newsamples[i,:].reshape(1,-1))) for i in range(newsamples.shape[0])])
    #Only use features in the m-of-n test construction that haven't been used
    #in previous tests on this branch.
    alreadyused = []
    for test in constraints:
        for subtest in test[1]:
            alreadyused.append(subtest[0])
    print('Already used', alreadyused)
    mntest = construct_test(newsamples, newlabels, alreadyused)
    #Add test to the node
    tree[nodename]['mntest'] = mntest
    #Generate daughter nodes
    tree[nodename + '0'] = create_node(constraints, mntest, False, nodename, samples, oracle)
    tree[nodename + '1'] = create_node(constraints, mntest, True, nodename, samples, oracle)
    tree[nodename]['0daughter'] = nodename+'0'
    tree[nodename]['1daughter'] = nodename+'1'
    tree[nodename]['predictedclass'] = None
    tree[nodename]['finished'] = True

def create_initial_node(samples, oracle, nodename, tree):
    """
    Creates first node in tree.
    """
    nodedict = {'constraints': [], 'reach': 1, 'finished': True,
                'fidelity': 1, 'reaches': [True for i in range(samples.shape[0])]}
    total = 1000
    significance = 0.05
    newsamples = draw_sample(samples, total, significance, nodedict['constraints'], samples)
    labels = np.array([float(oracle.predict(newsamples[i,:].reshape(1,-1))) for i in range(newsamples.shape[0])])
    mntest = construct_test(newsamples, labels, [])
    nodedict['mntest'] = mntest
    tree[nodename + '0'] = create_node([], mntest, False, nodename, samples, oracle)
    tree[nodename + '1'] = create_node([], mntest, True, nodename, samples, oracle)
    nodedict['0daughter'] = nodename + '0'
    nodedict['1daughter'] = nodename + '1'
    return nodedict

def find_best_node(tree):
    """Returns the node in a tree that has the best (highest) candidate value
    for expansion, as defined by reach(N)*(1-fidelity(N)).
    """
    bestval = 0
    for node in tree:
        if tree[node]['fidelity'] == 1 or tree[node]['finished']:
            candidateval = 0
        else:
            candidateval = tree[node]['reach']*(1-tree[node]['fidelity'])
        if candidateval > bestval:
            bestval = candidateval
            bestnode = node
    return bestnode

    
def build_tree(samples, oracle, size):
    """
    Whole tree building process.
    """
    tree = {}
    tree['1'] = create_initial_node(samples, oracle, '1', tree)
    nodestoexpand = True
    while (len(tree) < size) and nodestoexpand:
        #Find best node
        bestnode = find_best_node(tree)
        #Expand it
        print("Best node: ",bestnode)
        expand_node(tree[bestnode]['constraints'], samples, bestnode, tree, oracle)
        #Check there are still nodes to expand
        nodestoexpand = False
        for node in tree:
            if not tree[node]['finished']:
                nodestoexpand = True
    return tree
###TREE BUILDING CODE ENDS


###MAKING M OF N TEST CODE BEGINS

def make_candidate_tests(samples, labels, alreadyused):
    """
    A function that should take all features, all samples, and return the
    the possible breakpoints for each feature. These are the midpoints between
    any two samples that do not have the same label.
    """
    #Create empty dictionary to store features and their breakpoints
    bpdict = {}
    #Loop over each feature (assumes features are columns and samples are rows)
    for feature in range(samples.shape[1]):
        #Only generate breakpoints for a feature if it wasn't already used on this branch
        if feature not in alreadyused:
            #Get unique values for feature
            values = np.unique(samples[:,feature])
            breakpoints = []
            #Loop over values and check if diff classes between values
            for value in range(len(values)-1):
                #Check if different classes in associated labels, find midpoint if so
                labels1 = labels[samples[:,feature]==values[value]]
                labels2 = labels[samples[:,feature]==values[value+1]]
                l1unique = list(np.unique(labels1))
                l2unique = list(np.unique(labels2))
                if l1unique!=l2unique or (l1unique==l2unique==[0, 1]):
                    midpoint = (values[value]+values[value+1])/2
                    breakpoints.append(midpoint)
            #Trim list of breakpoints to 20 if too long
            if len(breakpoints)>20:
                idx = np.rint(np.linspace(0,len(breakpoints)-1, num =20)).astype(int)
                breakpoints = [breakpoints[i] for i in idx]
            #Add list of breakpoints to feature dict
            bpdict[feature] = breakpoints
    return bpdict
    

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
    #Create the test
    if incrementm:
        newm = test[0]+1
    else:
        newm = test[0]
    newfeats = list(test[1])
    newfeats.append((feature, threshold, greater))
    newtest = (newm, newfeats)
    return newtest

    
def make_mofn_tests(besttest, tests, samples, labels, improvement):
    """
    Finds the best m-of-n test, using a beam width of 2.
    improvement is the percentage by which gain should improve on addition of a
    new test. (Can be from 1.0+)
    
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
    n = 1
    #Set up loop to repeat until beam isn't changed
    while beamchanged:
        print('Test of size %d...'%n)
        n = n+1
        beamchanged = False
        #Loop over the current best m-of-n tests in beam
        for test in beam:
            #Get features used in the test already
            existingfeats = [subtest[0] for subtest in test[1]]
            #Loop over the single-features in candidate tests dict
            for feature in tests:
                #Check it hasn't been used in the test already
                if feature not in existingfeats:
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
                                if gain > improvement*min(currentgains):
                                    #Replace worst in beam if gain better than worst in beam
                                    currentbeam[np.argmin(currentgains)] = newtest
                                    currentgains[np.argmin(currentgains)] = gain
                                    beamchanged = True
        #Set new tests in beam and associated gains
        beam = list(currentbeam)
        beamgains = list(currentgains)
    #Return the best test in beam
    return beam[np.argmax(beamgains)]
    
def construct_test(samples, labels, alreadyused):
    """
    Takes samples and labels, and find the best m-of-n test to split on.
    """
    tests = make_candidate_tests(samples, labels, alreadyused)
    bestgain = 0;
    print('Finding best test...')
    for test in tests:
        for threshold in tests[test]:
            testgain = binary_info_gain(test, threshold, samples, labels)
            if testgain > bestgain:
            ###TODO: Write the binary_info_gain call above correctly    
                bestgain = testgain
                besttest = (test, threshold)
    print('Done.')
    improvement = 1.1
    mofntest = make_mofn_tests(besttest, tests, samples, labels, improvement)
    return mofntest

###MAKING M OF N TEST CODE ENDS

from sklearn.neural_network import MLPClassifier

###LOAD DATA
data = pd.read_csv('../../Data from Transparency Project/analytic.csv')

###SPLIT OUT
features = data.drop(['USERID', 'RG_case', 'ValidationSet', 'filter_$', 'RiskGroup1',
                      'first_active_product1_31days', 'first_active_product2_31days',
                      'first_active_product4_31days', 'RiskGroup2', 'RiskGroupCombined',
                      'first_active_games_31days', 'first_active_poker_31days',
                      'mostFrequentGame', 'playedFO', 'playedLA', 'playedPO',
                      'playedCA', 'playedGA', 'gender'], axis = 1)
labels = data['RG_case']
valsplit = data['ValidationSet']
featnum = features.shape[1]
featnames = features.columns.values.tolist()

trainfeats = features[valsplit==0].as_matrix()
trainlabels = labels[valsplit==0].as_matrix()
testfeats = features[valsplit==1].as_matrix()
testlabels = labels[valsplit==1].as_matrix()

###CLEAN BLANK SPACES
trainfeats[np.where(trainfeats == ' ')] = '0'
trainlabels[np.where(trainlabels == ' ')] = '0'
testfeats[np.where(testfeats == ' ')] = '0'
testlabels[np.where(testlabels == ' ')] = '0'

###CONVERT TO FLOAT
trainfeats = trainfeats.astype(np.float32)
trainlabels = trainlabels.astype(np.int)
testfeats = testfeats.astype(np.float32)
testlabels = testlabels.astype(np.int)

nn = MLPClassifier(hidden_layer_sizes=100, activation = 'logistic')
nn.fit(trainfeats, trainlabels)

testtree = build_tree(trainfeats[1:1500,:], nn, 10)