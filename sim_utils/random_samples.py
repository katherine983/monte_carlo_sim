# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 15:38:15 2022

@author: Wuestney
"""
import functools
import numpy as np
from numpy.random import PCG64

# Define functions to handle coordinating the RNG instances and use them to
# generate reproducible random number samples.


def genRandseq(statespace, nobs=100, generator='default', seed=None):
    """
    Generates a random sequence with uniform probability distribution
    from the state space given.

    Parameters
    ----------
    statespace : {INT, STR, ARRAY-LIKE}
        IF statespace IS INTEGER, THIS IS TAKEN TO BE THE SIZE OF THE
        ALPHABET OF THE STATE SPACE OF THE RANDOM VARIABLE. IF A STRING OR ARRAY-LIKE
        OBJECT, statespace IS TAKEN TO BE THE SET OF DISCRETE-VALUES
        COMPRISING THE STATE SPACE OF THE RANDOM VARIABLE.
    nobs : INT, DEFAULT=100
        THE LENGTH OF THE RANDOM SEQUENCE TO BE GENERATED. DEFAULT IS 100 SYMBOLS.
    generator : STRING OR INSTANCE OF numpy.random.Generator CLASS
        DEFAULT IS 'default' WHICH INIDICATES TO USE THE NUMPY default_rng()
        BITGENERATOR WHICH IMPLEMENTS THE CURRENT NUMPY DEFAULT BITGENERATOR.
    seed : {NONE, INT}, OPTIONAL
        SEED TO USE TO FEED THE BITGENERATOR FOR THE RANDOM NUMBER GENERATOR.
        DEFAULT IS NONE.

    Returns
    -------
    List of the randomly generated integers.

    (deprecated)
    states, seq : ARRAY, ARRAY
        RETURNS ARRAY OF THE STATE VALUES AND ARRAY OF THE INDICES OF THE STATES ARRAY
        THAT CAN BE USED TO CREATE THE RANDOM SEQUENCE OF STATES.

        IF THE STATES ARRAY IS JUST A SERIES OF SEQUENTIAL INTEGERS FROM [0:a] THEN THE
        SEQ ARRAY IS THE RANDOM SEQUENCE OF STATES.
    """

    if generator == 'default':
        # Set seed sequence using value given to seed arg.
        # If value is an entropy value from a previous seed sequence, ss will be identical to
        # the previous seed sequence.
        ss = np.random.SeedSequence(seed)
        # save entropy so that seed sequence can be reproduced later
        rng = np.random.default_rng(ss)
    else:
        #expects generator to be an instance of the np.random.BitGenerator class
        rng = np.random.default_rng(generator)

    # a is the cardinality of the statespace
    # states is a numpy array of the states in the statespace
    if type(statespace) is int:
        a = statespace
        states = np.array([i for i in range(a)])
    elif type(statespace) is str:
        a = len(statespace)
        states = np.array(list(statespace))
    elif type(statespace) is np.ndarray or tuple or list:
        a = len(statespace)
        states = np.array(statespace)
    # get sequence of random integers to use to slice states
    randints = rng.integers(a, size=nobs)
    # slice states with randints to get a sequence of random states
    seq = states[randints]
    return seq.tolist()


#define a decorator class to decorate random sample generating functions
#so that they use the same RNG in sequence for all their samples.
class Simulator:
    def __init__(self, func, nsim, seed=None):
        # update wrapper so that the metadata of the returned function is that of func, not the wrapper
        functools.update_wrapper(self, func)
        # func is function to generate random sample using an RNG
        self.func = func
        # nsim is number of random samples to generate
        self.nsim = nsim
        # initiate SeedSequence from seed arg that makes a high-quality seed to initiate the RNG
        # this is the recommended best practice for reproducible results
        # https://numpy.org/doc/stable/reference/random/bit_generators/generated/numpy.random.SeedSequence.html
        self.ss = np.random.SeedSequence(seed)
        self.seed = self.ss.entropy
        # initiate a random number generator using the PCG-64 bitgenerator
        self.rng = np.random.Generator(PCG64(self.ss))
        # empty dict to contain the bitgenerator states at each iteration
        # selg.bgstateseq will become a dict of dicts
        self.bgstateseq = {'initial state' : self.rng.bit_generator.state}

    def __call__(self, *args, **kwargs):
        # define process to run when function decorated by Simulator is called
        # *args and **kwargs to be passed to the sample generating func stored in self.func
        #print('inside __call__')
        self.sample = []
        #def wrapper(self, *args, **kwargs):
        #print('inside wrapped simulate function')

        # set the 'generator' kwarg to be the RNG defined during __init__()
        kwargs['generator'] = self.rng
        for rep in range(self.nsim):
            self.bgstateseq[rep] = self.rng.bit_generator.state
            seq = self.func(*args, **kwargs)
            #print(len(seq))
            self.sample.append(seq)
        self.bgstateseq['end'] = self.rng.bit_generator.state
        # returns list of lists
        # each sublist is the randomly generated symbolic sequence
        return self.sample
        #return wrapper