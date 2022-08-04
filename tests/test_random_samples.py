# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 16:17:56 2022

@author: Wuestney
"""

import numpy as np
import pytest
from sim_utils.random_samples import genRandseq, Simulator
from mc_measures.gen_mc_transition import GenMarkovTransitionProb as MCmatrix
from mc_measures.gen_mc_transition import gen_model, get_model, gen_sample


@pytest.fixture
def simulator_params():
    #set the simulator parameters

    # nsim is the number of independent samples to generate
    nsim = 20
    # seed for the RNG
    seed = None

def set_functype(disttype):
    #set disttype. Options: 'markov', 'uniform', 'regular'
    disttype = 'uniform'
    if disttype == 'uniform':
        func = genRandseq
    elif disttype == 'markov':
        func = gen_sample
    return func

#initiate simulator to handle nsim sequential simulations of the random sample func using the same RNG
#sim = Simulator(func, nsim, seed)


@pytest.fixture
def args_for_genRandseq(alpha_statespace, default_nobs, seed):
    return alpha_statespace, {'nobs' : default_nobs, 'generator':'default', 'seed' : seed}

def test_genRandseq_same_seed(args_for_genRandseq):
    statespace, kwargs = args_for_genRandseq
    randomseq1 = genRandseq(statespace, **kwargs)
    randomseq2 = genRandseq(statespace, **kwargs)
    assert randomseq1 == randomseq2

@pytest.mark.xfail
def test_genRandseq_same_seed(args_for_genRandseq):
    statespace, kwargs = args_for_genRandseq
    randomseq1 = genRandseq(statespace, **kwargs)
    seed = np.random.SeedSequence().entropy
    randomseq2 = genRandseq(statespace, **kwargs)
    assert randomseq1 == randomseq2