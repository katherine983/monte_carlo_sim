# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 16:17:56 2022

@author: Wuestney
"""

import numpy as np
from numpy.random import PCG64
import pytest
from sim_utils.random_samples import genRandseq, Simulator
from mc_measures.gen_mc_transition import GenMarkovTransitionProb as MCmatrix
from mc_measures.gen_mc_transition import gen_model, get_model, gen_sample

# @pytest.fixture(scope='module')
# def seed():
#     #seed for the RNG
#     return 70296455059587591496887789747131377293

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

def test_seedsequence_fromseed(seed):
    ss1 = np.random.SeedSequence(seed)
    seed1 = ss1.entropy
    ss2 = np.random.SeedSequence(seed)
    seed2 = ss2.entropy
    assert seed1 == seed2
    

@pytest.fixture
def args_for_genRandseq(alpha_statespace, default_nobs, seed):
    return alpha_statespace, {'nobs' : default_nobs, 'generator':'default', 'seed' : seed}

def test_genRandseq_same_seed(seed):
    # test genRandseq with same seed and default bitgenerator
    randomseq1 = genRandseq(statespace=4, nobs=50, generator='default', seed=seed)
    randomseq2 = genRandseq(statespace=4, nobs=50, generator='default', seed=seed)
    assert np.array_equal(randomseq1, randomseq2)

def test_genRandseq_same_generator(seed):
    # test genRandseq with bit generators initiated with same seed fed to generator arg
    ss = np.random.SeedSequence(seed)
    rng = np.random.Generator(PCG64(ss))
    randomseq1 = genRandseq(statespace=4, nobs=50, generator=rng)
    rng = np.random.Generator(PCG64(ss))
    randomseq2 = genRandseq(statespace=4, nobs=50, generator=rng)
    assert np.array_equal(randomseq1, randomseq2)
    
@pytest.mark.xfail
def test_genRandseq_same_generator(seed):
    # test genRandseq with same bit generator isntance fed to generator arg
    ss = np.random.SeedSequence(seed)
    rng = np.random.Generator(PCG64(ss))
    randomseq1 = genRandseq(statespace=4, nobs=50, generator=rng)
    randomseq2 = genRandseq(statespace=4, nobs=50, generator=rng)
    assert np.array_equal(randomseq1, randomseq2)
    
@pytest.mark.xfail
def test_genRandseq_same_seed(seed):
    #test that sequences generated from different seeds are not equal
    randomseq1 = genRandseq(statespace=4, nobs=50, generator='default', seed=seed)
    newseed = np.random.SeedSequence().entropy
    randomseq2 = genRandseq(statespace=4, nobs=50, generator='default', seed=newseed)
    assert np.array_equal(randomseq1, randomseq2)
    
class TestSimulator:
    @pytest.fixture(scope='class')
    def nsim(self):
        # nsim is the number of independent samples to generate
        nsim = 1
        return nsim
    
    
    @pytest.fixture(scope='class', params=[1, 5])
    def simulator_params(self, request, seed):
        #set the simulator parameters
    
        # nsim is the number of independent samples to generate
        nsim = request.param
        # seed for the RNG
        seed = seed
        return nsim, seed
    
    @pytest.fixture(scope='function')
    def genRandseq_simulator(self, nsim, seed):
        # create instance of Simulator class
        randseq_simulator = Simulator(genRandseq, nsim, seed)
        return randseq_simulator
    
    @pytest.fixture(scope='function')
    def genRandseq
    
    def test_simulator_seed(self, simulator_params, genRandseq_simulator):
        simulator1 = Simulator(genRandseq, *simulator_params)
        simulator2 = Simulator(genRandseq, *simulator_params)
        assert simulator1.seed == simulator2.seed
        
    def test_simulator_seed(self, simulator_params, genRandseq_simulator):
        new_simulator = Simulator(genRandseq, *simulator_params)
        assert genRandseq_simulator.seed == new_simulator.seed
        
    def test_simulator_from_saved_seed(self, simulator_params, genRandseq_simulator):
        nsim, seed = simulator_params
        simulator_clone = Simulator(genRandseq, nsim, seed=genRandseq_simulator.seed)
        assert genRandseq_simulator.seed == simulator_clone.seed
        
    def test_
    #test that simulators with different generating functions initiated with same seed have same state sequences
    
    #test that each sample generated from the same Simulator instance has a different state sequence
        
        