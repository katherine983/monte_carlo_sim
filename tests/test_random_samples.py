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

default_nobs_list = [20, 50, 100]
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
    def genRandseq_simulator(self, simulator_params):
        # create instance of Simulator class
        randseq_simulator = Simulator(genRandseq, *simulator_params)
        #returns a simulator twice, for each parametrized nsim value from simulator_params
        return randseq_simulator
    
    @pytest.fixture(scope='class')
    def genRandseq_simulator_clone(self, simulator_params):
        # second instance of Simulator class to use as comparison
        randseq_simulator = Simulator(genRandseq, *simulator_params)
        return randseq_simulator
    
    @pytest.fixture(scope='class')
    def genRandseq_samples(self, alpha4_statespace, default_nobs, genRandseq_simulator_clone):
        # samples generated from a Simulator instance initiated with genRandseq and simulator_params
        sample = genRandseq_simulator_clone(alpha4_statespace, nobs=default_nobs)
        return sample
    
    def test_simulator_seed(self, simulator_params, genRandseq_simulator):
        simulator1 = Simulator(genRandseq, *simulator_params)
        simulator2 = Simulator(genRandseq, *simulator_params)
        assert simulator1.seed == simulator2.seed
        
    def test_simulator_seed2(self, simulator_params, genRandseq_simulator):
        new_simulator = Simulator(genRandseq, *simulator_params)
        assert genRandseq_simulator.seed == new_simulator.seed
        
    def test_simulator_from_saved_seed(self, simulator_params, genRandseq_simulator):
        nsim, seed = simulator_params
        simulator_clone = Simulator(genRandseq, nsim, seed=genRandseq_simulator.seed)
        assert genRandseq_simulator.seed == simulator_clone.seed
        
    def test_simulator_initialstate(self, simulator_params, genRandseq_simulator):
        new_simulator = Simulator(genRandseq, *simulator_params)
        assert genRandseq_simulator.bgstateseq['initial state'] == new_simulator.bgstateseq['initial state']
        
    def test_sample_lengths(self, simulator_params, alpha4_statespace, default_nobs, genRandseq_simulator):
        # test that each sample generated from Simulator instances initiated with the same seeds have the same state sequence
        new_simulator = Simulator(genRandseq, *simulator_params)
        # gen nsim samples of length nobs from each simulator instance
        sample1 = genRandseq_simulator(alpha4_statespace, nobs=default_nobs)
        sample1_array = np.array(sample1)
        sample2 = new_simulator(alpha4_statespace, nobs=default_nobs)
        sample2_array = np.array(sample2)
        assert np.array_equal(sample1_array, sample2_array)
        
    def test_simulator_states_same(self, simulator_params, alpha4_statespace, default_nobs, genRandseq_simulator):
        # test that each sample generated from Simulator instances initiated with the same seeds have the same state sequence
        new_simulator = Simulator(genRandseq, *simulator_params)
        # gen nsim samples of length nobs from each simulator instance
        sample1 = genRandseq_simulator(alpha4_statespace, nobs=default_nobs)
        sample2 = new_simulator(alpha4_statespace, nobs=default_nobs)
        assert genRandseq_simulator.bgstateseq == new_simulator.bgstateseq
    
    #@pytest.mark.xfail(default_nobs == , zip(default_nobs_list, [argvalues)
    def test_simulator_states_same2(self, alpha4_statespace, default_nobs, genRandseq_simulator, genRandseq_simulator_clone):
        # test that genRandseq_simulator.bgstateseq is same as genRandseq_simulator_clone.bgstateseq
        # only after the final iteration of sample generation from genRandseq_simulator
        # using the standard parameters
        sample = genRandseq_simulator(alpha4_statespace, nobs=default_nobs)
        assert genRandseq_simulator.bgstateseq == genRandseq_simulator_clone.bgstateseq
        
        
    # def test_simulator_states_dif(self, alpha4_statespace, default_nobs, genRandseq_simulator_clone):
    #     # test that each sample generated from Simulator instances initiated with different seeds have different state sequences
        
    # @pytest.mark.xfail
    # def test_simulator_stateseq(self, alpha4_statespace, default_nobs, genRandseq_samples, genRandseq_simulator_clone):
    #     #test that each sample generated from the same Simulator instance has a different state sequence
    #     assert 
    
    def test_simulator_samples(self, alpha4_statespace, default_nobs, genRandseq_samples, genRandseq_simulator):
        #test that different simulators with same generating functions initiated with same seed have same samples
        new_sample = genRandseq_simulator(alpha4_statespace, nobs=default_nobs)
        assert genRandseq_samples == new_sample
    
    
        
        