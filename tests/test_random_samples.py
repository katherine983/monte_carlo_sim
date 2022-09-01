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


def idfn(val):
    if isinstance(val, (int,)):
        return f"Nval"

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
        nsim = 5
        return nsim
    
    
    @pytest.fixture(scope='class', params=[1, 5, 20])
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
    
    @pytest.fixture(scope='class', params=default_nobs_list)
    def genRandseq_samples(self, request, alpha4_statespace, genRandseq_simulator_clone):
        # samples generated from a Simulator instance initiated with genRandseq and simulator_params
        sample = genRandseq_simulator_clone(alpha4_statespace, nobs=request.param)
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
        
    @pytest.mark.parametrize("samplesize", default_nobs_list, ids=idfn)
    def test_sample_lengths(self, samplesize, simulator_params, alpha4_statespace, genRandseq_simulator):
        # test that each sample generated from Simulator instances initiated with the same seeds have the same state sequence
        new_simulator = Simulator(genRandseq, *simulator_params)
        # gen nsim samples of length nobs from each simulator instance
        sample1 = genRandseq_simulator(alpha4_statespace, nobs=samplesize)
        sample1_array = np.array(sample1)
        sample2 = new_simulator(alpha4_statespace, nobs=samplesize)
        sample2_array = np.array(sample2)
        assert np.array_equal(sample1_array, sample2_array)
        
    def test_simulator_states_same(self, simulator_params, alpha4_statespace, default_nobs, genRandseq_simulator):
        # test that each sample generated from Simulator instances initiated with the same seeds have the same state sequence
        new_simulator = Simulator(genRandseq, *simulator_params)
        # gen nsim samples of length nobs from each simulator instance
        sample1 = genRandseq_simulator(alpha4_statespace, nobs=default_nobs)
        sample2 = new_simulator(alpha4_statespace, nobs=default_nobs)
        assert genRandseq_simulator.bgstateseq == new_simulator.bgstateseq
    
    #@pytest.mark.parametrize("nobs", [pytest.param(20, marks=pytest.mark.xfail), pytest.param(50, marks=pytest.mark.xfail), 100])
    
    def test_simulator_states_same2(self, nsim, seed, alpha4_statespace, default_nobs):
        # test that sim1.rng.bit_generator.state is same as sim2.rng.bit_generator.state
        # after each sample generation
        sim1 = Simulator(genRandseq, nsim, seed)
        sim2 = Simulator(genRandseq, nsim, seed)
        sample1 = sim1(alpha4_statespace, nobs=default_nobs)
        sample2 = sim2(alpha4_statespace, nobs=default_nobs)
        assert sim1.rng.bit_generator.state == sim2.rng.bit_generator.state
        
    @pytest.fixture
    def genRandseq_1simulated(self, simulator_params, default_nobs_list, alpha4_statespace):
        # return all samples for all nobs and stateseq dicts for all samples
        
        simulator = Simulator(genRandseq, *simulator_params)
        #create dict to keep a log of bitgenerator state sequences for each sample
        simulatorstates = {}
        #empty dict to hold simulated datasets
        simulated = {}
        for n in default_nobs_list:
            sampname = f"N{n}"
            samples = simulator(alpha4_statespace, nobs=n)
            simulatorstates[sampname] = simulator.bgstateseq
            simulated[sampname] = samples
        return simulatorstates, simulated
    
    @pytest.fixture
    def genRandseq_2simulated_same(self, simulator_params, default_nobs_list, alpha4_statespace):
        # returns all samples for all nobs and stateseq dicts for all samples from two identical Simulators
        
        sim1 = Simulator(genRandseq, *simulator_params)
        #create dict to keep a log of bitgenerator state sequences for each sample
        simulatorstates1 = {}
        #empty dict to hold simulated datasets
        simulated1 = {}
        for n in default_nobs_list:
            sampname = f"N{n}"
            samples = sim1(alpha4_statespace, nobs=n)
            simulatorstates[sampname] = sim1.bgstateseq
            simulated[sampname] = samples
        
        sim2 = Simulator(genRandseq, *simulator_params)
        #create dict to keep a log of bitgenerator state sequences for each sample
        simulatorstates2 = {}
        #empty dict to hold simulated datasets
        simulated2 = {}
        for n in default_nobs_list:
            sampname = f"N{n}"
            samples = sim2(alpha4_statespace, nobs=n)
            simulatorstates[sampname] = sim2.bgstateseq
            simulated[sampname] = samples
        return {"sim1 bgstates": simulatorstates1, "sim1 samples" : simulated1, 
                "sim2 bgstates": simulatorstates2, "sim2 samples" : simulated2}
    
    @pytest.fixture
    def genRandseq_2simulated_dif(self, simulator_params, default_nobs_list, alpha4_statespace):
        # returns all samples for all nobs and stateseq dicts for all samples from two identical Simulators
        
        sim1 = Simulator(genRandseq, *simulator_params)
        #create dict to keep a log of bitgenerator state sequences for each sample
        simulatorstates1 = {}
        #empty dict to hold simulated datasets
        simulated1 = {}
        for n in default_nobs_list:
            sampname = f"N{n}"
            samples = sim1(alpha4_statespace, nobs=n)
            simulatorstates[sampname] = sim1.bgstateseq
            simulated[sampname] = samples
            
        #sim2 initiated with random seed
        sim2 = Simulator(genRandseq, *simulator_params)
        #create dict to keep a log of bitgenerator state sequences for each sample
        simulatorstates2 = {}
        #empty dict to hold simulated datasets
        simulated2 = {}
        for n in default_nobs_list:
            sampname = f"N{n}"
            samples = sim2(alpha4_statespace, nobs=n)
            simulatorstates[sampname] = sim2.bgstateseq
            simulated[sampname] = samples
        return {"sim1 bgstates": simulatorstates1, "sim1 samples" : simulated1, 
                "sim2 bgstates": simulatorstates2, "sim2 samples" : simulated2}
    
    def test_simulator_rng_states_dif_within_samples(self, nsim, seed, alpha4_statespace, default_nobs):
        # test that each sample generated for a given nobs from a single Simulator instance have different bgstates

        sim1 = Simulator(genRandseq, nsim)
        sampname = f"N{default_nobs}"
        samples = sim1(alpha4_statespace, default_nobs)
        simulatedstates = {sampname : sim1.bgstateseq}
        assert len(samples) == nsim
        states = [simulatedstates[sampname][statedict]['state'] for statedict in simulatedstates[sampname].keys()]
        if simulatedstates[sampname]['initial state']['state'] == simulatedstates[sampname][0]['state']:
            #assert that all states besides initial state and first sample state in the bgstateseq are unique
            assert len(set(states)) == (nsim + 1)
        else:
            #assert that all states in the bgstateseq are unique
            assert len(set(states)) == (nsim + 2)

    def test_simulator_states_dif_across_samples(self, genRandseq_1simulated, default_nobs_list):
        # test that each sample generated for every nobs from a single Simulator instance have different bgstates
        # genRandseq_1simulated uses same default_nobs_list to define the nobs
        simulatedstates, simulated = genRandseq_1simulated
        smallestsampsize = min(default_nobs_list)
        firstsampname = f"N{default_nobs_list[0]}"
        #get array of states of the batch of samples for the first nobs generated from the simulator
        firstsampstates = np.array([simulatedstates[firstsampname][statedict]['state'] for statedict in simulatedstates[firstsampname].keys()])
        prevbatch = list()
        statesincommon = []
        # for the rest of the nobs in the nobs list, get an bool array indicating
        # which bg states are the same as the first sample batch
        for nobs in default_nobs_list[1:]:
            sampname = f"N{nobs}"
            assert np.array(simulated[sampname]).shape[1] == nobs
            states = [simulatedstates[sampname][statedict]['state'] for statedict in simulatedstates[sampname].keys()]
            if len(prevbatch) == 0:
                prevbatch = np.array(states)
            statesincommon.append(np.equal(np.array(states), firstsampstates))
        #get number of True elements in each row of statesincommon
        statesincommonsum = np.array(statesincommon).sum(1)
        # if sample batches' bgstateseqs are non-overlapping then the sum of elements that equal eachother should be 1 (for the initial state)
        assert statesincommon

    def test_simulator_states_dif_across_simulators(self, simulator_params, alpha4_statespace, default_nobs_list, genRandseq_simulator):
        # test that each sample generated from Simulator instances initiated with different seeds have different state sequences
        nsim, seed = simulator_params
        new_simulator = Simulator(genRandseq, nsim)
        newsimulated = {}
        for n in nobs:
            name = f"N{n}"
            samples
    # @pytest.mark.xfail
    # def test_simulator_stateseq(self, alpha4_statespace, default_nobs, genRandseq_samples, genRandseq_simulator_clone):
    #     #test that each sample generated from the same Simulator instance has a different state sequence
    #     assert 
    
    def test_simulator_samples(self, simulator_params, default_nobs, alpha4_statespace):
        #test that different simulators with same generating functions initiated with same seed have same samples
        sim1 = Simulator(genRandseq, *simulator_params)
        sim2 = Simulator(genRandseq, *simulator_params)
        sample1 = sim1(alpha4_statespace, nobs=default_nobs)
        sample2 = sim2(alpha4_statespace, nobs=default_nobs)
        old_sample = np.array(sample1)
        new_sample = np.array(sample2)
        assert np.array_equal(old_sample, new_sample)
    
    
    
        
        