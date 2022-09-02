# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 16:33:51 2022
Module intended to check the simulated samples and bit generator states stored in a simulated file 
output from main.run() to make sure all samples were generated without any unexpected overlapping 
bitgenerator states

@author: Wuestney
"""
import json
import pathlib
import argparse
import numpy as np

def load_simulated(filename, verbose=False):
    with open(filename, 'r+') as fhand:
        data = json.load(fhand)
    samples = data['Simulated Samples']
    bgstates = data['BitGenerator states']
    if verbose:
        print("Sample names in data", samples.keys())
        print("Number of sample names", len(samples.keys()))
        for k,v in samples.items():
            print(f"Number of samples for sample {k}", len(v))
        print("Number of items BitGenerator states", len(bgstates.keys()))
        print("Names of items in BitGenerator states", bgstates.keys())
        for sampname in bgstates.keys():
            print(f"BitGenerator states for sample {sampname}:")
            for statedict in bgstates[sampname].keys():
                print(f"State index: {statedict}\tBitGenerator State: ", bgstates[sampname][statedict]['state'])
    return samples, bgstates

def simulator_rng_states_dif_within_samples(simulated, simulatedstates):
    uniquestates = []
    for sampname in simulatedstates.keys():
        nsim = len(simulated[sampname])
        states = [simulatedstates[sampname][statedict]['state']['state'] for statedict in simulatedstates[sampname].keys()]
        if simulatedstates[sampname]['initial state']['state']['state'] == simulatedstates[sampname]['0']['state']['state']:
            #assert that all states besides initial state and first sample state in the bgstateseq are unique
            uniquestates.append(len(set(states)) == (nsim + 1))
        else:
            #assert that all states in the bgstateseq are unique
            uniquestates.append(len(set(states)) == (nsim + 2))
    return all(uniquestates)

def simulator_states_dif_across_samples(simulatedstates):
    # function should map to function of same name in test_random_samples
    prevbatch = list()
    statesincommon = []
    # for the rest of the nobs in the nobs list, get an bool array indicating
    # which bg states are the same as the previous sample batch
    for sampname in simulatedstates.keys():
        states = [simulatedstates[sampname][statedict]['state']['state'] for statedict in simulatedstates[sampname].keys()]
        if len(prevbatch) == 0:
            prevbatch = np.array(states)
        else:
            statesincommon.append(np.equal(np.array(states), prevbatch))
            prevbatch = np.array(states)
    #get number of True elements in each row of statesincommon
    statesincommonsum = np.array(statesincommon).sum(1)
    #if simulations behaved well then should return True
    return all(np.equal(1, statesincommonsum))

def simulator_samps_dif_across_samples(simulated):
    samples = []
    for sampname in simulated.keys():
        samples.append(np.array(simulated[sampname]))
        
    samplesizes = [samp.shape[1] for samp in samples]
    batchsizes = [samp.size for samp in samples]
    smallestsampsize = min(samplesizes)
    smallestbatchsize = min(batchsizes)
    prevbatch = list()
    sampsincommontall = []
    sampsincommonwide = []
    for samp in samples:
        if len(prevbatch) == 0:
            prevbatch = samp
        else:
            sampsincommontall.append(np.array_equal(samp[:, 0:smallestsampsize], prevbatch[:, 0:smallestsampsize]))
            sampsincommonwide.append(np.array_equal(samp.flatten()[0:smallestbatchsize], prevbatch.flatten()[0:smallestbatchsize]))
            prevbatch = samp
    return sum([sum(sampsincommontall), sum(sampsincommonwide)]) == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Check simulated samples.')
    parser.add_argument('dir', type=str,
                        help='path to the directory containing the simulated sample files')
    args = parser.parse_args()
    file_paths = pathlib.Path(args.dir).glob('*.json')
    for file in file_paths:
        #print(file.resolve())
        samples, bgstates = load_simulated(file)
        print("Checking data in ", file.name)
        print("Samples with the same nobs were generated with unique bit generator states:", simulator_rng_states_dif_within_samples(samples, bgstates))
        print("Samples across different nobs were generated with unique bit generator states:", simulator_states_dif_across_samples(bgstates))
        print("Samples generated across different nobs have matching initial values:", simulator_samps_dif_across_samples(samples))
        
    