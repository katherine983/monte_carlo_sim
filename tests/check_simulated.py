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
    samples = data['Simulated Samples Location']
    for sampname, samppath in samples.items():
        fname = pathlib.Path(samppath).name
        newpath = pathlib.Path(filename).parent / fname
        samples[sampname] = newpath
    bgstates = data['BitGenerator states']
    if verbose:
        print("Sample names in data", samples.keys())
        print("Number of sample names", len(samples.keys()))
        print("Number of items in BitGenerator states", len(bgstates.keys()))
        print("Names of items in BitGenerator states", bgstates.keys())
        
        for sampname in bgstates.keys():
            with np.load(samples[sampname]) as sampnpz:
                nsim = len(sampnpz.files)
                print(f"Number of samples for sample {sampname}", nsim)
            print(f"BitGenerator states for sample {sampname}:")
            for statedict in bgstates[sampname].keys():
                print(f"State index: {statedict}\tBitGenerator State: ", bgstates[sampname][statedict]['state'])
    #changes directory path for the simulated sample data to match the path of filename
    return samples, bgstates

def simulator_rng_states_dif_within_samples(simulated, simulatedstates):
    uniquestates = []
    for sampname in simulatedstates.keys():
        with np.load(simulated[sampname]) as sampnpz:
            nobs = len(sampnpz['arr_0'])
            nsim = len(sampnpz.files)
        states = [simulatedstates[sampname][statedict]['state']['state'] for statedict in simulatedstates[sampname].keys()]
        if simulatedstates[sampname]['initial state']['state']['state'] == simulatedstates[sampname]['0']['state']['state']:
            #assert that all states besides initial state and first sample state in the bgstateseq are unique
            uniquestates.append(len(set(states)) == (nsim + 1))
        else:
            #assert that all states in the bgstateseq are unique
            uniquestates.append(len(set(states)) == (nsim + 2))
    #if simulation behaved correctly then should return True
    return all(uniquestates)

def simulator_states_dif_across_samples(simulatedstates):
    # function should map to function of same name in test_random_samples
    prevbatch = list()
    statesincommon = []
    # if there is only one set of samples corresponding to one nobs
    # then state overlap across samples is Not applicable
    if len(simulatedstates.keys()) == 1:
        return "Not applicable"
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

def simulator_samps_dif_across_samples(simulated, datadir):
    #simulated is the json data output from sim_files.sim_data_dump()
    #datadir is directory containing the simulated data output
    #list of sample sizes for each set of samples
    samplesizes = []
    #list total observations in each set of samples
    batchsizes = []
    samplepaths = []
    for sampname, samppath in samples.items():
        fname = pathlib.Path(samppath).name
        samppath = pathlib.Path(datadir).joinpath(fname)
        samplepaths.append(samppath)
        with np.load(samppath) as sampnpz:
            nobs = len(sampnpz['arr_0'])
            nsim = len(sampnpz.files)
            samplesizes.append(nobs)
            batchsizes.append(nobs*nsim)
    smallestsampsize = min(samplesizes)
    smallestbatchsize = min(batchsizes)
    prevbatch = list()
    sampsincommontall = []
    sampsincommonwide = []
    for sampname in samplepaths:
        with np.load(sampname) as sampnpz:
            samp = np.array([sampnpz[samp_array] for samp_array in sampnpz.files])
            if len(prevbatch) == 0:
                prevbatch = samp
            else:
                sampsincommontall.append(np.array_equal(samp[:, 0:smallestsampsize], prevbatch[:, 0:smallestsampsize]))
                sampsincommonwide.append(np.array_equal(samp.flatten()[0:smallestbatchsize], prevbatch.flatten()[0:smallestbatchsize]))
                prevbatch = samp
    # if there is no overlap in the samples then will return True
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
        print("Samples generated across different nobs have matching initial values:", simulator_samps_dif_across_samples(samples, args.dir))

        