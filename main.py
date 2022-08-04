#!/usr/bin/env python
# coding: utf-8

# # Demo of Monte Carlo Simulations for Apen, Sampen, and CGR-Renyi entropy measures

import sys, pathlib
import argparse
#print(sys.path)
sys.path.append(pathlib.Path(__file__).parent)
print(sys.path)
import datetime

import json
import numpy as np
from numpy.random import PCG64
import pandas as pd
import pyusm
import discreteMSE
#from discreteMSE.discreteMSE.entropy import apen, sampen
from mc_measures.gen_mc_transition import GenMarkovTransitionProb as MCmatrix
from mc_measures.gen_mc_transition import gen_model, get_model, gen_sample
from mc_measures import mc_entropy
from sim_utils import sim_files
from sim_utils.random_samples import genRandseq, Simulator


# set module global data

# this is same default sig2 array pyusm.usm_entropy.
SIG2V = ('1.000000e-10', '1.778279e-10', '3.162278e-10', '5.623413e-10',
                 '1.000000e-09', '1.778279e-09', '3.162278e-09', '5.623413e-09',
                 '1.000000e-08', '1.778279e-08', '3.162278e-08', '5.623413e-08',
                 '1.000000e-07', '1.778279e-07', '3.162278e-07', '5.623413e-07',
                 '1.000000e-06', '1.778279e-06', '3.162278e-06', '5.623413e-06',
                ' 1.000000e-05', '1.778279e-05', '3.162278e-05', '5.623413e-05',
                '1.000000e-04', '1.778279e-04', '3.162278e-04', '5.623413e-04',
                '1.000000e-03', '1.778279e-03', '3.162278e-03', '5.623413e-03',
                '1.000000e-02', '1.778279e-02', '3.162278e-02', '5.623413e-02',
                '1.000000e-01', '1.778279e-01', '3.162278e-01', '5.623413e-01',
                '1', '1.778279e+00', '3.162278e+00', '5.623413e+00', '10', '1.778279e+01',
                '3.162278e+01', '5.623413e+01', '100')

# define functions to compute expected entropy values for different generating distributions
def theta_iiduniform(a, sig2v=SIG2V):
    # a is the cardinality of the alphabet of the generating function
    #apen and sampen and renyi are expected to give identical results for iid uniformly distributed random numbers
    apen = np.log(a)
    sampen = np.log(a)
    #placeholder for formula
    renyi_disc = np.log(a)
    renyi_cont = {}
    for sig2 in sig2v:
        rn2 = a * ((1/(-12*sig2)) - np.log((2*np.sqrt(sig2)*np.sqrt(np.pi))))
        renyi_cont[sig2] = rn2
    return dict(list(zip(('apen', 'sampen', 'renyi_disc', 'renyi_cont'), (apen, sampen, renyi_disc, renyi_cont))))

def theta_markov(MC_model):
    # function expects instance of a GMTP class as input
    #apen of a Markov chain is equivalent to the entropy rate of the Markov chain
    apen = mc_entropy.markov_apen(MC_model)
    sampen = mc_entropy.markov_sampen(MC_model)
    # theoretical renyi of mc chain not defined at this time
    # this is place holder
    renyi = None
    return dict(list(zip(('apen', 'sampen', 'renyi'), (apen, sampen, renyi))))

# combine making usm instance and computing renyi entropy into one function so
# that it can be passed to a Simulator class
def cgr_renyi(data, sig2v=SIG2V, A=None, refseq=None, Plot=False):
    """
    Combines making the usm coordinates from a data sequence and calculating the renyi entropy values on the CGR object.

    Parameters
    ----------
    data : STRING OR ARRAY-TYPE OF INTEGERS
        DISCRETE-VALUED DATA SEQUENCE.
    sig2v : VECTOR WITH VARIANCES, SIG2, TO USE WITH PARZEN METHOD
    A : LIST CONTAINING ALL POSSIBLE SYMBOLS OF THE ALPHABET OF THE SEQUENCE. The default is None.
        If default, will take alphabet as set of unique characters in seq.
    refseq : STRING, optional
        NAME OF SEQUENCE. The default is None.
    Plot : BOOL, optional
        PLOT OPTION PASSED TO renyi2usm().
        OPTION TO PLOT ENTROPY VALUES AS A FUNCTION OF THE LOG OF THE KERNEL
        VARIANCES, SIG2V. ENTROPY VALUES ON THE Y AXIS AND LN(SIG2) VALUES
        ON THE X AXIS.

    Returns
    -------
    Dictionary containing renyi quadratic entropy of the USM for each sig2 value.

    """
    cgr = pyusm.USM.make_usm(data, A=A, seed='centroid')
    cgr_coords = np.asarray(cgr.fw)
    renyi = pyusm.usm_entropy.renyi2usm(cgr_coords, sig2v, refseq=refseq, Plot=Plot, deep_copy=False)
    return renyi


# ## Create generic routine for generating series of random samples

#set the simulator parameters

# nsim is the number of independent samples to generate
nsim = 20
# seed for the RNG
seed = None
#set disttype. Options: 'markov', 'uniform', 'regular'
disttype = 'uniform'
if disttype == 'uniform':
    func = genRandseq
elif disttype == 'markov':
    func = gen_sample


#initiate simulator to handle nsim sequential simulations of the random sample func using the same RNG
sim = Simulator(func, nsim, seed)


#set sample parameters

#set size of alphabet of discrete-valued random variable
a = 4
#set of sample sizes (sequence lengths) to generate during each iteration
nobs = [50, 100]
#distribution function (optional), use to define a probability distribution for generating the random samples
distname = None
#markov order, to be used later
mc_order = None
#markov transition matrix object, placeholder
MC_model =  None
#set of values making up the discrete-valued state space of the random variable X
states = [chr(ord('a')+i) for i in range(a)]




#create a dict of the true estimand values
if disttype == 'uniform':
    distname = 'iiduni'
    thetas = dict(list(zip(('apen', 'sampen', 'renyi'), theta_iiduniform(a))))
elif disttype == 'markov':
    distname == ''
    thetas = dict(list(zip(('apen', 'sampen', 'renyi'), theta_markov(MC_model))))
#create dict to keep a log of bitgenerator state sequences for each sample
simulatorstates = {}
#empty dict to hold simulated datasets
simulated = {}
#empty list to hold entropy estimates (theta hats)
estimates = []
for n in nobs:
    if disttype == 'uniform':
        samples = sim(states, n)
    elif disttype == 'markov':
        #add alphabet size, a, to n as the first a states will be dropped from the sample
        T = n + a
        samples = sim(MC_model, states, n)
    sampname = f'{distname}A{a}N{n}'
    simulatorstates[sampname] = sim.bgstateseq
    simulated[sampname] = samples
    values = []
    mvals = [1, 2, 3, 4]
    sig2v = SIG2V
    for i in range(len(samples)):
        vals = {'sample' : i}
        #print(samples[i])
        renyis = cgr_renyi(samples[i], sig2v, A=states, refseq=f'{sampname}i{i}', Plot=False)
        mests = []
        for m in mvals:
            est = {'m' : m, 'apen' : discreteMSE.apen(samples[i], m)[0], 'sampen' : discreteMSE.sampen(samples[i], m, refseq=f'{sampname}i{i}')[0]}
            mests.append(est)
        vals.update({'renyi_hats' : [renyis,], 'theta_hats' : mests})
        values.append(vals)
    estimates.append({'sampname': sampname, 'nobs' : n, 'values' : values})



#assert estimates is list of dicts


# In[19]:


#save simulated datasets as a json file
#make list to contain the extra args to feed to sim_files.sim_data_dump()
#in the order [alphabet, data generating distribution, Markov order]
addinfo = [a, distname, mc_order]
outpath = sim_files.create_output_file_path(out_name=f'{distname}A{a}Nsim{nsim}_{datetime.date.today().isoformat()}.json', overide=True)
sim_files.sim_data_dump(simulated, simulatorstates, outpath, *addinfo)
estsoutpath = sim_files.create_output_file_path(out_name=f'{distname}A{a}Nsim{nsim}_{datetime.date.today().isoformat()}_estimates.json', overide=True)
sim_files.sim_est_dump(thetas, estimates, estsoutpath, *addinfo)


# In[22]:


