#!/usr/bin/env python
# coding: utf-8

# # Demo of Monte Carlo Simulations for Apen, Sampen, and CGR-Renyi entropy measures

import sys, pathlib
import copy
import argparse
#print(sys.path)
sys.path.append(pathlib.Path(__file__).parent)
#print(sys.path)
import datetime
import json
import numpy as np
#from memory_profiler import profile
import pyusm
import discreteMSE
from mc_measures.gen_mc_transition import get_model, gen_sample
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

# this is the default set of m values for ApEn and SampEn
MVALS = [1, 2, 3, 4]

# define functions to compute expected entropy values for different generating distributions
def theta_iiduniform(a, sig2v=SIG2V):
    # a is the cardinality of the alphabet of the generating function
    a = np.array(a)
    #apen and sampen and renyi are expected to give identical results for iid uniformly distributed random numbers
    apen = np.log(a)
    sampen = np.log(a)
    #placeholder for formula
    renyi_disc = np.log(a)
    renyi_cont = {}
    for sig2 in sig2v:
        s2 = np.array(sig2, dtype=np.float64)
        rn2 = a * ((1/(-24*s2)) - np.log((2*np.sqrt(s2)*np.sqrt(np.pi))))
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
#@profile
def compute_cgr_renyi(data, sig2v=SIG2V, A=None, refseq=None, Plot=False):
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

#@profile
def compute_apen_and_sampen(data, mvals=MVALS, refseq=None):
    mests = []
    for m in mvals:
        est = {'m' : m, 'apen' : discreteMSE.apen(data, m)[0], 'sampen' : discreteMSE.sampen(data, m, refseq=refseq)[0]}
        mests.append(est)
    return mests

#@profile
def run(nsim, nobs, disttype, outroot, jobarray, jobid=None, seed=None, markovname=None):
    """
    generic routine for generating series of random samples, computing entropy
    statistics from each sample and saving all data to json files. For each number
    given nobs, nsim samples of that size will be generated from the disttype.

    Parameters
    ----------
    nsim : INT
        NSIM IS THE NUMBER OF INDEPENDENT SAMPLES TO GENERATE.
    nobs : LIST OF INT(S)
        NOBS IS THE NUMBER OF "OBSERVATIONS" TO RANDOMLY GENERATE WITHIN EACH
        SAMPLE GENERATED. (IE SIMULATED SAMPLE SIZE).
    disttype : STRING
        INDICATE THE TYPE OF PROBABILITY DISTRIBUTION TO USE TO GENERATE THE
        RANDOM SAMPLES. Options: 'markov', 'uniform'
    outroot : STRING
        PATH TO THE OUTPUT DIRECTORY
    jobarray : INT
        FOR SLURM ARRAY JOBS, EXPECTS $SLURM_ARRAY_TASK_ID
    jobid : INT or NONE
        FOR SLURM JOB ID. Default is None. If default, jobid will be set to datetime.datetime.now()
    seed : INT, optional.
        SEED IS THE SEED FOR THE RNG, CAN BE AN INTEGER (128-BIT RECOMMENDED)
        OR NONE. IF NONE THEN RNG SEED WILL BE GENERATED FROM THE SYSTEM ENTROPY.
        The default is None.
    markovname : STRING
        NAME OF THE MARKOV MODEL BESIDES THE ARRAY NUMBER. I.E. ORDER1ALPH4, ORDER2ALPH18, ETC.
    Returns
    -------
    None.
    """
    # get iso string of date when simulation begins to use for output directories
    simdate = datetime.date.today().isoformat()
    print(f"Beginning simulation run at {datetime.datetime.now().isoformat(timespec='minutes')}")
    # try:
    #     nsim = int(nsim)
    #     assert type(nsim) is int, "Please provide valid integer for nsim."
    # except ValueError:
    #     print("Please provide valid integer for nsim.")
    #     raise
    # except AssertionError as err:
    #     print("Failed assertion: {}".format(err))
    #     raise
        
    # try:
    #     nobs = [int(nobs[i]) for i in range(len(nobs)):  
    #     assert type(n) is int, "Please provide valid integer for nobs."
    # except ValueError:
    #     print("Please provide valid integer for nobs.")
    #     raise
    # except AssertionError as err:
    #     print("Failed assertion: {}".format(err))
    #     raise
    multifile = False
    
    #if jobid is None output file names will be appended with current datetime
    if jobid is None:
        # datetime to append to output file names
        jobid = datetime.datetime.now().strftime('%Y-%m-%dT%H%M%S')
        
    # set up simulation parameters based on disttype
    if disttype == 'uniform':
        func = genRandseq
        
        if multifile:
            # get list of file paths
            paramfiles =  sim_files.get_data_file_path(out_dir='input_data/iiduniform')
        else:
            # get data file path
            paramfiles = sim_files.get_data_file_path(out_dir='input_data/iiduniform', 
                                                      out_name=f'iiduniform_{jobarray}.json')
            paramfiles = [paramfiles]
            
    elif disttype == 'markov':
        func = gen_sample
        
        if multifile:
            # get list of file paths
            paramfiles = sim_files.get_data_file_path(out_dir='input_data/mc_matrices')
        else:
            # get data file path
            paramfiles = sim_files.get_data_file_path(out_dir='input_data/mc_matrices', 
                                                      out_name=f'{markovname}_{jobarray}.json')
            paramfiles = [paramfiles]
    else:
        raise Exception("""Unknown disttype. Please check spelling or documentation
                        for allowed distribution types.""")
                            
    # initiate simulator to handle nsim sequential simulations of the random sample func using the same RNG
    # simulator initiated with the PCG-64 bitgenerator
    sim = Simulator(func, nsim, seed)
    
    #set sample parameters
    for file in paramfiles:
        if disttype == 'uniform':
            # get saved sample parameters for sample generating function from file
            with open(file, 'r') as fhand:
                params = json.load(fhand)
            #set size of alphabet of discrete-valued random variable
            a = int(params['a'])
            distname = 'iiduni'
            #set of values making up the discrete-valued state space of the random variable X
            #states = [chr(ord('a')+i) for i in range(a)]
            states = np.array(range(a))
            mc_order = 0
        if disttype == 'markov':
            # get saved markov matrix from file
            MC_model = get_model(file)
            a = MC_model.m
            mc_order = MC_model.k
            distname = f'markovA{a}Order{mc_order}'
            states = MC_model.alph
    
        
        #create a dict of the true estimand values
        if disttype == 'uniform':
            thetas = theta_iiduniform(a)
        elif disttype == 'markov':
            thetas = theta_markov(MC_model)
        #create dict to keep a log of bitgenerator state sequences for each sample
        simulatorstates = {}
        #empty dict to hold simulated dataset filepaths
        simulated = {}
        #empty list to hold entropy estimates (theta hats)
        estimates = []
        for n in nobs:
            print(f"Beginning simulations for sample sizes {n}")
            #get a list containing nsim sample sequences
            if disttype == 'uniform':
                samples = sim(states, n)
            elif disttype == 'markov':
                #add 500 to n as the first 500 states will be dropped from the sample
                T = n + 500
                samples = sim(MC_model, states[0:mc_order], T, dropfirst=500)
            sampname = f'{distname}A{a}N{n}'
            samppath = sim_files.create_output_file_path(root_dir=outroot,
                                                         out_dir=f'simulation_output/simulated_{simdate}',
                                                         out_name=f"{sampname}_data_{jobid}.npz",
                                                         overide=False)
            #save string path to saved samples
            simulated[sampname] = str(samppath)
            #save samples as compressed numpy binary files 
            np.savez_compressed(samppath, *samples)
            
            #save rng states for sampname
            simulatorstates[sampname] = copy.deepcopy(sim.bgstateseq)
            
            values = []
            #mvals = [1, 2, 3, 4]
            #sig2v = SIG2V
            for i in range(len(samples)):
                vals = {'sample' : i}
                #print(samples[i])
                renyis = compute_cgr_renyi(samples[i], SIG2V, A=states, refseq=f'{sampname}i{i}', Plot=False)
                #moved to the function compute_apen_and_sampen()
                # mests = []
                # for m in mvals:
                #     est = {'m' : m, 'apen' : discreteMSE.apen(samples[i], m)[0], 'sampen' : discreteMSE.sampen(samples[i], m, refseq=f'{sampname}i{i}')[0]}
                #     mests.append(est)
                mests = compute_apen_and_sampen(samples[i], mvals=MVALS, refseq=f'{sampname}i{i}')
                vals.update({'renyi_hats' : [renyis,], 'theta_hats' : mests})
                values.append(vals)
            estimates.append({'sampname': sampname, 'nobs' : n, 'values' : values})
    
        #assert estimates is list of dicts
    
        #save simulation metadata to a json file
        
        #make list to contain the extra args to feed to sim_files.sim_data_dump()
        #in the order [alphabet, data generating distribution, Markov order]
        addinfo = [a, distname, mc_order]
        outpath = sim_files.create_output_file_path(root_dir=outroot, 
                                                    out_dir=f'simulation_output/simulated_{simdate}', 
                                                    out_name=f'{distname}A{a}Nsim{nsim}_{jobid}.json', 
                                                    overide=False)
        sim_files.sim_data_dump(simulated, simulatorstates, outpath, *addinfo)
        # save entropy estimates as a json file
        estsoutpath = sim_files.create_output_file_path(root_dir=outroot, 
                                                        out_dir=f'simulation_output/estimates_{simdate}', 
                                                        out_name=f'{distname}A{a}Nsim{nsim}_{jobid}_estimates.json', 
                                                        overide=False)
        sim_files.sim_est_dump(nsim, thetas, estimates, estsoutpath, *addinfo)
    
    print(f"Simulations completed {datetime.datetime.now().isoformat(timespec='minutes')}")
    return

if __name__ == "__main__":
    import os
    threads = os.getenv("OMP_NUM_THREADS")
    if threads:
        print("Number of CPUs available:", os.environ["OMP_NUM_THREADS"])
    parser = argparse.ArgumentParser(description='Run entropy simulations.')
    parser.add_argument('nsim', type=int,
                        help='number of samples to generate')
    parser.add_argument('--nobs', nargs='+', type=int, required=True,
                        help='list of sample sizes to generate')
    parser.add_argument('-d', '--disttype', type=str, choices=['uniform', 'markov'],
                        required=True, help='distribution type to sample from')
    parser.add_argument('-j', '--jobarray',  type=int, required=True, help='slurm array task id')
    parser.add_argument('-o', '--outroot',
                        required=True, help='Path for output files to go')
    parser.add_argument('--seed', type=str, help='seed for RNG')
    parser.add_argument('--markovname', type=str, help='name of Markov model')
    args = parser.parse_args()
    # print(args.nsim, type(args.nsim))
    # print(args.nobs, type(args.nobs))
    # print(args.disttype, type(args.disttype))
    run(args.nsim, args.nobs, args.disttype, args.outroot, args.jobarray, seed=args.seed, markovname=args.markovname)