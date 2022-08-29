# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 15:55:40 2022

@author: Wuestney
"""
import datetime
import json
from pathlib import Path
import sys



def create_output_file_path(root_dir=None, out_dir='sim_data', out_name=None, overide=False):
    """
    Create path object for the output filepath for the output file.

    Parameters

    Optional Keyword arguments:
    root_dir -- string literal or pathlike object refering to the target root directory
    out_dir -- string literal representing target directory to be appended to root if root is not target directory
    out_name -- string literal representing the filename for the output file

    Return : path object to save simulation data in

    """
    if not root_dir:
        root_dir = Path.cwd()
    dirpath = Path(root_dir, out_dir)
    if not dirpath.is_dir():
        dirpath.mkdir(parents=True)
    if not out_name:
        out_name = f"simdata_{datetime.datetime.now().isoformat(timespec='seconds')}.txt"
    out_path = dirpath / out_name
    if out_path.exists():
        if overide is False:
            raise Exception("Output file already exists. Please enter unique filepath information or use overide==True.")
        else:
            return out_path
    else:
        return out_path

def get_data_file_path(root_dir=None, out_dir='sim_data', out_name=None):
    """ Create path object for the output filepath for the output file.

    Optional Keyword arguments:
    root_dir -- string literal or pathlike object refering to the target root directory
    out_dir -- string literal representing target directory to be appended to root if root is not target directory
    out_name -- string literal representing the filename for the output file

    Return : path object or list of path objects to retrieve or save simulation data in

    """
    if not root_dir:
        root_dir = Path.cwd()
    dirpath = Path(root_dir, out_dir)
    if not dirpath.is_dir():
        dirpath.mkdir(parents=True)
    if not out_name:
        out_paths = list(dirpath.glob('*.json'))
        return out_paths
    else:
        out_path = dirpath / out_name
        if not out_path.exists():
            raise Exception("File does not exist. Please enter existing filepath.")
        else:
            return out_path

def sim_data_dump(simulated, states, outpath, *args, **kwargs):
    """Writes random samples to a json file. Opens the file, writes, and closes it.

    simulated : DICT
        DICT CONTAINING THE SIMULATION NAME AS A KEY AND A LIST OF THE SIMULATED SAMPLES AS THE VALUE.
        SAMPLE SEQUENCES SHOULD ALSO BE IN LIST OR STRING FORM
        OR ELSE THEY WILL CAUSE AN ERROR IN THE JSON SERIALIZER.
    states : DICT
        DICTIONARY CONTAINING THE BITGENERATOR STATES CORRESPONDING TO EACH SAMPLE IN SIMULATED.
        EACH KEY IN states IS AN INTEGER CORRESPONDING TO THE INDEX OF THE SAMPLE
        IN SIMULATED.
    outpath : PATH-LIKE OBJECT OR STRING LITERAL OF A FILE PATH
        TAKES A PATH OBJECT POINTING TO A FILE LOCATION FOR THE OUTPUT TO BE WRITTEN TO.
    *args : OPTIONAL ADDITIONAL ARGS ALLOWED IN THIS ORDER:
        alph : INTEGER REPRESENTING SIZE OF ALPHABET OR STRING OR LIST CONTAINING THE STATES IN THE ALPHABET
        dist_name : NAME OF THE DISTRIBUTION USED TO GENERATE THE SAMPLES.
        mc_order : INTEGER
    **kwargs : OPTIONAL
        ANY ADDITIONAL ARGUMENTS TO BE FED TO JSON.DUMP(). DO NOT USE ANY NAMED
        ARGS FOR SIM DATA AS THESE WILL BE PLACED IN **KWARGS AND FED TO THE JSON
        FUNCTION, NOT SAVED AS DATA IN THE JSON FILE.
    """
    addons = {'Alphabet' : None, 'Name of Generating Distribution' : None, 'Markov Order' : None}
    if args[0]:
        addons['Alphabet'] = args[0]
    if args[1]:
        addons['Name of Generating Distribution'] = args[1]
    if args[2]:
        addons['Markov Order'] = args[2]
    with open(outpath, 'w+') as fouthand:
        data = {'Simulated Samples' : simulated,
                'BitGenerator states' : states}
        data.update(addons)
        data.update({'Date_created' : datetime.datetime.now().isoformat(timespec='seconds')})
        json.dump(data, fouthand, **kwargs)

def sim_est_dump(nsim, thetas, estimates, outpath, *args, **kwargs):
    """
    Function to save Monte Carlo simulation estimand estimates to a json file.

    Parameters
    ----------
    nsim : INT
        NUMBER OF SIMULATED SAMPLES PER DISTRIBUTION PARAMETER SETUP
    thetas : DICT OR OTHER JSON SERIALIZABLE OBJECT
        DICT CONTAINING THE ANALYTICALLY DERIVED TRUE ESTIMAND VALUES.
    estimates : JSON SERIALIZABLE OBJECTS (LIST, DICT)
        NESTED LISTS AND DICTS CONTAINING THE ESTIMATES OF THE ESTIMANDS
        OBTAINED FROM EACH SIMULATED SAMPLE.
    outpath : PATH-LIKE OBJECT OR STRING LITERAL OF A FILE PATH
        takes a path object pointing to a file location for the output to be written to.
    *args : ADDITIONAL ARGS ALLOWED IN THIS ORDER:
        alph : INTEGER REPRESENTING SIZE OF ALPHABET OR STRING OR LIST CONTAINING THE STATES IN THE ALPHABET
        dist_name : NAME OF THE DISTRIBUTION USED TO GENERATE THE SAMPLES.
        mc_order : INTEGER
    **kwargs : DICT, OPTIONAL
        any additional arguments to be fed to json.dump()

    Returns
    -------
    None.

    """
    addons = {'Alphabet' : None, 'Name of Generating Distribution' : None, 'Markov Order' : None}
    if args[0]:
        addons['Alphabet'] = args[0]
    if args[1]:
        addons['Name of Generating Distribution'] = args[1]
    if args[2]:
        addons['Markov Order'] = args[2]
    with open(outpath, 'w+') as fouthand:
        data = {'nsim' : nsim, 'Thetas' : thetas, 'Estimates' : estimates}
        data.update(addons)
        data.update({'Date_created' : datetime.datetime.now().isoformat(timespec='seconds')})
        json.dump(data, fouthand, **kwargs)