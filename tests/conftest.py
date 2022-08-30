# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 16:04:29 2022

@author: Wuestney
"""

import pytest

default_nobs_list = [20, 50, 100]
@pytest.fixture(scope='session')
def alpha4_statespace():
    statespace = ['a', 'b', 'c', 'd']
    return statespace

@pytest.fixture(scope='session', params=default_nobs_list)
def default_nobs(request):
    return request.param

@pytest.fixture(scope='session')
def seed():
    # seed generated on 8/3/22 from np.random.SeedSequence().entropy
    seed = 264402757879708171514988060933040011692
    return seed

