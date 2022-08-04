# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 16:04:29 2022

@author: Wuestney
"""

import pytest

@pytest.fixture
def alpha_statespace():
    statespace = ['a', 'b', 'c', 'd']
    return statespace

@pytest.fixture
def default_nobs():
    return 20

@pytest.fixture
def seed():
    # seed generated on 8/3/22 from np.random.SeedSequence().entropy
    seed = 264402757879708171514988060933040011692
    return seed

