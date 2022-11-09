# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 21:10:49 2022

@author: Wuestney
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pathlib


datafilepath =
fhand = open(estsoutpath)
estimates_loaded= json.load(fhand)
fhand.close()
estimates_loaded


# ## Read estimates dataset into a Pandas Dataframe

# In[ ]:


def monteCarloSE(biasarray):
    nsim = len(biasarray)
    squaredsum=np.square(biasarray).sum()
    SE = np.sqrt(squaredsum*(1/(nsim*(nsim-1))))


# In[ ]:


data = estimates
data = estimates_loaded


# In[26]:


theta_hats = pd.json_normalize(data=estimates, record_path=['values', 'theta_hats'],
                               meta=["sampname", "nobs", ["values", "sample"]])
theta_hats.set_index(['sampname', 'nobs', 'values.sample'])
theta_hats


# In[25]:


theta_hatsload = pd.json_normalize(data=estimates_loaded, record_path=['Estimates', 'values', 'theta_hats'],
                               meta=[['Estimates', "sampname"], ['Estimates', "nobs"], ['Estimates', "values", "sample"]])
theta_hatsload


# In[28]:


renyi_hats = pd.json_normalize(data=estimates, record_path=['values', 'renyi_hats'],
                               meta=["sampname", "nobs", ['values', 'sample']])
renyi_hats


# In[27]:


renyi_hatsload = pd.json_normalize(data=estimates_loaded, record_path=['Estimates', 'values', 'renyi_hats'],
                               meta=[['Estimates', "sampname"], ['Estimates', "nobs"], ['Estimates', "values", "sample"]])
renyi_hatsload


# In[15]:


renyi_hats_long = pd.melt(renyi_hats, id_vars=['sampname', 'nobs', 'values.sample'],
                          var_name='sig2', value_name='renyi2')
renyi_hats_long.set_index(['sampname', 'nobs', 'values.sample'])


# In[ ]:


dfestswide = pd.merge(theta_hats, renyi_hats, how='outer', on=['sampname', 'nobs', 'values.sample'])
dfestswide


# In[ ]:


dfests = pd.wide_to_long(dfestswide, 'kernelvar', i=['nobs', 'values.sample'], j='sig2', sep='_')
dfests
#print(dfests.columns)


# In[ ]:


dfindex = dfests.index
dfindex


# In[ ]:


print(list(zip(('apen', 'sampen', 'renyi'), theta_iiduniform(a))))
print(thetas)


# In[ ]:


#dftheta = pd.json_normalize(data=thetas)
dftheta = pd.DataFrame(thetas, index=dfindex)
dftheta.rename(columns={'apen': 'theta.apen', 'sampen': 'theta.sampen', 'renyi': 'theta.renyi'}, inplace=True)
dftheta


# In[ ]:


df = pd.concat([dfests, dftheta], axis=1)
df[df['nobs']== 100]


# In[ ]:


dfests = pd.json_normalize(data=estimates, record_path=['values', 'theta_hats'],
                       meta=["sampname", "nobs", ["values", "m"]])
meta=["sampname", "nobs", ["values", "sample"]

