
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import sklearn.utils
import xgboost as xgb
from tldextract import tldextract
from urllib.parse import urlparse, parse_qs

from script import get_tokens
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression


# In[2]:

df = pd.read_csv('data/data.csv')


# In[6]:

df2 = pd.read_csv('../phish-ml/online-valid.csv')


# In[ ]:

df2 = df2.drop(['phish_id', 'phish_detail_url', 'submission_time', 'verified', 'verification_time', 'online', 'target'], axis=1)


# In[18]:

new_df = pd.concat((df, df2), ignore_index=True)


# In[20]:

new_df['target'] = 0


# In[21]:

new_df.loc[new_df['label'] == 'bad', 'target'] = 1


# In[25]:

new_df = new_df.drop(['label'], axis=1)


# In[27]:

new_df.to_csv('data/data_phish.csv', index=False)


# In[2]:

df = pd.read_csv('data/data_phish.csv')


# In[7]:

df = df.drop(['Unnamed: 0'], axis=1)


# In[9]:

df.to_csv('data/data_phish.csv', index=False)


# In[5]:

get_tokens('freeserials.spb.ru/key/68703.htm')

