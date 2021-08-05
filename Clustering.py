#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# K-means
#https://realpython.com/k-means-clustering-python/


# # K-means - Coworkers

# In[1]:


import pandas as pd
import numpy as np


# In[5]:


pip install yellowbrick


# In[6]:


import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from yellowbrick.cluster import KElbowVisualizer


# In[7]:


# load dataset

file = "/Users/Eigenaar/Dropbox/&Master DSS/Thesis/Dataset"
MHdataset = pd.read_csv( file + "/MHdatasetComplete_Final.csv")


# In[8]:


MHdataset['Gender'] = pd.Categorical(MHdataset['Gender'])
MHdataset['Country'] = pd.Categorical(MHdataset['Country'])
MHdataset['Self_Employed'] = pd.Categorical(MHdataset['Self_Employed'])
MHdataset['Family_History'] = pd.Categorical(MHdataset['Family_History'])
MHdataset['Treatment'] = pd.Categorical(MHdataset['Treatment'])
MHdataset['Employees'] = pd.Categorical(MHdataset['Employees'])
MHdataset['Remote_Work'] = pd.Categorical(MHdataset['Remote_Work'])
MHdataset['Tech_Company'] = pd.Categorical(MHdataset['Tech_Company'])
MHdataset['MHbenefits'] = pd.Categorical(MHdataset['MHbenefits'])
MHdataset['Anonymity'] = pd.Categorical(MHdataset['Anonymity'])
MHdataset['MHoptions'] = pd.Categorical(MHdataset['MHoptions'])
MHdataset['Wellness_Program'] = pd.Categorical(MHdataset['Wellness_Program'])
MHdataset['Resources'] = pd.Categorical(MHdataset['Resources'])
MHdataset['Anonymity'] = pd.Categorical(MHdataset['Anonymity'])
MHdataset['Medical_Leave'] = pd.Categorical(MHdataset['Medical_Leave'])
MHdataset['MHnegConsequences'] = pd.Categorical(MHdataset['MHnegConsequences'])
MHdataset['PHnegConsequences'] = pd.Categorical(MHdataset['PHnegConsequences'])
MHdataset['Coworkers'] = pd.Categorical(MHdataset['Coworkers'])
MHdataset['Supervisors'] = pd.Categorical(MHdataset['Supervisors'])
MHdataset['MHinterview'] = pd.Categorical(MHdataset['MHinterview'])
MHdataset['PHinterview'] = pd.Categorical(MHdataset['PHinterview'])
MHdataset['MHvsPH'] = pd.Categorical(MHdataset['MHvsPH'])
MHdataset['CoworkernegCons'] = pd.Categorical(MHdataset['CoworkernegCons'])


# In[9]:


def Z_Scores(df):
    
    df2 = df
    cols = list(df.columns)
    
    for col in cols:
        if str(df.dtypes[col]) != 'category':
            
            col_zscore = col + '_zscore'
            df2[col_zscore] = (df[col] - df[col].mean())/df[col].std(ddof=0)
            
        if str(df.dtypes[col]) == 'category':
            df2[col] = df[col]
            
    return df2


# In[10]:


MHdataset = Z_Scores(MHdataset)


# In[11]:


MHdataset["Gender"] = MHdataset["Gender"].cat.codes
MHdataset["Country"] = MHdataset["Country"].cat.codes
MHdataset["Self_Employed"] = MHdataset["Self_Employed"].cat.codes
MHdataset["Family_History"] = MHdataset["Family_History"].cat.codes
MHdataset["Treatment"] = MHdataset["Treatment"].cat.codes
MHdataset["Employees"] = MHdataset["Employees"].cat.codes
MHdataset["Remote_Work"] = MHdataset["Remote_Work"].cat.codes
MHdataset["Tech_Company"] = MHdataset["Tech_Company"].cat.codes
MHdataset["MHbenefits"] = MHdataset["MHbenefits"].cat.codes
MHdataset["MHoptions"] = MHdataset["MHoptions"].cat.codes
MHdataset["Wellness_Program"] = MHdataset["Wellness_Program"].cat.codes
MHdataset["Resources"] = MHdataset["Resources"].cat.codes
MHdataset["Anonymity"] = MHdataset["Anonymity"].cat.codes
MHdataset["Medical_Leave"] = MHdataset["Medical_Leave"].cat.codes
MHdataset["MHnegConsequences"] = MHdataset["MHnegConsequences"].cat.codes
MHdataset["PHnegConsequences"] = MHdataset["PHnegConsequences"].cat.codes
MHdataset["MHinterview"] = MHdataset["MHinterview"].cat.codes
MHdataset["PHinterview"] = MHdataset["PHinterview"].cat.codes
MHdataset["MHvsPH"] = MHdataset["MHvsPH"].cat.codes
MHdataset["CoworkernegCons"] = MHdataset["CoworkernegCons"].cat.codes

MHdataset['Supervisors'] = MHdataset["Supervisors"].cat.codes


# In[12]:


MHdataset.info()


# In[13]:


ContextualFeatures = MHdataset.loc[:,['Employees', 'Remote_Work', 'Tech_Company',
                                    'MHbenefits', 'MHoptions', 'Wellness_Program',
                                     'Resources', 'Anonymity', 'Medical_Leave']]


# In[14]:


ContextualFeatures.info()


# In[16]:


model = KElbowVisualizer(KMeans(), k=7)
model.fit(ContextualFeatures)
model.show()


# In[16]:


kmeans = KMeans(init="random",
                n_clusters=3,
                n_init=10,
                max_iter=300,
                random_state=123)


# In[17]:


kmeans.fit(ContextualFeatures)


# In[22]:


print(kmeans.n_iter_)
print(kmeans.cluster_centers_)
print(kmeans.inertia_)


# In[19]:


MHdataset['Cluster'] = kmeans.labels_[:]


# In[33]:


MHdataset.info()


# In[34]:


MHdataset.to_csv(file+'/MHdatasetComplete_Final_Clusters.csv', index = False)

