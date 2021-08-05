#!/usr/bin/env python
# coding: utf-8

# In[82]:


#https://www.datacamp.com/community/tutorials/naive-bayes-scikit-learn

import pandas as pd 
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  
import matplotlib.pyplot as plt


# # Majority - coworkers

# In[91]:


from scipy.stats import mode


# In[83]:


file = "/Users/Eigenaar/Dropbox/&Master DSS/Thesis/Dataset"
MHdataset = pd.read_csv( file + "/MHdatasetComplete_Final.csv")


# In[84]:


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


# In[85]:


MHdataset = Z_Scores(MHdataset)


# In[104]:


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


# In[105]:


X = MHdataset.loc[:, MHdataset.columns != 'Coworkers']
y = MHdataset['Coworkers']


# In[106]:


y = pd.Series(y).array


# In[89]:


def majority_class(y):
    return mode(y)[0]


# In[109]:


yhat = [majority_class(y) for _ in range(len(y))]


# In[111]:


print(confusion_matrix(y, yhat))
print(classification_report(y, yhat))


# ## Majority - Supervisors

# In[86]:


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

MHdataset['Coworkers'] = MHdataset["Coworkers"].cat.codes


# In[87]:


X = MHdataset.loc[:, MHdataset.columns != 'Supervisors']
y = MHdataset['Supervisors']


# In[88]:


y = pd.Series(y).array


# In[92]:


yhat = [majority_class(y) for _ in range(len(y))]


# In[93]:


print(confusion_matrix(y, yhat))
print(classification_report(y, yhat))


# # Coworkers full

# In[1]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB


# In[3]:


file = "/Users/Eigenaar/Dropbox/&Master DSS/Thesis/Dataset"
MHdataset = pd.read_csv( file + "/MHdatasetComplete_Final.csv")


# In[4]:


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


# In[5]:


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


# In[6]:


MHdataset = Z_Scores(MHdataset)


# In[7]:


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


# In[8]:


X = MHdataset.loc[:, MHdataset.columns != 'Coworkers']
y = MHdataset['Coworkers']


# In[9]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=123) 


# In[10]:


X = X.drop(['Age','Power_Distance','Individualism','Masculinity', 'Uncertainty_Avoidance', 'Long_Term_Orientation', 'Indulgence'], axis=1)


# In[11]:


y = pd.Series(y).array


# In[12]:


def prepare_target_features(y_train, y_test):
    le = LabelEncoder()
    le.fit(y_train)
    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)
    return y_train_enc, y_test_enc


# In[13]:


y_train_enc, y_test_enc = prepare_target_features(y_train, y_test)


# In[14]:


from imblearn.over_sampling import SMOTE


# In[15]:


print("Before OverSampling, counts of label '2': {}".format(sum(y_train_enc==2)))
print("Before OverSampling, counts of label '1': {}".format(sum(y_train_enc==1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train_enc==0)))

sm = SMOTE(random_state=123)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train_enc.ravel())

print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))

print("After OverSampling, counts of label '2': {}".format(sum(y_train_res==2)))
print("After OverSampling, counts of label '1': {}".format(sum(y_train_res==1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res==0)))


# In[16]:


#Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB

#Create a Gaussian Classifier
model = GaussianNB()

#Train the model using the training sets
model.fit(X_train_res, y_train_res.ravel())

#Predict the response for test dataset
predictions = model.predict(X_test)


# In[17]:


print(confusion_matrix(y_test_enc, predictions))
print(classification_report(y_test_enc, predictions))


# In[16]:


#old
print(confusion_matrix(y_test_enc, predictions))
print(classification_report(y_test_enc, predictions))


# # Coworkers FS

# In[18]:


file = "/Users/Eigenaar/Dropbox/&Master DSS/Thesis/Dataset"
MHdataset = pd.read_csv( file + "/MHdatasetComplete_Final.csv")


# In[19]:


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


# In[20]:


MHdataset = Z_Scores(MHdataset)


# In[21]:


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


# In[22]:


FeatureSelection = MHdataset.loc[:,['Self_Employed', 'MHnegConsequences', 'PHnegConsequences','Supervisors', 'MHinterview', 'Indulgence']]


# In[23]:


X = FeatureSelection
y = MHdataset['Coworkers']


# In[24]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=123)


# In[25]:


y = pd.Series(y).array


# In[26]:


y_train_enc, y_test_enc = prepare_target_features(y_train, y_test)


# In[27]:


sm = SMOTE(random_state=123)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train_enc.ravel())


# In[29]:


#Create a Gaussian Classifier
model = GaussianNB()

#Train the model using the training sets
model.fit(X_train_res, y_train_res.ravel())

#Predict the response for test dataset
predictions = model.predict(X_test)


# In[30]:


print(confusion_matrix(y_test_enc, predictions))
print(classification_report(y_test_enc, predictions))


# In[27]:


#old
print(confusion_matrix(y_test_enc, predictions))
print(classification_report(y_test_enc, predictions))


# # Coworker Cluster

# In[31]:


file = "/Users/Eigenaar/Dropbox/&Master DSS/Thesis/Dataset"
MHdataset = pd.read_csv( file + "/MHdatasetComplete_Final_Clusters.csv")


# In[32]:


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
MHdataset['Cluster'] = pd.Categorical(MHdataset['Cluster'])


# In[33]:


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
MHdataset["Cluster"] = MHdataset["Cluster"].cat.codes

MHdataset['Supervisors'] = MHdataset["Supervisors"].cat.codes


# In[34]:


ClusterSelection = MHdataset.drop(['Employees', 'Remote_Work', 'Tech_Company',
                                    'MHbenefits', 'MHoptions', 'Wellness_Program',
                                     'Resources', 'Anonymity', 'Medical_Leave'], axis = 1)


# In[35]:


X = ClusterSelection.loc[:, ClusterSelection.columns != 'Coworkers']
y = MHdataset['Coworkers']


# In[36]:


X = X.drop(['Age','Power_Distance','Individualism','Masculinity', 'Uncertainty_Avoidance', 'Long_Term_Orientation', 'Indulgence'], axis=1)


# In[37]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=123)


# In[38]:


y = pd.Series(y).array


# In[39]:


y_train_enc, y_test_enc = prepare_target_features(y_train, y_test)


# In[40]:


sm = SMOTE(random_state=123)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train_enc.ravel())


# In[41]:


#Create a Gaussian Classifier
model = GaussianNB()

#Train the model using the training sets
model.fit(X_train_res, y_train_res.ravel())

#Predict the response for test dataset
predictions = model.predict(X_test)


# In[42]:


print(confusion_matrix(y_test_enc, predictions))
print(classification_report(y_test_enc, predictions))


# In[38]:


# old
print(confusion_matrix(y_test_enc, predictions))
print(classification_report(y_test_enc, predictions))


# # Supervisors Full

# In[43]:


file = "/Users/Eigenaar/Dropbox/&Master DSS/Thesis/Dataset"
MHdataset = pd.read_csv( file + "/MHdatasetComplete_Final.csv")


# In[44]:


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


# In[45]:


MHdataset = Z_Scores(MHdataset)


# In[46]:


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

MHdataset['Coworkers'] = MHdataset["Coworkers"].cat.codes


# In[47]:


X = MHdataset.loc[:, MHdataset.columns != 'Supervisors']
y = MHdataset['Supervisors']


# In[48]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=123) 


# In[49]:


X = X.drop(['Age','Power_Distance','Individualism','Masculinity', 'Uncertainty_Avoidance', 'Long_Term_Orientation', 'Indulgence'], axis=1)


# In[50]:


y = pd.Series(y).array


# In[51]:


y_train_enc, y_test_enc = prepare_target_features(y_train, y_test)


# In[53]:


sm = SMOTE(random_state=123)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train_enc.ravel())


# In[54]:


#Create a Gaussian Classifier
model = GaussianNB()

#Train the model using the training sets
model.fit(X_train_res, y_train_res.ravel())

#Predict the response for test dataset
predictions = model.predict(X_test)


# In[55]:


print(confusion_matrix(y_test_enc, predictions))
print(classification_report(y_test_enc, predictions))


# In[49]:


#old
print(confusion_matrix(y_test_enc, predictions))
print(classification_report(y_test_enc, predictions))


# # Supervisors FS

# In[56]:


file = "/Users/Eigenaar/Dropbox/&Master DSS/Thesis/Dataset"
MHdataset = pd.read_csv( file + "/MHdatasetComplete_Final.csv")


# In[57]:


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


# In[58]:


MHdataset = Z_Scores(MHdataset)


# In[59]:


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

MHdataset['Coworkers'] = MHdataset["Coworkers"].cat.codes


# In[60]:


MHdataset.info()


# In[61]:


FeatureSelection = MHdataset.loc[:,['Self_Employed', 'MHnegConsequences', 'PHnegConsequences','Coworkers', 'MHinterview', 'Indulgence']]


# In[62]:


X = FeatureSelection
y = MHdataset['Supervisors']


# In[63]:


X.info()


# In[64]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=123)


# In[65]:


y = pd.Series(y).array


# In[66]:


y_train_enc, y_test_enc = prepare_target_features(y_train, y_test)


# In[67]:


sm = SMOTE(random_state=123)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train_enc.ravel())


# In[68]:


#Create a Gaussian Classifier
model = GaussianNB()

#Train the model using the training sets
model.fit(X_train_res, y_train_res.ravel())

#Predict the response for test dataset
predictions = model.predict(X_test)


# In[69]:


print(confusion_matrix(y_test_enc, predictions))
print(classification_report(y_test_enc, predictions))


# In[88]:


# old
print(confusion_matrix(y_test_enc, predictions))
print(classification_report(y_test_enc, predictions))


# # Supervisors Clusters

# In[70]:


file = "/Users/Eigenaar/Dropbox/&Master DSS/Thesis/Dataset"
MHdataset = pd.read_csv( file + "/MHdatasetComplete_Final_Clusters.csv")


# In[71]:


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
MHdataset['Cluster'] = pd.Categorical(MHdataset['Cluster'])


# In[72]:


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
MHdataset["Cluster"] = MHdataset["Cluster"].cat.codes

MHdataset['Coworkers'] = MHdataset["Coworkers"].cat.codes


# In[73]:


ClusterSelection = MHdataset.drop(['Employees', 'Remote_Work', 'Tech_Company',
                                    'MHbenefits', 'MHoptions', 'Wellness_Program',
                                     'Resources', 'Anonymity', 'Medical_Leave'], axis = 1)


# In[74]:


X = ClusterSelection.loc[:, ClusterSelection.columns != 'Supervisors']
y = MHdataset['Supervisors']


# In[75]:


X = X.drop(['Age','Power_Distance','Individualism','Masculinity', 'Uncertainty_Avoidance', 'Long_Term_Orientation', 'Indulgence'], axis=1)


# In[76]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=123)


# In[77]:


y = pd.Series(y).array


# In[78]:


y_train_enc, y_test_enc = prepare_target_features(y_train, y_test)


# In[79]:


sm = SMOTE(random_state=123)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train_enc.ravel())


# In[80]:


#Create a Gaussian Classifier
model = GaussianNB()

#Train the model using the training sets
model.fit(X_train_res, y_train_res.ravel())

#Predict the response for test dataset
predictions = model.predict(X_test)


# In[81]:


print(confusion_matrix(y_test_enc, predictions))
print(classification_report(y_test_enc, predictions))


# In[99]:


# old
print(confusion_matrix(y_test_enc, predictions))
print(classification_report(y_test_enc, predictions))

