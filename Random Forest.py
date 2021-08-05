#!/usr/bin/env python
# coding: utf-8

# # Random Forest - Coworker - Full

# In[ ]:


# https://www.kaggle.com/sociopath00/random-forest-using-gridsearchcv


# In[1]:


import pandas as pd 
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC  
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV


# In[2]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


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


# In[ ]:


# random forest does not require feature scaling


# In[5]:


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


# In[6]:


X = MHdataset.loc[:, MHdataset.columns != 'Coworkers']
y = MHdataset['Coworkers']


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=123) 


# In[8]:


y = pd.Series(y).array


# In[9]:


def prepare_target_features(y_train, y_test):
    le = LabelEncoder()
    le.fit(y_train)
    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)
    return y_train_enc, y_test_enc


# In[10]:


y_train_enc, y_test_enc = prepare_target_features(y_train, y_test)


# In[11]:


rfc=RandomForestClassifier(random_state=123)


# In[13]:


from imblearn.over_sampling import SMOTE


# In[14]:


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


# In[15]:


param_grid = { 
    'n_estimators' : [100, 200, 400],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}


# In[16]:


grid = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
grid.fit(X_train_res, y_train_res.ravel())


# In[17]:


print(grid.best_params_)


# In[18]:


# fill in best params
model =RandomForestClassifier(random_state=123, n_estimators = 400, max_features= 'log2',  max_depth=8, criterion='entropy')


# In[19]:


model.fit(X_train_res, y_train_res.ravel())
# 
predictions = model.predict(X_test)

print(confusion_matrix(y_test_enc, predictions))
print(classification_report(y_test_enc, predictions))


# In[28]:


# old
predictions = model.predict(X_test)

print(confusion_matrix(y_test_enc, predictions))
print(classification_report(y_test_enc, predictions))
print(accuracy_score(y_test_enc, predictions))


# # Random Forest - Coworker - FS

# In[20]:


file = "/Users/Eigenaar/Dropbox/&Master DSS/Thesis/Dataset"
MHdataset = pd.read_csv( file + "/MHdatasetComplete_Final.csv")


# In[21]:


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


# In[22]:


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


# In[35]:


MHdataset.info()


# In[23]:


FeatureSelection = MHdataset.loc[:,['Self_Employed', 'MHnegConsequences', 'PHnegConsequences','Supervisors', 'MHinterview', 'Indulgence']]



# In[24]:


X = FeatureSelection
y = MHdataset['Coworkers']


# In[25]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=123) 


# In[26]:


y = pd.Series(y).array


# In[27]:


y_train_enc, y_test_enc = prepare_target_features(y_train, y_test)


# In[28]:


sm = SMOTE(random_state=123)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train_enc.ravel())


# In[29]:


rfc=RandomForestClassifier(random_state=123)


# In[30]:


param_grid = { 
    'n_estimators' : [100, 200, 400],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}


# In[32]:


grid = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
grid.fit(X_train_res, y_train_res.ravel())


# In[33]:


print(grid.best_params_)


# In[34]:


# fill in best params
model =RandomForestClassifier(random_state=123, n_estimators = 200, max_features= 'auto',  max_depth=7, criterion='gini')


# In[35]:


model.fit(X_train_res, y_train_res.ravel())
# 
predictions = model.predict(X_test)

print(confusion_matrix(y_test_enc, predictions))
print(classification_report(y_test_enc, predictions))


# In[62]:


# old
predictions = model.predict(X_test)

print(confusion_matrix(y_test_enc, predictions))
print(classification_report(y_test_enc, predictions))
print(accuracy_score(y_test_enc, predictions))


# # Random Forest - Coworker - Cluster

# In[36]:


file = "/Users/Eigenaar/Dropbox/&Master DSS/Thesis/Dataset"
MHdataset = pd.read_csv( file + "/MHdatasetComplete_Final_Clusters.csv")


# In[37]:


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


# In[38]:


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


# In[91]:


MHdataset.info()


# In[39]:


ClusterSelection = MHdataset.drop(['Employees', 'Remote_Work', 'Tech_Company',
                                    'MHbenefits', 'MHoptions', 'Wellness_Program',
                                     'Resources', 'Anonymity', 'Medical_Leave'], axis = 1)



# In[40]:


X = ClusterSelection.loc[:, ClusterSelection.columns != 'Coworkers']
y = MHdataset['Coworkers']


# In[41]:


X = X.drop(['Age_zscore','Power_Distance_zscore','Individualism_zscore','Masculinity_zscore', 'Uncertainty_Avoidance_zscore', 'Long_Term_Orientation_zscore', 'Indulgence_zscore'], axis=1)


# In[96]:


print(X.info())


# In[42]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=123) 


# In[43]:


y = pd.Series(y).array


# In[44]:


y_train_enc, y_test_enc = prepare_target_features(y_train, y_test)


# In[45]:


sm = SMOTE(random_state=123)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train_enc.ravel())


# In[46]:


rfc=RandomForestClassifier(random_state=123)


# In[47]:


param_grid = { 
    'n_estimators' : [100, 200, 400],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}


# In[48]:


grid = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
grid.fit(X_train_res, y_train_res.ravel())


# In[49]:


print(grid.best_params_)


# In[50]:


# fill in best params
model =RandomForestClassifier(random_state=123, n_estimators = 400, max_features= 'auto',  max_depth=8, criterion='gini')


# In[51]:


model.fit(X_train_res, y_train_res.ravel())
# 
predictions = model.predict(X_test)

print(confusion_matrix(y_test_enc, predictions))
print(classification_report(y_test_enc, predictions))


# In[106]:


# old
predictions = model.predict(X_test)

print(confusion_matrix(y_test_enc, predictions))
print(classification_report(y_test_enc, predictions))
print(accuracy_score(y_test_enc, predictions))


# --------------

# # Random Forest - Supervisor - Full

# In[ ]:


# https://www.kaggle.com/sociopath00/random-forest-using-gridsearchcv


# In[19]:


import pandas as pd 
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC  
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV


# In[2]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


# In[52]:


file = "/Users/Eigenaar/Dropbox/&Master DSS/Thesis/Dataset"
MHdataset = pd.read_csv( file + "/MHdatasetComplete_Final.csv")


# In[53]:


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


# In[ ]:


# random forest does not require feature scaling


# In[54]:


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


# In[55]:


X = MHdataset.loc[:, MHdataset.columns != 'Supervisors']
y = MHdataset['Supervisors']


# In[56]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=123) 


# In[57]:


y = pd.Series(y).array


# In[113]:


def prepare_target_features(y_train, y_test):
    le = LabelEncoder()
    le.fit(y_train)
    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)
    return y_train_enc, y_test_enc


# In[58]:


y_train_enc, y_test_enc = prepare_target_features(y_train, y_test)


# In[59]:


sm = SMOTE(random_state=123)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train_enc.ravel())


# In[60]:


rfc=RandomForestClassifier(random_state=123)


# In[61]:


param_grid = { 
    'n_estimators' : [100, 200, 400],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}


# In[62]:


grid = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
grid.fit(X_train_res, y_train_res.ravel())


# In[63]:


print(grid.best_params_)


# In[64]:


# fill in best params
model =RandomForestClassifier(random_state=123, n_estimators = 200, max_features= 'auto',  max_depth=8, criterion='gini')


# In[65]:


model.fit(X_train_res, y_train_res.ravel())

predictions = model.predict(X_test)

print(confusion_matrix(y_test_enc, predictions))
print(classification_report(y_test_enc, predictions))


# In[121]:


# old
predictions = model.predict(X_test)

print(confusion_matrix(y_test_enc, predictions))
print(classification_report(y_test_enc, predictions))
print(accuracy_score(y_test_enc, predictions))


# # Random Forest - Supervisor - FS

# In[66]:


file = "/Users/Eigenaar/Dropbox/&Master DSS/Thesis/Dataset"
MHdataset = pd.read_csv( file + "/MHdatasetComplete_Final.csv")


# In[67]:


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


# In[68]:


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


# In[125]:


MHdataset.info()


# In[69]:


FeatureSelection = MHdataset.loc[:,['Self_Employed', 'MHnegConsequences', 'PHnegConsequences','Coworkers', 'MHinterview', 'Indulgence']]



# In[70]:


X = FeatureSelection
y = MHdataset['Supervisors']


# In[71]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=123) 


# In[72]:


y = pd.Series(y).array


# In[73]:


y_train_enc, y_test_enc = prepare_target_features(y_train, y_test)


# In[74]:


sm = SMOTE(random_state=123)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train_enc.ravel())


# In[75]:


rfc=RandomForestClassifier(random_state=123)


# In[76]:


param_grid = { 
    'n_estimators' : [100, 200, 400],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}


# In[77]:


grid = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
grid.fit(X_train_res, y_train_res.ravel())


# In[78]:


print(grid.best_params_)


# In[79]:


# fill in best params
model =RandomForestClassifier(random_state=123, n_estimators = 400, max_features= 'auto',  max_depth=8, criterion='gini')


# In[80]:


model.fit(X_train_res, y_train_res.ravel())

predictions = model.predict(X_test)

print(confusion_matrix(y_test_enc, predictions))
print(classification_report(y_test_enc, predictions))


# In[137]:


# old
predictions = model.predict(X_test)

print(confusion_matrix(y_test_enc, predictions))
print(classification_report(y_test_enc, predictions))
print(accuracy_score(y_test_enc, predictions))


# # Random Forest - Supervisors - Cluster

# In[81]:


file = "/Users/Eigenaar/Dropbox/&Master DSS/Thesis/Dataset"
MHdataset = pd.read_csv( file + "/MHdatasetComplete_Final_Clusters.csv")


# In[82]:


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


# In[83]:


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


# In[141]:


MHdataset.info()


# In[84]:


ClusterSelection = MHdataset.drop(['Employees', 'Remote_Work', 'Tech_Company',
                                    'MHbenefits', 'MHoptions', 'Wellness_Program',
                                     'Resources', 'Anonymity', 'Medical_Leave'], axis = 1)



# In[85]:


X = ClusterSelection.loc[:, ClusterSelection.columns != 'Supervisors']
y = MHdataset['Supervisors']


# In[86]:


X = X.drop(['Age_zscore','Power_Distance_zscore','Individualism_zscore','Masculinity_zscore', 'Uncertainty_Avoidance_zscore', 'Long_Term_Orientation_zscore', 'Indulgence_zscore'], axis=1)


# In[145]:


print(X.info())


# In[87]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=123) 


# In[88]:


y = pd.Series(y).array


# In[89]:


y_train_enc, y_test_enc = prepare_target_features(y_train, y_test)


# In[90]:


sm = SMOTE(random_state=123)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train_enc.ravel())


# In[91]:


rfc=RandomForestClassifier(random_state=123)


# In[92]:


param_grid = { 
    'n_estimators' : [100, 200, 400],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}


# In[93]:


grid = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
grid.fit(X_train, y_train_enc)


# In[94]:


print(grid.best_params_)


# In[95]:


# fill in best params
model =RandomForestClassifier(random_state=123, n_estimators = 200, max_features= 'auto',  max_depth=8, criterion='entropy')


# In[96]:


model.fit(X_train_res, y_train_res.ravel())

predictions = model.predict(X_test)

print(confusion_matrix(y_test_enc, predictions))
print(classification_report(y_test_enc, predictions))


# In[155]:


# old
predictions = model.predict(X_test)

print(confusion_matrix(y_test_enc, predictions))
print(classification_report(y_test_enc, predictions))
print(accuracy_score(y_test_enc, predictions))

