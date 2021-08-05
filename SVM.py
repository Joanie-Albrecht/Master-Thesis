#!/usr/bin/env python
# coding: utf-8

# # SVM - Coworkers - Full

# In[110]:


#https://www.vebuso.com/2020/03/svm-hyperparameter-tuning-using-gridsearchcv/

import pandas as pd 

from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import LabelEncoder


# In[111]:


from sklearn.svm import SVC  
from sklearn.metrics import classification_report, confusion_matrix  
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV


# In[113]:


# load dataset
file = "/Users/Eigenaar/Dropbox/&Master DSS/Thesis/Dataset"
MHdataset = pd.read_csv( file + "/MHdatasetComplete_Final.csv")


# In[114]:


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


# In[115]:


MHdataset = Z_Scores(MHdataset)


# In[116]:


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


MHdataset.info()


# In[117]:


X = MHdataset.loc[:, MHdataset.columns != 'Coworkers']
y = MHdataset['Coworkers']


# In[118]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=123) 


# In[119]:


X = X.drop(['Age','Power_Distance','Individualism','Masculinity', 'Uncertainty_Avoidance', 'Long_Term_Orientation', 'Indulgence'], axis=1)


# In[120]:


y = pd.Series(y).array


# In[13]:


def prepare_target_features(y_train, y_test):
    le = LabelEncoder()
    le.fit(y_train)
    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)
    return y_train_enc, y_test_enc


# In[121]:


y_train_enc, y_test_enc = prepare_target_features(y_train, y_test)


# In[122]:


param_grid = {'C': [1, 10], 'gamma': [0.01,0.001],'kernel': ['rbf', 'sigmoid']}


# In[15]:


from imblearn.over_sampling import SMOTE

print("Number transactions X_train dataset: ", X_train.shape)
print("Number transactions y_train dataset: ", y_train_enc.shape)
print("Number transactions X_test dataset: ", X_test.shape)
print("Number transactions y_test dataset: ", y_test_enc.shape)


# In[16]:


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


# In[17]:


param_grid = {'C': [1, 10], 'gamma': [0.01,0.001],'kernel': ['rbf', 'sigmoid']}

model = SVC()
grid = GridSearchCV(model, param_grid, cv=5, verbose=2)
grid.fit(X_train_res, y_train_res.ravel())


# In[18]:


print(grid.best_estimator_)


# In[19]:


# Model specifieren!
model = SVC(C = 10, gamma = 0.01, random_state = 123)

model.fit(X_train_res, y_train_res.ravel())
# 
predictions = model.predict(X_test)

print(confusion_matrix(y_test_enc, predictions))
print(classification_report(y_test_enc, predictions))


# # SVM - Coworkers - FS

# In[126]:


file = "/Users/Eigenaar/Dropbox/&Master DSS/Thesis/Dataset"
MHdataset = pd.read_csv( file + "/MHdatasetComplete_Final.csv")


# In[127]:


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


# In[128]:


MHdataset = Z_Scores(MHdataset)


# In[129]:


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


# In[130]:


FeatureSelection = MHdataset.loc[:,['Self_Employed', 'MHnegConsequences', 'PHnegConsequences','Supervisors', 'MHinterview', 'Indulgence']]


# In[131]:


X = FeatureSelection
y = MHdataset['Coworkers']


# In[132]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=123) 


# In[133]:


y = pd.Series(y).array


# In[136]:


y_train_enc, y_test_enc = prepare_target_features(y_train, y_test)


# In[29]:


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


# In[30]:


param_grid = {'C': [1, 10], 'gamma': [0.01,0.001],'kernel': ['rbf', 'sigmoid']}

model = SVC()
grid = GridSearchCV(model, param_grid, cv=5, verbose=2)
grid.fit(X_train_res, y_train_res.ravel())


# In[31]:


print(grid.best_params_)


# In[32]:


model = SVC(C = 10, gamma = 0.01, kernel = 'rbf', random_state = 123)

model.fit(X_train_res, y_train_res.ravel())
# 
predictions = model.predict(X_test)

print(confusion_matrix(y_test_enc, predictions))
print(classification_report(y_test_enc, predictions))


# # SVM - Coworker - Cluster

# In[33]:


file = "/Users/Eigenaar/Dropbox/&Master DSS/Thesis/Dataset"
MHdataset = pd.read_csv( file + "/MHdatasetComplete_Final_Clusters.csv")


# In[34]:


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


# In[35]:


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


# In[36]:


MHdataset.info()


# In[37]:


ClusterSelection = MHdataset.drop(['Employees', 'Remote_Work', 'Tech_Company',
                                    'MHbenefits', 'MHoptions', 'Wellness_Program',
                                     'Resources', 'Anonymity', 'Medical_Leave'], axis = 1)


# In[38]:


X = ClusterSelection.loc[:, ClusterSelection.columns != 'Coworkers']
y = MHdataset['Coworkers']


# In[39]:


X = X.drop(['Age','Power_Distance','Individualism','Masculinity', 'Uncertainty_Avoidance', 'Long_Term_Orientation', 'Indulgence'], axis=1)


# In[40]:


print(X.info())


# In[41]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=123)


# In[42]:


y = pd.Series(y).array


# In[43]:


y_train_enc, y_test_enc = prepare_target_features(y_train, y_test)


# In[44]:


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


# In[45]:


param_grid = {'C': [1, 10], 'gamma': [0.01,0.001],'kernel': ['rbf', 'sigmoid']}

model = SVC()
grid = GridSearchCV(model, param_grid, cv=5, verbose=2)
grid.fit(X_train_res, y_train_res.ravel())


# In[46]:


print(grid.best_estimator_)


# In[47]:


model = SVC(C = 10, gamma = 0.01, random_state = 123)

model.fit(X_train_res, y_train_res.ravel())
# 
predictions = model.predict(X_test)

print(confusion_matrix(y_test_enc, predictions))
print(classification_report(y_test_enc, predictions))


# -------------------

# # SVM - Supervisors - Full

# In[1]:


#https://www.vebuso.com/2020/03/svm-hyperparameter-tuning-using-gridsearchcv/

import pandas as pd 

from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import LabelEncoder


# In[2]:


from sklearn.svm import SVC  
from sklearn.metrics import classification_report, confusion_matrix  
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV


# In[140]:


# load dataset
file = "/Users/Eigenaar/Dropbox/&Master DSS/Thesis/Dataset"
MHdataset = pd.read_csv( file + "/MHdatasetComplete_Final.csv")


# In[141]:


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


# In[71]:


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


# In[142]:


MHdataset = Z_Scores(MHdataset)


# In[143]:


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


# In[144]:


MHdataset.info()


# In[145]:


X = MHdataset.loc[:, MHdataset.columns != 'Supervisors']
y = MHdataset['Supervisors']


# In[146]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=123) 


# In[147]:


X = X.drop(['Age','Power_Distance','Individualism','Masculinity', 'Uncertainty_Avoidance', 'Long_Term_Orientation', 'Indulgence'], axis=1)


# In[148]:


y = pd.Series(y).array


# In[79]:


def prepare_target_features(y_train, y_test):
    le = LabelEncoder()
    le.fit(y_train)
    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)
    return y_train_enc, y_test_enc


# In[149]:


y_train_enc, y_test_enc = prepare_target_features(y_train, y_test)


# In[103]:


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


# In[104]:


param_grid = {'C': [1, 10], 'gamma': [0.01,0.001],'kernel': ['rbf', 'sigmoid']}

model = SVC()
grid = GridSearchCV(model, param_grid, cv=5, verbose=2)
grid.fit(X_train_res, y_train_res.ravel())


# In[105]:


print(grid.best_estimator_)


# In[106]:


model = SVC(C = 10, gamma = 0.01, random_state = 123)

model.fit(X_train_res, y_train_res.ravel())
# 
predictions = model.predict(X_test)

print(confusion_matrix(y_test_enc, predictions))
print(classification_report(y_test_enc, predictions))


# # SVM - Supervisors - FS

# In[153]:


file = "/Users/Eigenaar/Dropbox/&Master DSS/Thesis/Dataset"
MHdataset = pd.read_csv( file + "/MHdatasetComplete_Final.csv")


# In[154]:


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


# In[155]:


MHdataset = Z_Scores(MHdataset)


# In[156]:


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


# In[157]:


FeatureSelection = MHdataset.loc[:,['Self_Employed', 'MHnegConsequences', 'PHnegConsequences','Coworkers', 'MHinterview', 'Indulgence']]


# In[158]:


X = FeatureSelection
y = MHdataset['Supervisors']


# In[159]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=123) 


# In[160]:


y = pd.Series(y).array


# In[161]:


y_train_enc, y_test_enc = prepare_target_features(y_train, y_test)


# In[74]:


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


# In[75]:


param_grid = {'C': [1, 10], 'gamma': [0.01,0.001],'kernel': ['rbf', 'sigmoid']}

model = SVC()
grid = GridSearchCV(model, param_grid, cv=5, verbose=2)
grid.fit(X_train_res, y_train_res.ravel())


# In[76]:


print(grid.best_estimator_)


# In[77]:


model = SVC(C = 10, gamma = 0.01, random_state = 123)

model.fit(X_train_res, y_train_res.ravel())
# 
predictions = model.predict(X_test)

print(confusion_matrix(y_test_enc, predictions))
print(classification_report(y_test_enc, predictions))


# # SVM - Supervisor - Cluster

# In[167]:


file = "/Users/Eigenaar/Dropbox/&Master DSS/Thesis/Dataset"
MHdataset = pd.read_csv( file + "/MHdatasetComplete_Final_Clusters.csv")


# In[168]:


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


# In[169]:


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


# In[81]:


MHdataset.info()


# In[170]:


ClusterSelection = MHdataset.drop(['Employees', 'Remote_Work', 'Tech_Company',
                                    'MHbenefits', 'MHoptions', 'Wellness_Program',
                                     'Resources', 'Anonymity', 'Medical_Leave'], axis = 1)


# In[171]:


X = ClusterSelection.loc[:, ClusterSelection.columns != 'Supervisors']
y = MHdataset['Supervisors']


# In[172]:


X = X.drop(['Age','Power_Distance','Individualism','Masculinity', 'Uncertainty_Avoidance', 'Long_Term_Orientation', 'Indulgence'], axis=1)


# In[85]:


print(X.info())


# In[173]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=123)


# In[174]:


y = pd.Series(y).array


# In[175]:


y_train_enc, y_test_enc = prepare_target_features(y_train, y_test)


# In[89]:


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


# In[90]:


param_grid = {'C': [1, 10], 'gamma': [0.01,0.001],'kernel': ['rbf', 'sigmoid']}

model = SVC()
grid = GridSearchCV(model, param_grid, cv=5, verbose=2)
grid.fit(X_train_res, y_train_res.ravel())


# In[91]:


print(grid.best_estimator_)


# In[92]:


model = SVC(C = 10, gamma = 0.01, random_state = 123)

model.fit(X_train_res, y_train_res.ravel())
# 
predictions = model.predict(X_test)

print(confusion_matrix(y_test_enc, predictions))
print(classification_report(y_test_enc, predictions))

