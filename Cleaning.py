#!/usr/bin/env python
# coding: utf-8

# # 1. Cleaning

# ## 1.1 Importing Packages and Load Dataset

# In[1]:


import pandas as pd
import numpy as np
pd.options.display.max_columns = None
pd.options.display.max_rows = None


# In[2]:


import os
cwd = os.getcwd()


# In[3]:


file = "/Users/Eigenaar/Dropbox/&Master DSS/Thesis/Dataset"


# In[117]:


# Load first dataset

file = "/Users/Eigenaar/Dropbox/&Master DSS/Thesis/Dataset"

mh2014 = pd.read_excel(file + "/Mental Health in Tech Survey (Responses).xlsx")
print(type(mh2014))


# In[118]:


# First dataset feature name cleanup 

mh2014_2 = mh2014.rename(columns={'If you live in the United States, which state or territory do you live in?': 'State', 'Are you self-employed?': 'Self_Employed', 
                                  'Do you have a family history of mental illness?': 'Family_History', 'Have you sought treatment for a mental health condition?': 'Treatment',
                                 'If you have a mental health condition, do you feel that it interferes with your work?': 'Interference', 
                                 'How many employees does your company or organization have?': 'Employees', 'How easy is it for you to take medical leave for a mental health condition?': 'Medical_Leave', 
                                 'Do you think that discussing a mental health issue with your employer would have negative consequences?': 'MHnegConsequences', 
                                 'Do you think that discussing a physical health issue with your employer would have negative consequences?': 'PHnegConsequences', 
                                 'Would you be willing to discuss a mental health issue with your coworkers?': 'Coworkers', 'Would you be willing to discuss a mental health issue with your direct supervisor(s)?': 'Supervisors',
                                 'Would you bring up a mental health issue with a potential employer in an interview?': 'MHinterview', 
                                 'Would you bring up a physical health issue with a potential employer in an interview?': 'PHinterview', 
                                 'Do you feel that your employer takes mental health as seriously as physical health?': 'MHvsPH', 
                                 'Have you heard of or observed negative consequences for coworkers with mental health conditions in your workplace?': 'CoworkernegCons', 
                                 'Any additional notes or comments': 'Comments', 
                                 'Do you work remotely (outside of an office) at least 50% of the time?': 'Remote_Work', 
                                 'Is your employer primarily a tech company/organization?': 'Tech_Company', 
                                 'Does your employer provide mental health benefits?': 'MHbenefits', 
                                 'Do you know the options for mental health care your employer provides?': 'MHoptions', 
                                 'Has your employer ever discussed mental health as part of an employee wellness program?': 'Wellness_Program',
                                 'Does your employer provide resources to learn more about mental health issues and how to seek help?': 'Resources', 
                                 'Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources?': 'Anonymity',
                                 })


# In[119]:


mh2014_2.info()


# In[120]:


# Drop unnecessary columns 

mh2014 = mh2014_2.drop(['Timestamp', 'Comments', 'Interference'], axis=1)
mh2014.info()


# In[121]:


# Load second dataset

mh2016 = pd.read_csv( file + "/mental-heath-in-tech-2016_20161114.csv")
#print(mh2016.head())
#print(type(mh2016))
print(mh2016.info())


# In[122]:


# List of col names first dataset

#for col in mh2014.columns: 
    #print(col) 


# In[123]:


# Second dataset feature name cleanup 
mh2016_2 = mh2016.rename(columns={'What is your age?': 'Age', 'What is your gender?': 'Gender', 'What country do you live in?':'Country',
                                  'What US state or territory do you live in?': 'State', 'Are you self-employed?': 'Self_Employed', 
                                  'Do you have a family history of mental illness?': 'Family_History', 'Have you sought treatment for a mental health condition?': 'Treatment', 
                                 'How many employees does your company or organization have?': 'Employees', 'If a mental health issue prompted you to request a medical leave from work, asking for that leave would be:': 'Medical_Leave', 
                                 'Do you think that discussing a mental health disorder with your employer would have negative consequences?': 'MHnegConsequences', 
                                 'Do you think that discussing a physical health issue with your employer would have negative consequences?': 'PHnegConsequences', 
                                 'Would you feel comfortable discussing a mental health disorder with your coworkers?': 'Coworkers', 'Would you feel comfortable discussing a mental health disorder with your direct supervisor(s)?': 'Supervisors',
                                 'Would you bring up a mental health issue with a potential employer in an interview?': 'MHinterview', 
                                 'Would you bring up a physical health issue with a potential employer in an interview?': 'PHinterview', 
                                 'Do you feel that your employer takes mental health as seriously as physical health?': 'MHvsPH', 
                                 'Have you heard of or observed negative consequences for co-workers who have been open about mental health issues in your workplace?': 'CoworkernegCons', 
                                 'Do you work remotely?': 'Remote_Work', 
                                 'Is your employer primarily a tech company/organization?': 'Tech_Company', 
                                 'Does your employer provide mental health benefits as part of healthcare coverage?': 'MHbenefits', 
                                 'Do you know the options for mental health care available under your employer-provided coverage?': 'MHoptions', 
                                 'Has your employer ever formally discussed mental health (for example, as part of a wellness campaign or other official communication)?': 'Wellness_Program',
                                 'Does your employer offer resources to learn more about mental health concerns and options for seeking help?': 'Resources', 
                                 'Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources provided by your employer?': 'Anonymity',
                                 'Would you be willing to bring up a physical health issue with a potential employer in an interview?':'PHinterview', 
                                 'Have you ever sought treatment for a mental health issue from a mental health professional?': 'Treatment'})


# In[124]:


#for col in mh2016_2.columns: 
    #print(col) 


# In[125]:


extractedFeatures2016 = mh2016_2[['Age', 'Gender', 'Country', 'State', 'Self_Employed', 'Family_History',
                                 'Treatment', 'Employees', 'Remote_Work', 'Tech_Company', 'MHbenefits', 
                                 'MHoptions', 'Wellness_Program', 'Resources', 'Anonymity', 'Medical_Leave', 'MHnegConsequences',
                                 'PHnegConsequences', 'Coworkers', 'Supervisors', 'MHinterview', 'PHinterview',
                                 'MHvsPH', 'CoworkernegCons']]


# In[126]:


extractedFeatures2016.info()


# In[127]:


#for col in extractedFeatures2016.columns:
    #print(col)


# ## 1.2 Combine Datasets from 2014 and 2016

# In[128]:


# Combine both datasets
MHdataset = mh2014.append(extractedFeatures2016, ignore_index=True)


# In[129]:


print(MHdataset.head())


# In[130]:


MHdataset.info()


# ## 1.3 Categorical variables

# In[131]:


#print(MHdataset.dtypes)
#MHdataset["Gender"] = MHdataset["Gender"].astype("category")
MHdataset['Gender'] = pd.Categorical(MHdataset['Gender'])
MHdataset['Country'] = pd.Categorical(MHdataset['Country'])
MHdataset['State'] = pd.Categorical(MHdataset['State'])
MHdataset['Self_Employed'] = pd.Categorical(MHdataset['Self_Employed'])
MHdataset['Family_History'] = pd.Categorical(MHdataset['Family_History'])
MHdataset['Treatment'] = pd.Categorical(MHdataset['Treatment'])
MHdataset['Employees'] = pd.Categorical(MHdataset['Employees'])
MHdataset['Remote_Work'] = pd.Categorical(MHdataset['Remote_Work'])
MHdataset['Tech_Company'] = pd.Categorical(MHdataset['Tech_Company'])
MHdataset['MHbenefits'] = pd.Categorical(MHdataset['MHbenefits'])
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


# In[132]:


print(MHdataset.dtypes)


# ## 1.3.1 Clean Gender Variable 

# In[133]:


#print(MHdataset["Gender"].unique)
print(MHdataset['Gender'].cat.categories)


# In[134]:


# Clean gender variable 

MHdataset["Gender"].replace({" Female": "Female", "A little about you": None,
                            'AFAB': 'Other', 'Agender': 'Other', 'All':'Other', 
                            'Androgyne': 'Other', 'Androgynous': 'Other', 'Bigender': 'Other', 
                            'Cis Female': 'Female', 'Cis Male': 'Male', 'Cis Man': 'Male', 
                            'Cis female ': 'Female', 'Cis male': 'Male', 'Cis-woman': 'Female', 
                            'Cisgender Female': 'Female', 'Dude': 'Male', 'Enby': 'Other', 
                            'F': 'Female', 'Femake': 'Female', 'Female ': 'Female', 
                            'Female (cis)': 'Female', 'Female (props for making this a freeform field, though)': 'Female', 
                            'Female (trans)': 'Female', 'Female assigned at birth ': 'Female', 
                            'Female or Multi-Gender Femme': 'Other', 'Fluid': 'Other', 
                            'Genderfluid': 'Other', 'Genderfluid (born female)': 'Other', 'Genderflux demi-girl': 'Other', 
                            'Genderqueer': 'Other', 'Guy (-ish) ^_^': 'Male', 'Human': None, 'I identify as female.': 'Female', 
                            'I\'m a man why didn\'t you make this a drop down question. You should of asked sex? And I would of answered yes please. Seriously how much text can this take? ': 'Male', 
                            'M': 'Male', 'MALE': 'Male', 'Mail': 'Male', 'Make': 'Male', 'Mal': 'Male', 
                            'Male ': 'Male', 'Male (CIS)': 'Male', 'Male (cis)': 'Male', 'Male (trans, FtM)': 'Male',
                            'Male-ish': 'Male', 'Male.': 'Male', 'Male/genderqueer': 'Other', 
                            'Malr': 'Male', 'Man': 'Male', 'M|': 'Male', 'Nah': None , 'Neuter': 'Other', 
                            'Nonbinary': 'Other', 'Other/Transfeminine': 'Other', 'Queer': 'Other', 
                            'Sex is male': 'Male', 'Trans woman': 'Female', 'Trans-female': 'Female', 'Transgender woman': 'Female', 
                            'Transitioned, M2F': 'Female', 'Unicorn': None, 'Woman': 'Female', 'cis male': 'Male', 
                            'cis man': 'Male', 'cis-female/femme': 'Female', 'cisdude': 'Male', 'f': 'Female', 'fem': 'Female', 
                            'femail': 'Female', 'female': 'Female', 'female ': 'Female', 'female-bodied; no feelings about gender': 'Other', 
                            'female/woman': 'Female', 'fluid': 'Other', 'fm': 'Female', 
                            'genderqueer': 'Other', 'genderqueer woman': 'Other', 'human': None, 'm': 'Male', 
                            'mail': 'Male', 'maile': 'Male', 'male': 'Male', 'male ': 'Male',
                            'male 9:1 female, roughly': 'Other', 'male leaning androgynous': 'Male', 
                            'man': 'Male', 'msle': 'Male', 'mtf': 'Female', 'nb masculine': 'Other', 'non-binary': 'Other', 
                            'none of your business': None, 'ostensibly male, unsure what that really means': 'Male', 
                            'p': None, 'queer': 'Other', 'queer/she/they': 'Other', 'something kinda male?': 'Other', 
                            'woman': 'Female'}, inplace=True)


# In[135]:


MHdataset['Gender'] = pd.Categorical(MHdataset['Gender'])
print(MHdataset['Gender'].cat.categories)


# ## 1.3.2 Clean Country Variable

# In[136]:


print(MHdataset['Country'].cat.categories)


# In[137]:


MHdataset["Country"].replace({'United States of America': 'United States'}, inplace= True)


# In[138]:


MHdataset['Country'] = pd.Categorical(MHdataset['Country'])
print(MHdataset['Country'].cat.categories)


# ## 1.3.3 Clean State Variable

# In[139]:


print(MHdataset['State'].cat.categories)


# In[140]:


MHdataset['State'].replace({'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 
                           'California': 'CA', 'Colorado': 'CO', 'Connecticut': 'CT',
                           'Delaware': 'DE', 'District of Columbia': 'DC', 'Florida': 'FL',
                           'Georgia': 'GA', 'Idaho': 'ID', 'Iowa': 'IA', 'Illinois': 'IL', 'Indiana': 'IN',
                           'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
                           'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Missouri': 'MO', 'Montana': 'MT',
                           'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM',
                           'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK',
                           'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC',
                           'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT', 
                           'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV', 'Wisconsin': 'WI'}, inplace=True)


# In[141]:


MHdataset['State'] = pd.Categorical(MHdataset['State'])
print(MHdataset['State'].cat.categories)


# ## 1.3.4 Clean Self_Employed Variable

# In[142]:


print(MHdataset['Self_Employed'].cat.categories)


# In[143]:


MHdataset['Self_Employed'].replace({'No': 0, 'Yes': 1}, inplace=True)


# In[144]:


MHdataset['Self_Employed'] = pd.Categorical(MHdataset['Self_Employed'])
print(MHdataset['Self_Employed'].cat.categories)


# ## 1.3.5 Clean Family_History Variable

# In[145]:


print(MHdataset['Family_History'].cat.categories)
# Everything okay


# In[146]:


MHdataset['Family_History'].replace({'I don\'t know': 'Don\'t know','No': 0, 'Yes': 1}, inplace=True)


# In[147]:


MHdataset['Family_History'] = pd.Categorical(MHdataset['Family_History'])
print(MHdataset['Family_History'].cat.categories)


# ## 1.3.6 Clean Treatment Variable

# In[148]:


print(MHdataset['Treatment'].cat.categories)


# In[149]:


MHdataset['Treatment'].replace({'No': 0, 'Yes': 1}, inplace=True)


# In[150]:


MHdataset['Treatment'] = pd.Categorical(MHdataset['Treatment'])
print(MHdataset['Treatment'].cat.categories)


# ## 1.3.7 Clean Employees Variable

# In[151]:


print(MHdataset['Employees'].cat.categories)
# Change to Ordinal Variable 


# In[152]:


MHdataset['Employees'].replace({'1-5': 0, '6-25': 1, '26-100':2, '100-500':3, '500-1000':4, 'More than 1000':5 }, inplace=True)


# In[153]:


MHdataset['Employees'] = pd.Categorical(MHdataset['Employees'])
print(MHdataset['Employees'].cat.categories)


# ## 1.3.8 Clean Remote_Work Variable

# In[154]:


print(MHdataset['Remote_Work'].cat.categories)
# 2014 dataset -> Do you work remotely at least 50% of the time -> Yes or No
# 2016 dataset -> Do you work remotely? -> Always, Never or Sometimes 

# Decision:
# Change Always to Yes and Never to No


# In[155]:


MHdataset['Remote_Work'].replace({'Always': 'Yes', 'Never': 'No'}, inplace=True)


# In[156]:


MHdataset['Remote_Work'] = pd.Categorical(MHdataset['Remote_Work'])
print(MHdataset['Remote_Work'].cat.categories)


# In[157]:


MHdataset['Remote_Work'].replace({'No': 0, 'Sometimes': 1, 'Yes':2}, inplace=True)


# In[158]:


MHdataset['Remote_Work'] = pd.Categorical(MHdataset['Remote_Work'])
print(MHdataset['Remote_Work'].cat.categories)


# ## 1.3.9 Clean Tech_Company Variable

# In[159]:


print(MHdataset['Tech_Company'].cat.categories)


# In[160]:


MHdataset['Tech_Company'].replace({'No': 0, 'Yes': 1}, inplace=True)


# In[161]:


MHdataset['Tech_Company'] = pd.Categorical(MHdataset['Tech_Company'])
print(MHdataset['Tech_Company'].cat.categories)


# ## 1.3.10 Clean MHbenefits Variable

# In[162]:


print(MHdataset['MHbenefits'].cat.categories)


# In[163]:


MHdataset['MHbenefits'].replace({'I don\'t know': 'Don\'t know', 'Not eligible for coverage / N/A': 'NA', 'No': 0, 'Yes': 1 }, inplace=True)


# In[164]:


MHdataset['MHbenefits'].replace({'Don\'t know':'Other', 'NA': 'Other'}, inplace=True)


# In[165]:


MHdataset['MHbenefits'] = pd.Categorical(MHdataset['MHbenefits'])
print(MHdataset['MHbenefits'].cat.categories)


# ## 1.3.11 Clean MHoptions Variable

# In[166]:


print(MHdataset['MHoptions'].cat.categories)


# In[167]:


MHdataset['MHoptions'].replace({'I am not sure': 'Not sure', 'No': 0, 'Yes': 1}, inplace=True)


# In[168]:


MHdataset['MHoptions'].replace({'Not sure':'Other'}, inplace=True)


# In[169]:


MHdataset['MHoptions'] = pd.Categorical(MHdataset['MHoptions'])
print(MHdataset['MHoptions'].cat.categories)


# ## 1.3.12 Clean Wellness_Program Variable

# In[170]:


print(MHdataset['Wellness_Program'].cat.categories)


# In[171]:


MHdataset['Wellness_Program'].replace({'I don\'t know': 'Don\'t know', 'No': 0, 'Yes': 1}, inplace=True)


# In[172]:


MHdataset['Wellness_Program'].replace({'Don\'t know':'Other'}, inplace=True)


# In[173]:


MHdataset['Wellness_Program'] = pd.Categorical(MHdataset['Wellness_Program'])
print(MHdataset['Wellness_Program'].cat.categories)


# ## 1.3.13 Clean Resources Variable

# In[174]:


print(MHdataset['Resources'].cat.categories)


# In[175]:


MHdataset['Resources'].replace({'I don\'t know': 'Don\'t know', 'No': 0, 'Yes': 1}, inplace=True)


# In[176]:


MHdataset['Resources'].replace({'Don\'t know':'Other'}, inplace=True)


# In[177]:


MHdataset['Resources'] = pd.Categorical(MHdataset['Resources'])
print(MHdataset['Resources'].cat.categories)


# ## 1.3.14 Clean Anonymity Variable

# In[178]:


print(MHdataset['Anonymity'].cat.categories)


# In[179]:


MHdataset['Anonymity'].replace({'I don\'t know': 'Don\'t know', 'No': 0, 'Yes': 1}, inplace=True)


# In[180]:


MHdataset['Anonymity'].replace({'Don\'t know':'Other'}, inplace=True)


# In[181]:


MHdataset['Anonymity'] = pd.Categorical(MHdataset['Anonymity'])
print(MHdataset['Anonymity'].cat.categories)


# ## 1.3.15 Clean Medical_Leave Variable

# In[182]:


print(MHdataset['Medical_Leave'].cat.categories)


# In[183]:


MHdataset['Medical_Leave'].replace({'I don\'t know': 'Don\'t know', 'Very difficult':0, 
                                   'Somewhat difficult':1, 'Neither easy nor difficult':2,
                                   'Somewhat easy': 3, 'Very easy':4 }, inplace=True)


# In[184]:


MHdataset['Medical_Leave'].replace({'Don\'t know':'Other'}, inplace=True)


# In[185]:


MHdataset['Medical_Leave'] = pd.Categorical(MHdataset['Medical_Leave'])
print(MHdataset['Medical_Leave'].cat.categories)


# ## 1.3.16 Clean MHnegConsequences Variable

# In[186]:


print(MHdataset['MHnegConsequences'].cat.categories)
# Everything okay


# In[187]:


MHdataset['MHnegConsequences'].replace({'No': 0, 'Yes': 1}, inplace=True)


# In[188]:


MHdataset['MHnegConsequences'] = pd.Categorical(MHdataset['MHnegConsequences'])
print(MHdataset['MHnegConsequences'].cat.categories)


# ## 1.3.17 Clean PHnegConsequences Variable

# In[189]:


print(MHdataset['PHnegConsequences'].cat.categories)
# Everything okay


# In[190]:


MHdataset['PHnegConsequences'].replace({'No': 0, 'Yes': 1}, inplace=True)


# In[191]:


MHdataset['PHnegConsequences'] = pd.Categorical(MHdataset['PHnegConsequences'])
print(MHdataset['PHnegConsequences'].cat.categories)


# ## 1.3.18 Clean Coworkers Variable

# In[192]:


print(MHdataset['Coworkers'].cat.categories)

# 2014 dataset -> Yes, No or Some of them
# 2016 dataset -> Yes, No or Maybe

# Decision:
# Change Maybe and Some of them to Other


# In[193]:


MHdataset['Coworkers'].replace({'Maybe': 'Other', 'Some of them': 'Other', 'No': 0, 'Yes': 1}, inplace=True)


# In[194]:


MHdataset['Coworkers'] = pd.Categorical(MHdataset['Coworkers'])
print(MHdataset['Coworkers'].cat.categories)


# ## 1.3.19 Clean Supervisors Variable

# In[195]:


print(MHdataset['Supervisors'].cat.categories)

# 2014 dataset -> Yes, No or Some of them
# 2016 dataset -> Yes, No or Maybe

# Decision:
# Change Maybe and Some of them to Other


# In[196]:


MHdataset['Supervisors'].replace({'Maybe': 'Other', 'Some of them': 'Other', 'No': 0, 'Yes': 1}, inplace=True)


# In[197]:


MHdataset['Supervisors'] = pd.Categorical(MHdataset['Supervisors'])
print(MHdataset['Supervisors'].cat.categories)


# ## 1.3.20 Clean MHinterview Variable

# In[198]:


print(MHdataset['MHinterview'].cat.categories)
# Everything okay


# In[199]:


MHdataset['MHinterview'].replace({'No': 0, 'Yes': 1}, inplace=True)


# In[200]:


MHdataset['MHinterview'] = pd.Categorical(MHdataset['MHinterview'])
print(MHdataset['MHinterview'].cat.categories)


# ## 1.3.21 Clean PHinterview Variable

# In[201]:


print(MHdataset['PHinterview'].cat.categories)
# Everything okay


# In[202]:


MHdataset['PHinterview'].replace({'No': 0, 'Yes': 1}, inplace=True)


# In[203]:


MHdataset['PHinterview'] = pd.Categorical(MHdataset['PHinterview'])
print(MHdataset['PHinterview'].cat.categories)


# ## 1.3.22 Clean MHvsPH Varaible

# In[204]:


print(MHdataset['MHvsPH'].cat.categories)


# In[205]:


MHdataset['MHvsPH'].replace({'I don\'t know': 'Don\'t know', 'No': 0, 'Yes': 1}, inplace=True)


# In[206]:


MHdataset['MHvsPH'] = pd.Categorical(MHdataset['MHvsPH'])
print(MHdataset['MHvsPH'].cat.categories)


# ## 1.2.23 Clean CoworkernegCons Variable

# In[207]:


print(MHdataset['CoworkernegCons'].cat.categories)
# Everything okay


# In[208]:


MHdataset['CoworkernegCons'].replace({'No': 0, 'Yes': 1}, inplace=True)


# In[209]:


MHdataset['CoworkernegCons'] = pd.Categorical(MHdataset['CoworkernegCons'])
print(MHdataset['CoworkernegCons'].cat.categories)


# # 1.4 Continuous Variable

# In[210]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[211]:


plt.figure(figsize=(10,5))
plt.xlim(0,400)
plt.ylabel('Age')
sns.boxplot(x=MHdataset['Age'])
plt.show()


# In[212]:


# Delete illegal values for Age variable 

def CleanAge(Dataset, Variable):
    for value in Dataset[Variable]:
        if value == None:
            continue
        if value < 18: 
            Dataset[Variable].replace({value:None}, inplace=True) #werken vanaf 18
        if value > 80:
            Dataset[Variable].replace({value:None}, inplace=True)
            
    return Dataset[Variable]


# In[213]:


MHdataset['Age'] = CleanAge(MHdataset, 'Age')


# In[214]:


plt.figure(figsize=(10,5))
plt.xlim(0,100)
plt.ylabel('Age')
sns.boxplot(x=MHdataset['Age'])
plt.show()


# # 2. Recoding Variables

# In[215]:


2693-1591


# ## Recode Country Variable into Hofstede dimensions

# In[216]:


# Recode the country variable into the hofstede dimensions 
# Power Distance, Individualism, Masculinity, Uncertainty avoidance, Long Term Orientation, Indulgence
# https://www.hofstede-insights.com/product/compare-countries/

# visualise how many from which country
print(MHdataset['Country'].value_counts())


# In[217]:


import requests
import pandas as pd
from bs4 import BeautifulSoup


# In[218]:


def DatasetRecodedCountry(Dataset, Variable):

    # Content from web
    url = "https://www.hofstede-insights.com/wp-json/v1/country/"
    dicts = requests.get(url=url).json()

    #country = ['Afghanistan', 'Belgium']
    #testdf = pd.DataFrame([[1, "Afghanistan"], [2,'Belgium']], columns=['A', 'Country'])
    #print(testdf)

    # Initialize empty dataframe
    df = pd.DataFrame(columns=['Power_Distance', 'Individualism', 'Masculinity', 
                           'Uncertainty_Avoidance', 'Long_Term_Orientation', 'Indulgence'])
    #print(df)
    newlist = []
    for dictionary in dicts:
        newlist.append(dictionary['title'])
    #print(newlist)
    
    #print(newlist)

    for value in Dataset[Variable]:
        if value in newlist:
            dictionary2 = next(item for item in dicts if item["title"] == value or item["title"] == value+'*')
            #print(dictionary2)
            
            df = df.append({'Power_Distance': dictionary2['pdi'], 'Individualism':dictionary2['idv'], 'Masculinity':dictionary2['mas'],
                        'Uncertainty_Avoidance':dictionary2['uai'], 'Long_Term_Orientation':dictionary2['lto'],
                        'Indulgence':dictionary2['ind']}, ignore_index=True)
        elif value not in newlist:
        
            df = df.append({'Power_Distance': None, 'Individualism':None, 'Masculinity':None,
                            'Uncertainty_Avoidance':None, 'Long_Term_Orientation':None,
                            'Indulgence':None}, ignore_index=True)
    #print(df)

    result = pd.concat([Dataset, df], axis=1)
    #print(result.head())
    return result
            


# In[219]:


# Issue with the hofstede site 
MHdataset = DatasetRecodedCountry(MHdataset, 'Country')
print(MHdataset.head())


# In[220]:


MHdataset = MHdataset.drop(['State'], axis=1)
print(MHdataset.info())


# In[8]:


import pandas_profiling
import pandas as pd


# In[222]:


MHdataset.to_csv(file+'/MHdatasetComplete_test.csv', index = False)


# In[4]:


file = "/Users/Eigenaar/Dropbox/&Master DSS/Thesis/Dataset"
MHdataset = pd.read_csv( file + "/MHdatasetComplete_test.csv")


# In[224]:


MHdataset.info()


# In[225]:


# doesn't load when dtype = category
MHdataset.profile_report()


# In[6]:


df = MHdataset[["Age", "Power_Distance", "Individualism", "Masculinity", "Uncertainty_Avoidance", "Long_Term_Orientation", "Indulgence"]]

corrMatrix = df.corr()


# In[7]:


print(corrMatrix)


# In[ ]:


file = "/Users/Eigenaar/Dropbox/&Master DSS/Thesis/Dataset"


# In[ ]:


MHdataset_2 = pd.read_csv( file + "/MHdatasetComplete.csv")
print(MHdataset_2.head())


# In[ ]:


MHdataset['Power_Distance'] = MHdataset_2['Power_Distance']


# In[ ]:


MHdataset['Individualism'] = MHdataset_2['Individualism']
MHdataset['Masculinity'] = MHdataset_2['Masculinity']
MHdataset['Uncertainty_Avoidance'] = MHdataset_2['Uncertainty_Avoidance']
MHdataset['Long_Term_Orientation'] = MHdataset_2['Long_Term_Orientation']
MHdataset['Indulgence'] = MHdataset_2['Indulgence']


# In[ ]:


print(MHdataset.head())


# ## Missing values: replace by majority

# In[9]:


MHdataset = MHdataset.fillna(MHdataset.mode().iloc[0])


# In[10]:


print(MHdataset.info())


# In[11]:


MHdataset.profile_report()


# ## Save dataset as CSV - Complete dataset

# In[ ]:


MHdataset.to_csv(file+'/MHdatasetComplete_Final.csv', index = False)


# ## Save dataset as CSV - 2014 dataset

# In[ ]:


MHdataset2014 = MHdataset.iloc[:1260, :]
print(MHdataset2014.info())


# In[ ]:


MHdataset2014.to_csv(file+'/MHdataset2014_Final.csv', index = False)


# ## Save dataset as CSV - 2016 Dataset

# In[ ]:


MHdataset2016 = MHdataset.iloc[1260:,:]
print(MHdataset2016.info())


# In[ ]:


MHdataset2016.to_csv(file+'/MHdataset2016_Final.csv', index = False)


# # 3. EDA

# ### Load Dataset

# In[ ]:


file = "/Users/Eigenaar/Dropbox/&Master DSS/Thesis/Dataset"
MHdataset = pd.read_csv( file + "/MHdatasetComplete.csv")


# In[ ]:


MHdataset.info()


# In[ ]:


# Maybe keep NaN -> feature selection
#MHdataset = MHdataset.replace({np.nan: None})

#https://stackoverflow.com/questions/32617811/imputation-of-missing-values-for-categories-in-pandas
MHdataset = MHdataset.fillna(MHdataset.mode().iloc[0])


# In[ ]:


MHdataset.info()


# In[ ]:


MHdataset = MHdataset.drop(['State'], axis=1)


# In[ ]:


MHdataset.head()


# ## 3.1 Report

# In[ ]:


pip install pandas-profiling


# In[ ]:


import pandas_profiling
import pandas as pd


# In[ ]:


MHdataset.profile_report()


# In[ ]:


MHdataset2016 = pd.read_csv( file + "/MHdataset2016_Final.csv")


# In[ ]:


MHdataset2016.info()


# In[ ]:





# ## 3.1.1 Relationship between target variables

# # ENCODING
# 
# https://pbpython.com/categorical-encoding.html

# In[ ]:





# In[ ]:


#https://stackoverflow.com/questions/48035381/correlation-among-multiple-categorical-variables-pandas


# In[ ]:


#https://towardsdatascience.com/processing-and-visualizing-multiple-categorical-variables-with-python-nbas-schedule-challenges-b48453bff813


# ## 3.2 Outlier Analysis

# In[ ]:


#https://machinelearningmastery.com/model-based-outlier-detection-and-removal-in-python/


# In[ ]:


#https://towardsdatascience.com/ways-to-detect-and-remove-the-outliers-404d16608dba


# In[ ]:


#https://medium.com/@shreyash0023/anomaly-detection-on-a-categorical-and-continuous-dataset-d5af7aa287d2


# ## 3.3 Missing Data Analysis 

# In[ ]:


# https://machinelearningmastery.com/handle-missing-data-python/

print(MHdataset.describe())


# # 4. K-means Clustering or C-means Clustering

# In[ ]:


# K-means
## https://realpython.com/k-means-clustering-python/

# C-means
## https://pypi.org/project/fuzzy-c-means/
## https://www.kaggle.com/prateekk94/fuzzy-c-means-clustering-on-iris-dataset -> use this!!
## https://pythonhosted.org/scikit-fuzzy/auto_examples/plot_cmeans.html


# In[ ]:


pip install fuzzy-c-means


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
from fcmeans import FCM
from matplotlib import pyplot as plt


# In[ ]:


# optimise number of clusters 

for cluster in range(4):
    fcm = FCM(n_clusters=cluster)
    fcm.fit(X)


# In[ ]:


# outputs
fcm_centers = fcm.centers
fcm_labels = fcm.predict(X)

# plot result
f, axes = plt.subplots(1, 2, figsize=(11,5))
axes[0].scatter(X[:,0], X[:,1], alpha=.1)
axes[1].scatter(X[:,0], X[:,1], c=fcm_labels, alpha=.1)
axes[1].scatter(fcm_centers[:,0], fcm_centers[:,1], marker="+", s=500, c='w')
plt.savefig('images/basic-clustering-output.jpg')
plt.show()


# # 5. Feature selection (Gradient Tree Boosting) - Coworkers

# In[ ]:


# https://machinelearningmastery.com/gradient-boosting-machine-ensemble-in-python/


# In[ ]:


y = MHdataset['Coworkers']
X = MHdataset.loc[:, MHdataset.columns != 'Coworkers']


# In[ ]:


# evaluate gradient boosting algorithm for classification
from numpy import mean
from numpy import std
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier


# In[ ]:


# TREAT MISSING VARIABLES FIRST
# define the model
model = GradientBoostingClassifier()

# define the evaluation method
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

# evaluate the model on the dataset
n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
print(n_scores)

# report performance
print('Mean Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))


# In[ ]:


pip install xgboost


# In[ ]:


conda install -c conda-forge py-xgboost


# In[ ]:


conda update -n base -c defaults conda


# In[ ]:


from sklearn.preprocessing import OneHotEncoder


# In[ ]:


y = np.reshape(y, (-1, 1))


# In[ ]:


print(y.shape)
print(X.shape)


# In[ ]:


onehot = OneHotEncoder()
y = onehot.fit_transform(y)
x = onehot.transform(X)


# In[ ]:


# Feature importance
# https://machinelearningmastery.com/feature-importance-and-feature-selection-with-xgboost-in-python/

from numpy import loadtxt
from xgboost import XGBClassifier
from matplotlib import pyplot

model = XGBClassifier()
model.fit(X, y)

print(model.feature_importances_)
# plot
pyplot.bar(range(len(model.feature_importances_)), model.feature_importances_)
pyplot.show()


# # 6. SVM

# In[ ]:


# scaling variables -> Z scores 
# ONLY CONTINUOUS VARIABLES: 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np

from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import  StandarScaler 
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=123)

column_trans = ColumnTransformer(
    [('scaler', StandardScaler(), ['Age'])], remainder='passthrough') 
     
column_scaled = column_trans.fit_transform(X)

column_scaled


# In[ ]:


# Optimization hyperparamters 
# https://automl.github.io/SMAC3/master/examples/SMAC4HPO_svm.html


# In[ ]:


from sklearn import svm

clf = svm.SVC(decision_function_shape='ovo') # multi-class classification: one vs rest
clf.fit(X_train, y_train)
clf.predict(X_test)

