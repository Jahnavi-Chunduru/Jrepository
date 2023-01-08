#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


train=pd.read_csv(r"C:\Users\pmjv0\Downloads\train.csv\train.csv")


# In[4]:


test=pd.read_csv(r"C:\Users\pmjv0\Downloads\test.csv\test.csv")


# In[5]:


print("training samples:",train.shape)


# In[6]:


print("testing samples:",test.shape)


# In[7]:


train.dtypes


# In[9]:


train.head(10)


# In[10]:


train.describe()


# In[11]:


#number of null values for each column
train.isnull().sum()


# In[17]:


train.columns


# In[15]:


#target variable distribution through kdeplot and boxplot
import matplotlib.pyplot as mat
import seaborn as sns
mat.figure(figsize=(15,7))
sns.kdeplot(train.site_eui , color = "#ffd329")


# In[18]:


sns.boxplot(train.site_eui , color = "#ffd514")


# In[23]:


def countplot_features(df_train, feature, title):
    '''Takes a column from the dataframe and plots the distribution (after count).'''
    
           
    mat.figure(figsize = (10, 5))
    
    sns.countplot(df_train[feature], color = '#ff355d')
        
    mat.title(title, fontsize=15)    
    mat.show();
cat_features=['State_Factor', 'building_class', 'facility_type']
for feature in cat_features:
    fig = countplot_features(train, feature=feature, title = "Frequency of "+ feature)


# In[25]:


# missing values
import numpy as np
train['year_built'] =train['year_built'].replace(np.nan, 2022)
train['energy_star_rating']=train['energy_star_rating'].replace(np.nan,train['energy_star_rating'].mean())
train['direction_max_wind_speed']= train['direction_max_wind_speed'].replace(np.nan,train['direction_max_wind_speed'].mean())
train['direction_peak_wind_speed']= train['direction_peak_wind_speed'].replace(np.nan,train['direction_peak_wind_speed'].mean())
train['max_wind_speed']=train['max_wind_speed'].replace(np.nan,train['max_wind_speed'].mean())
train['days_with_fog']=train['days_with_fog'].replace(np.nan,train['days_with_fog'].mean())

test['year_built'] =test['year_built'].replace(np.nan, 2022)
test['energy_star_rating']=test['energy_star_rating'].replace(np.nan,test['energy_star_rating'].mean())
test['direction_max_wind_speed']= test['direction_max_wind_speed'].replace(np.nan,test['direction_max_wind_speed'].mean())
test['direction_peak_wind_speed']= test['direction_peak_wind_speed'].replace(np.nan,test['direction_peak_wind_speed'].mean())
test['max_wind_speed']=test['max_wind_speed'].replace(np.nan,test['max_wind_speed'].mean())
test['days_with_fog']=test['days_with_fog'].replace(np.nan,test['days_with_fog'].mean())


# In[26]:


train.isnull().sum()


# In[27]:


test.isnull().sum()


# In[28]:


#encoding categorical attributes
from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()

train['State_Factor']= le.fit_transform(train['State_Factor']).astype("uint8")
test['State_Factor']= le.fit_transform(test['State_Factor']).astype("uint8")

train['building_class']= le.fit_transform(train['building_class']).astype("uint8")
test['building_class']= le.fit_transform(test['building_class']).astype("uint8")

train['facility_type']= le.fit_transform(train['facility_type']).astype("uint8")
test['facility_type']= le.fit_transform(test['facility_type']).astype("uint8")


# In[29]:


train.head()


# In[30]:


test.head()

