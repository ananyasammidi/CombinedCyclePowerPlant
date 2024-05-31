#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns


# In[5]:


#loading data
df = pd.read_csv('ccp.csv')


# In[6]:


#grabbing the first 10 rows to get a picture of what the data looks like
df.head(10)


# In[7]:


# creating a data frame with temperature, exhaust vacuum, relative humidity, and ambient pressure features
X = df[['AT', 'V', 'AP', 'RH']]
# creating a data frame with the target variable
y = df['PE']


# In[8]:


# (9568, 4)
X.shape


# In[9]:


from sklearn.model_selection import train_test_split


# In[59]:


# splitting the data into a training set and a test set, using a third of the data as the test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[35]:


# (6410, 4)
X_train.shape


# In[50]:


#importing LinearRegression Model and appropriate metrics
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
model = LinearRegression()


# In[51]:


#fitting model onto training set
model.fit(X_train, y_train)


# In[52]:


# making predicitions with testing set
y_pred = model.predict(X_test)


# In[53]:


# calculating metrics to compare models
mean_absolute = mean_absolute_error(y_test, y_pred)
mean_squared = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# In[54]:


print(mean_absolute) #3.6641203776437523
print(mean_squared) #21.246921543137933
print(r2) #0.9271761736761965


# In[55]:


from sklearn.tree import DecisionTreeRegressor


# In[56]:


# fitting the decision tree and creating predictions
decision_tree = DecisionTreeRegressor(random_state=42)
decision_tree.fit(X_train, y_train)
y_pred_decision_tree = decision_tree.predict(X_test)


# In[57]:


#calculating metrics for decision tree regression
mean_absolute_error_tree = mean_absolute_error(y_test, y_pred_decision_tree)
mean_squared_error_tree = mean_squared_error(y_test, y_pred_decision_tree)
r2_tree = r2_score(y_test, y_pred_decision_tree)


# In[58]:


print(mean_absolute_error_tree) #3.1368017732742244
print(mean_squared_error_tree) #20.31034762507916
print(r2_tree) #0.9303862809008898
# decision tree regression performs better than linear regression

