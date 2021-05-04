#!/usr/bin/env python
# coding: utf-8

# In[31]:


import pandas as pd


# In[32]:


#Loading dataset
df = pd.read_csv("airfoil_self_noise.csv")


# In[33]:


df


# In[34]:


#Checking for null values
df.isnull()


# In[35]:


#Checking correlation between the variables
df.corr()
#Displacement is 70% correlated with Angle of Attack


# In[36]:


df.drop(columns = ["Displacement"], inplace = True)


# In[37]:


import numpy as np
from sklearn import linear_model


# In[38]:


df.columns


# In[44]:


regr = LinearRegression()
regr.fit(df[['Frquency(Hz)', 'Angle_of_Attack', 'Chord_Length',
       'Free_stream_velocity']], df[['Sound_pressure_level']])


# In[50]:


print(regr.coef_)


# In[52]:


import matplotlib.pyplot as plt


# In[56]:


predictedSound = regr.predict([[200, 130, 120, 890]])


# In[57]:


print(predictedSound)


# In[72]:


df = df.rename(columns = {'Frquency(Hz)':'frequency'})


# In[73]:


#import necessary libraries 
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols

#fit simple linear regression model
model = ols('Sound_pressure_level ~ Angle_of_Attack + frequency + Chord_Length + Free_stream_velocity', data=df).fit()

#view model summary
print(model.summary())


# In[74]:


#create residual vs. predictor plot for 'frequency'
fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_regress_exog(model, 'frequency', fig=fig)


# In[75]:


#create residual vs. predictor plot for 'Angle_of_Attack'
fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_regress_exog(model, 'Angle_of_Attack', fig=fig)


# In[76]:


#create residual vs. predictor plot for 'Chord_Length'
fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_regress_exog(model, 'Chord_Length', fig=fig)


# In[77]:


#create residual vs. predictor plot for 'Free_stream_velocity'
fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_regress_exog(model, 'Free_stream_velocity', fig=fig)


# In[79]:


df['log_sound'] = np.log10(df['Sound_pressure_level'])


# In[80]:


#import necessary libraries 
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols

#fit simple linear regression model
log_model = ols('log_sound ~ Angle_of_Attack + frequency + Chord_Length + Free_stream_velocity', data=df).fit()

#view model summary
print(log_model.summary())


# In[81]:


df.columns


# In[83]:


#Training and testing
from sklearn.model_selection import train_test_split


# In[87]:


x_train, x_test, y_train, y_test = train_test_split(df[['frequency', 'Angle_of_Attack', 'Chord_Length',
       'Free_stream_velocity']], df[['Sound_pressure_level']], test_size = 0.2)


# In[89]:


from sklearn.linear_model import LinearRegression


# In[127]:


lr = LinearRegression()


# In[128]:


lr.fit(x_train, y_train)


# In[129]:


x_test_1 = lr.predict(x_test)


# In[130]:


y_test


# In[136]:


x_test_1


# In[135]:


#Checing accuracy of the model
lr.score(x_test, y_test)

