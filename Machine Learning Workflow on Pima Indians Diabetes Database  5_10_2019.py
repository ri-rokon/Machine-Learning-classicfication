#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pylab as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.linear_model import LinearRegression
from sklearn import datasets
import pandas as pd
data_frame = pd.read_csv("E:/thesisCode/data/pima-data.csv")


# In[2]:


# Deleting 'skin' column completely
del data_frame['skin']
# Checking if the action was successful or not
data_frame.head()


# In[3]:


# Mapping the values
map_diabetes = {True : 1, False : 0}

# Setting the map to the data_frame
data_frame['diabetes'] = data_frame['diabetes'].map(map_diabetes)

# Let's see what we have done
data_frame.head()


# In[4]:


x= data_frame
y=x


# In[5]:


y.head(3)


# In[6]:


x.head(3)


# In[7]:


col_list = ['num_preg' , 'glucose_conc' , 'diastolic_bp' , 'thickness' , 'insulin' , 'bmi' , 'diab_pred' , 'age']
col_list_y=['diabetes']


# In[8]:


x = x[col_list]


# In[9]:


y = y[col_list_y]


# In[10]:


x.head(3)


# In[11]:


y.head(3)


# In[12]:


from sklearn.impute import SimpleImputer
#imp_mean = SimpleImputer(missing_values=0, strategy='mean')
my_imputer = SimpleImputer(missing_values=0, strategy='mean')

x = pd.DataFrame(my_imputer.fit_transform(x))


# In[13]:


x.head(3)


# In[14]:


x.columns = ['num_preg' , 'glucose_conc' , 'diastolic_bp' , 'thickness' , 'insulin' , 'bmi' , 'diab_pred' , 'age']


# In[16]:


x.head(3)


# In[21]:


from sklearn.model_selection import train_test_split

# Saving 30% for testing
split_test_size = 0.30

# Splitting using scikit-learn train_test_split function

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = split_test_size, random_state = 42)


# In[19]:


x_train.head(3)


# In[20]:


x_test.head(3)


# In[22]:


y_train.head(3)


# In[23]:


y_test.head(3)


# In[39]:


from sklearn.naive_bayes import GaussianNB

# create Gaussian Naive Bayes model object and train it with the data
nb_model = GaussianNB()

nb_model.fit(x_train, y_train.values.ravel())


# In[41]:


prediction_from_trained_data = nb_model.predict(x_train)


# In[42]:


# performance metrics library
from sklearn import metrics

# get current accuracy of the model

accuracy = metrics.accuracy_score(y_train, prediction_from_trained_data)

print ("Accuracy of our naive bayes model is : {0:.4f}".format(accuracy))


# In[44]:


# this returns array of predicted results from test_data
prediction_from_test_data = nb_model.predict(x_test)

accuracy = metrics.accuracy_score(y_test, prediction_from_test_data)

print ("Accuracy of our naive bayes model is: {0:0.4f}".format(accuracy))


# In[50]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


# In[51]:


models = []
models.append(('KNN', KNeighborsClassifier()))
models.append(('SVC', SVC()))
models.append(('LR', LogisticRegression()))
models.append(('DT', DecisionTreeClassifier()))
models.append(('GNB', GaussianNB()))
models.append(('RF', RandomForestClassifier()))
models.append(('GB', GradientBoostingClassifier()))


# In[52]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score


# In[57]:


names = []
scores = []
for name, model in models:
    model.fit(x_train, y_train.values.ravel())
    y_pred = model.predict(x_test)
    scores.append(accuracy_score(y_test, y_pred))
    names.append(name)
tr_split = pd.DataFrame({'Name': names, 'Score': scores})
print(tr_split)


# In[ ]:




