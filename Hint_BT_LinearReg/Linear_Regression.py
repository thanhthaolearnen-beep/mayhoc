#!/usr/bin/env python
# coding: utf-8

# # Linear Regression on Boston Housing Dataset
# 
# This data was originally a part of UCI Machine Learning Repository and has been removed now. This data also ships with the scikit-learn library. 
# There are 506 samples and 13 feature variables in this data-set. The objective is to predict the value of prices of the house using the given features.
# 
# The description of all the features is given below:
# 
#   **CRIM**: Per capita crime rate by town
# 
#   **ZN**: Proportion of residential land zoned for lots over 25,000 sq. ft
# 
#   **INDUS**: Proportion of non-retail business acres per town
# 
#   **CHAS**: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
# 
#   **NOX**: Nitric oxide concentration (parts per 10 million)
# 
#   **RM**: Average number of rooms per dwelling
# 
#   **AGE**: Proportion of owner-occupied units built prior to 1940
# 
#   **DIS**: Weighted distances to five Boston employment centers
# 
#   **RAD**: Index of accessibility to radial highways
# 
#   **TAX**: Full-value property tax rate per $10,000
# 
#   **B**: 1000(Bk - 0.63)², where Bk is the proportion of [people of African American descent] by town
# 
#   **LSTAT**: Percentage of lower status of the population
# 
#   **MEDV**: Median value of owner-occupied homes in $1000s
# 
# 
# 

# I**mport the required Libraries**

# In[1]:


import numpy as np
import matplotlib.pyplot as plt 

import pandas as pd  
import seaborn as sns 

get_ipython().run_line_magic('matplotlib', 'inline')


# **Load the Boston Housing DataSet from scikit-learn**

# In[2]:


from sklearn.datasets import load_boston

boston_dataset = load_boston()

# boston_dataset is a dictionary
# let's check what it contains
boston_dataset.keys()


# **Load the data into pandas dataframe**

# In[3]:


boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
boston.head()


# **The target values is missing from the data. Create a new column of target values and add it to dataframe**

# In[4]:


boston['MEDV'] = boston_dataset.target


# **Data preprocessing**

# In[5]:


# check for missing values in all the columns
boston.isnull().sum()


# **Data Visualization**

# In[6]:


# set the size of the figure
sns.set(rc={'figure.figsize':(10,8.27)})

# plot a histogram showing the distribution of the target values
sns.distplot(boston['MEDV'], bins=50)
plt.show()


# **Correlation matrix**

# In[7]:


# compute the pair wise correlation for all columns  
correlation_matrix = boston.corr().round(2)


# In[8]:


# use the heatmap function from seaborn to plot the correlation matrix
# annot = True to print the values inside the square
sns.heatmap(data=correlation_matrix, annot=True)


# **Observations**
# 
# 
# 
# 
# *   From the above coorelation plot we can see that **MEDV** is strongly correlated to **LSTAT**, **RM**
# 
# 
# 

# In[9]:


plt.figure(figsize=(20, 5))

features = ['LSTAT', 'RM']
target = boston['MEDV']

for i, col in enumerate(features):
    plt.subplot(1, len(features) , i+1)
    x = boston[col]
    y = target
    plt.scatter(x, y, marker='o')
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel('MEDV')


# **Prepare the data for training**

# In[10]:


X = pd.DataFrame(np.c_[boston['LSTAT'], boston['RM']], columns = ['LSTAT','RM'])
Y = boston['MEDV']
print("X",X)


# **Split the data into training and testing sets**

# In[11]:


from sklearn.model_selection import train_test_split

# splits the training and test data set in 80% : 20%
# assign random_state to any value.This ensures consistency.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# **Train the model using sklearn LinearRegression**

# In[12]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

lin_model = LinearRegression()
lin_model.fit(X_train, Y_train)


# In[13]:


# model evaluation for training set

y_train_predict = lin_model.predict(X_train)
rmse = (np.sqrt(mean_squared_error(Y_train, y_train_predict)))
r2 = r2_score(Y_train, y_train_predict)

print("The model performance for training set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")

# model evaluation for testing set

y_test_predict = lin_model.predict(X_test)
# root mean square error of the model
rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))

# r-squared score of the model
r2 = r2_score(Y_test, y_test_predict)

print("The model performance for testing set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))


# In[14]:


# plotting the y_test vs y_pred
# ideally should have been a straight line
plt.scatter(Y_test, y_test_predict, marker='o')
plt.xlabel('Y_test')
plt.ylabel('y_test_predict')
plt.show()


# In[15]:


Check = np.c_[Y_test.values,y_test_predict]
print(Check)


# In[ ]:




