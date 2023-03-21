#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import numpy as np


# In[3]:


df = pd.read_csv('https://raw.githubusercontent.com/edyoda/data-science-complete-tutorial/master/Data/house_rental_data.csv.txt')
df


# In[165]:


#Use pandas to get some insights into the data
df.info()


# In[166]:


df.describe()


# In[167]:


df.columns


# In[168]:


df.dtypes


# In[169]:


df.shape


# In[170]:


df.size


# In[171]:


#Show some interesting visualization of the data
px.scatter(x=df['Sqft'],y = df['Price'])


# In[173]:


figure = px.scatter_3d(df,x  = 'Sqft',y = 'Price',z = 'TotalFloor')


# In[174]:


figure


# In[175]:


fig = px.histogram(df,x ='Sqft',y = 'Price')
fig


# In[21]:


#Manage data for training & testing 


# In[4]:


X=df[['Sqft','Floor','TotalFloor']]
Y=df['Price']


# In[5]:


X.head()


# In[6]:


Y.head()


# In[7]:


x_train,x_test,Y_train,Y_test = train_test_split(X,Y,random_state = 32)


# In[8]:


model = KNeighborsClassifier()


# In[9]:


model.fit(x_train,Y_train) 


# In[10]:


model.score(x_test,Y_test)


# In[11]:


x_test


# In[12]:


model.predict([[768.528,3,4]])


# In[160]:


#Finding a better value of k 


# In[13]:



df1 = df[['Sqft', 'Price']]
df1


# In[14]:


list1=[]

for i in range(1,8):
   model= KMeans(n_clusters= i, init='k-means++', random_state=0)
   model.fit(df)
   list1.append(model.inertia_)


# In[15]:


list1


# In[189]:


plt.plot(range(1,8), list1)
plt.show()


# In[16]:


final_model = KMeans(n_clusters = 3,init = "k-means++",random_state = 0)#my model of clusters have 5
final_model.fit(df1)
    
    
df1['TotalFloor'] = final_model.labels_


# In[17]:


df1


# In[18]:


px.scatter(df1,x ="Sqft",y = "Price",color = "TotalFloor")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




