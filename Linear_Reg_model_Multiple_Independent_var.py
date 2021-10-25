#!/usr/bin/env python
# coding: utf-8

# 1.importing lib and dataset.
# 2.checking for multicollinearity and removing it.
# 3.scaling the dataset.
# 4.creating test and train partition.
# 5.implementing linear regression model using scikit    learn.
# 6.generating prediction over the testset.
# 7.evaluating the model.
# 8.plotting the residual.
# 9.verifying the assumption of the lineat reg.
# 10.visualizing the coef. to interpret the model result.

# # Importing Libraries
# 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


data=pd.read_csv('Transformed_Housing_Data2.csv')
data.head()


# # Scaling the dataset
# 

# In[4]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()

Y=data['Sale_Price']
X=scaler.fit_transform(data.drop(columns=['Sale_Price']))
X=pd.DataFrame(data=X,columns=data.drop(columns=['Sale_Price']).columns)
X.head()


# # Checking and removing Multicollinearity.

# In[5]:


X.corr()


# In[7]:


## pair of independent variables with correlation greater than 0.5

k=X.corr()
z=[[str(i),str(j)] for i in k.columns for j in k.columns if (k.loc[i,j]>abs(0.5))&(i!=j)]
z,len(z)


# # Calculating VIF
# 

# In[25]:


from statsmodels.stats.outliers_influence import variance_inflation_factor as vifa

vif_data=X 

##calculating VIF for every column
VIF= pd.Series([vifa(vif_data.values,i)for i in range(vif_data.shape[1])],index=vif_data.columns)
VIF,len(VIF)


# In[ ]:





# In[21]:


def MC_remover(data):
    vif=pd.Series([vifa(data.values,i) for i in range(data.shape[1])],index=data.columns)
    if vif.max()>5:
        print(vif[vif==vif.max()].index[0],' has been removed')
        data=data.drop(columns=[vif[vif==vif.max()].index[0]])
        return data
    else:
        print('No collinearity present anymore')
        return data


# In[22]:


for i in range(7):
    vif_data=MC_remover(vif_data)
    
vif_data.head()


# # Remaining Columns
# 

# In[24]:


#calculating the vif for the remaining columns.
VIF=pd.Series([vifa(vif_data.values,i) for i in range(vif_data.shape[1])],index=vif_data.columns)
VIF, len(vif_data.columns)


# # Train/Test  set

# In[27]:


X=vif_data
y=data['Sale_Price']


# In[29]:


from sklearn.model_selection import train_test_split as tts
x_train,x_test,y_train,y_test=tts(X,Y,test_size=0.3,random_state=101)

x_train.shape,x_test.shape,y_train.shape,y_test.shape


# # Linear Regression

# In[30]:


from sklearn.linear_model import LinearRegression
lr=LinearRegression(normalize=True)
lr.fit(x_train,y_train)


# In[32]:


lr.coef_


# # Generate Prediction Over The Test Set.

# In[34]:


# predictions=lr.predict(x_test)
lr.score(x_test,y_test)


# # Residual plot

# In[35]:


residuals =predictions -y_test
residual_table=pd.DataFrame({
    'residuals':residuals,
    'predictions':predictions
})
residua_table=residual_table.sort_values(by='predictions')


# In[36]:


z=[i for i in range(int(residual_table['predictions'].max()))]
k=[0 for i in range(int(residual_table['predictions'].max()))]


# In[40]:


plt.figure(dpi=130,figsize=(17,7))

plt.scatter(residual_table['predictions'],residual_table['residuals'],color='red',s=2)
plt.plot(z,k,color='green',linewidth=3,label='regression line')
plt.ylim(-800000,800000)
plt.xlabel('fitted points (ordered by predictions)')
plt.ylabel('residuals')
plt.legend()
plt.show()


# In[41]:


plt.figure(dpi=100, figsize=(10,7))
plt.hist(residual_table['residuals'],color='red',bins=200)
plt.xlabel('residuals')
plt.ylabel('frequency')
plt.title('distribution of residuals')
plt.show()


# # Interpreting our result.
# 

# # Model Coefficients

# In[42]:


coefficients_table=pd.DataFrame({
    'column':x_train.columns,
    'coefficients':lr.coef_
})

coefficient_table=coefficients_table.sort_values(by='coefficients')


# In[43]:


plt.figure(figsize=(8,6),dpi=120)
x=coefficient_table['column']
y=coefficient_table['coefficients']
plt.barh(x,y)
plt.xlabel('coefficients')
plt.ylabel('variables')
plt.title('normalized coefficient plot')
plt.show()

