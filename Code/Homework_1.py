#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Kaylena Mann
#ADEC7430
#Homework_1

#importing libraries
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import sklearn
from sklearn.model_selection import train_test_split


# In[2]:


# STEP 1. Create a dataset with 10,000 rows and 4 random variables 

n = 10000
np.random.seed(39) #generating seed for reproducability 

#random normal variables with mean of 0 and SD of 1 (default)
x1 = np.random.normal(0, 1, n)
x2 = np.random.normal(0, 1, n)
#random uniform variables with min of 0 and max of 1 (default)
x3 = np.random.uniform(0, 1, n)
x4 = np.random.uniform(0, 1, n)
x2_sq = x2**2


# In[3]:


# STEP 2. Add another variable ("Y") as the linear combination, with some coefficients and noise.

# The coefficients were based on this study: https://digitalcommons.coastal.edu/cgi/viewcontent.cgi?article=1220&context=etd!
b0 = -1.32 # intercept
b1 = 0.23  # Covid Cases
b2 = -0.01 # Enrollment Intensity
b3 = 0.74 # Campus Setting
b4 = 0.13 # Political affiliation 
sq_term = -0.3 #squared term

#adding error
error = np.random.normal(0, 0.5, n)   

#generating the equation for y
y = (b0 + b1*x1 + b2*x2 + b3*x3 + b4*x4 + sq_term*x2_sq +
     error)
#creating the data_frame
data = pd.DataFrame({'x1': x1,'x2': x2,'x3': x3,'x4': x4, 'x2_sq': x2_sq, 'y': y})
print(data.head())


# In[4]:


# STEP 3. Split the dataset into 70% for training and 30% for testing

train, test = train_test_split(data, 
                                test_size = .30,
                                random_state = 39)

# Adding variables to training and testing
X_train = train[['x1','x2','x3','x4','x2_sq']]
y_train = train['y']
X_test = test[['x1','x2','x3','x4','x2_sq']]
y_test = test['y']

# adding intercept
X_train_sm = sm.add_constant(X_train)
X_test_sm = sm.add_constant(X_test)


# In[5]:


# STEP 4. Estimate the linear regression coefficients using OLS, compute the MSE on both datasets

model = sm.OLS(y_train, X_train_sm)
results = model.fit()

#Calculating the MSE
yhat_test = results.predict(X_test_sm) #Predicted outcome on the test data
yhat_train = results.predict(X_train_sm) #Predicted outcome on the train data
MSE_test = np.mean((y_test - yhat_test)**2) #subtracting the predicted values from actual values and squaring for test MSE
MSE_train = np.mean((y_train - yhat_train)**2) #subtracting the predicted values from actual values and squaring for training MSE

# printing all results
print(results.summary()) 
print("Training MSE:", MSE_train) 
print("Test MSE:", MSE_test)


# In[6]:


# STEP 5. Use bootstrapping to create 10 other samples from the data I created

B = 10
coef_rows = []

#For loop using the full original data with replacement, running OLS, and then appending the parameters into the coef_rows
for b in range(B):
    bootstrap = data.sample(10000, replace=True, random_state=33+b)
    model = smf.ols('y ~ x1 + x2 + x3 + x4 + x2_sq', data = bootstrap).fit()
    coef_rows.append(model.params.reindex(['Intercept','x1','x2','x3','x4','x2_sq']))


# In[7]:


# STEP 6. Estimate the linear regression coefficients using OLS for each of the 10 bootstrap samples

bootstrap_sample_coefs = pd.DataFrame(coef_rows)


# In[8]:


# STEP 7. Compute mean and standard deviation for each parameter

bootstrap_summary = pd.DataFrame({'mean': bootstrap_sample_coefs.mean(), 'std': bootstrap_sample_coefs.std(ddof=1)})
print(bootstrap_summary)
print(bootstrap_sample_coefs)


# In[9]:


# STEP 8. What can you say about the coefficients in STEP #4 when looking at STEP #7?
print(bootstrap_summary)
print(results.summary()) 

print('''CONCLUSION
The bootstrap results very closely match the estimates from our original OLS validation training model. On average, the bootstrap means were off by about 0.01-0.02. The standard deviations from bootstrapping are small and approximate the standard error values from the original regression, indicating high precision in coefficients and good reliability in our  standard errors. Overall, this makes us more confident in the original results provided from validation and indicates that our model is less susceptible to sampling variability, performing consistently across resamples. This makes sense, because there was only a tiny amount of a noise added when simulating the data. Not only did our model generalize well with new test data without evidence of overfitting, but it also demonstrated stability across resampling. ''')

