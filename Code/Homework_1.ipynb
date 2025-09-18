{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "38dc5873-33b2-45a4-b086-0a3b6fc47c3c",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Kaylena Mann\n",
    "#ADEC7430\n",
    "#Homework_1\n",
    "\n",
    "#importing libraries\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import statsmodels.api as sm\n",
    "import statsmodels.formula.api as smf\n",
    "import sklearn\n",
    "from sklearn.model_selection import train_test_split"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "84add4d6-34bb-488c-ba90-9f365585a684",
   "metadata": {},
   "outputs": [],
   "source": [
    "# STEP 1. Create a dataset with 10,000 rows and 4 random variables \n",
    "\n",
    "n = 10000\n",
    "np.random.seed(39) #generating seed for reproducability \n",
    "\n",
    "#random normal variables with mean of 0 and SD of 1 (default)\n",
    "x1 = np.random.normal(0, 1, n)\n",
    "x2 = np.random.normal(0, 1, n)\n",
    "#random uniform variables with min of 0 and max of 1 (default)\n",
    "x3 = np.random.uniform(0, 1, n)\n",
    "x4 = np.random.uniform(0, 1, n)\n",
    "x2_sq = x2**2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "0c448c39-e64b-44fe-86f5-c780c2bcf983",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "         x1        x2        x3        x4     x2_sq         y\n",
      "0  1.404840 -0.412341  0.701333  0.909778  0.170025  1.043773\n",
      "1  0.221121 -1.026576  0.751686  0.521624  1.053858 -1.611487\n",
      "2 -0.145327 -1.585903  0.074105  0.851362  2.515089 -1.509581\n",
      "3  0.123199 -0.358885  0.444786  0.973354  0.128799 -1.516847\n",
      "4  0.606027 -1.320395  0.856817  0.499827  1.743443 -0.639308\n"
     ]
    }
   ],
   "source": [
    "# STEP 2. Add another variable (\"Y\") as the linear combination, with some coefficients and noise.\n",
    "\n",
    "# The coefficients were based on this study: https://digitalcommons.coastal.edu/cgi/viewcontent.cgi?article=1220&context=etd!\n",
    "b0 = -1.32 # intercept\n",
    "b1 = 0.23  # Covid Cases\n",
    "b2 = -0.01 # Enrollment Intensity\n",
    "b3 = 0.74 # Campus Setting\n",
    "b4 = 0.13 # Political affiliation \n",
    "sq_term = -0.3 #squared term\n",
    "\n",
    "#adding error\n",
    "error = np.random.normal(0, 0.5, n)   \n",
    "\n",
    "#generating the equation for y\n",
    "y = (b0 + b1*x1 + b2*x2 + b3*x3 + b4*x4 + sq_term*x2_sq +\n",
    "     error)\n",
    "#creating the data_frame\n",
    "data = pd.DataFrame({'x1': x1,'x2': x2,'x3': x3,'x4': x4, 'x2_sq': x2_sq, 'y': y})\n",
    "print(data.head())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "9e67b5ae-b9fd-4b36-b593-350f02957514",
   "metadata": {},
   "outputs": [],
   "source": [
    "# STEP 3. Split the dataset into 70% for training and 30% for testing\n",
    "\n",
    "train, test = train_test_split(data, \n",
    "                                test_size = .30,\n",
    "                                random_state = 39)\n",
    "\n",
    "# Adding variables to training and testing\n",
    "X_train = train[['x1','x2','x3','x4','x2_sq']]\n",
    "y_train = train['y']\n",
    "X_test = test[['x1','x2','x3','x4','x2_sq']]\n",
    "y_test = test['y']\n",
    "\n",
    "# adding intercept\n",
    "X_train_sm = sm.add_constant(X_train)\n",
    "X_test_sm = sm.add_constant(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "c2374563-9b8f-44fa-87da-391b468fa3fd",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "                            OLS Regression Results                            \n",
      "==============================================================================\n",
      "Dep. Variable:                      y   R-squared:                       0.529\n",
      "Model:                            OLS   Adj. R-squared:                  0.529\n",
      "Method:                 Least Squares   F-statistic:                     1573.\n",
      "Date:                Wed, 17 Sep 2025   Prob (F-statistic):               0.00\n",
      "Time:                        22:27:53   Log-Likelihood:                -5015.8\n",
      "No. Observations:                7000   AIC:                         1.004e+04\n",
      "Df Residuals:                    6994   BIC:                         1.008e+04\n",
      "Df Model:                           5                                         \n",
      "Covariance Type:            nonrobust                                         \n",
      "==============================================================================\n",
      "                 coef    std err          t      P>|t|      [0.025      0.975]\n",
      "------------------------------------------------------------------------------\n",
      "const         -1.3462      0.016    -82.353      0.000      -1.378      -1.314\n",
      "x1             0.2270      0.006     38.714      0.000       0.216       0.239\n",
      "x2            -0.0049      0.006     -0.835      0.404      -0.017       0.007\n",
      "x3             0.7457      0.021     36.372      0.000       0.706       0.786\n",
      "x4             0.1685      0.020      8.244      0.000       0.128       0.209\n",
      "x2_sq         -0.2996      0.004    -70.543      0.000      -0.308      -0.291\n",
      "==============================================================================\n",
      "Omnibus:                        0.634   Durbin-Watson:                   1.978\n",
      "Prob(Omnibus):                  0.728   Jarque-Bera (JB):                0.646\n",
      "Skew:                          -0.023   Prob(JB):                        0.724\n",
      "Kurtosis:                       2.990   Cond. No.                         8.31\n",
      "==============================================================================\n",
      "\n",
      "Notes:\n",
      "[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.\n",
      "Training MSE: 0.2454207232521446\n",
      "Test MSE: 0.25195450541118436\n"
     ]
    }
   ],
   "source": [
    "# STEP 4. Estimate the linear regression coefficients using OLS, compute the MSE on both datasets\n",
    "\n",
    "model = sm.OLS(y_train, X_train_sm)\n",
    "results = model.fit()\n",
    "\n",
    "#Calculating the MSE\n",
    "yhat_test = results.predict(X_test_sm) #Predicted outcome on the test data\n",
    "yhat_train = results.predict(X_train_sm) #Predicted outcome on the train data\n",
    "MSE_test = np.mean((y_test - yhat_test)**2) #subtracting the predicted values from actual values and squaring for test MSE\n",
    "MSE_train = np.mean((y_train - yhat_train)**2) #subtracting the predicted values from actual values and squaring for training MSE\n",
    "\n",
    "# printing all results\n",
    "print(results.summary()) \n",
    "print(\"Training MSE:\", MSE_train) \n",
    "print(\"Test MSE:\", MSE_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "79a0d48c-dec2-408c-8a80-5aca5a47fb12",
   "metadata": {},
   "outputs": [],
   "source": [
    "# STEP 5. Use bootstrapping to create 10 other samples from the data I created\n",
    "\n",
    "B = 10\n",
    "coef_rows = []\n",
    "\n",
    "#For loop using the full original data with replacement, running OLS, and then appending the parameters into the coef_rows\n",
    "for b in range(B):\n",
    "    bootstrap = data.sample(10000, replace=True, random_state=33+b)\n",
    "    model = smf.ols('y ~ x1 + x2 + x3 + x4 + x2_sq', data = bootstrap).fit()\n",
    "    coef_rows.append(model.params.reindex(['Intercept','x1','x2','x3','x4','x2_sq']))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "09494780-ec03-4e67-a0ab-322876516442",
   "metadata": {},
   "outputs": [],
   "source": [
    "# STEP 6. Estimate the linear regression coefficients using OLS for each of the 10 bootstrap samples\n",
    "\n",
    "bootstrap_sample_coefs = pd.DataFrame(coef_rows)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "e08981ab-4595-4415-82d9-bfb0f69e0893",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "               mean       std\n",
      "Intercept -1.327371  0.010017\n",
      "x1         0.232018  0.002619\n",
      "x2        -0.004880  0.003324\n",
      "x3         0.729014  0.011309\n",
      "x4         0.165079  0.011594\n",
      "x2_sq     -0.304516  0.004015\n",
      "   Intercept        x1        x2        x3        x4     x2_sq\n",
      "0  -1.310553  0.233044 -0.006173  0.713527  0.162566 -0.303298\n",
      "1  -1.314000  0.232813 -0.004744  0.708460  0.168787 -0.307747\n",
      "2  -1.339257  0.235744 -0.011322  0.734219  0.176820 -0.303338\n",
      "3  -1.324174  0.234400 -0.008626  0.723928  0.168417 -0.312357\n",
      "4  -1.332990  0.228568 -0.003446  0.741371  0.149775 -0.298286\n",
      "5  -1.319219  0.228544 -0.005636  0.726845  0.149709 -0.299715\n",
      "6  -1.335273  0.229142 -0.001836  0.740411  0.154047 -0.307150\n",
      "7  -1.330690  0.231361 -0.000029  0.727077  0.165447 -0.304431\n",
      "8  -1.329514  0.234802 -0.004699  0.739621  0.169416 -0.303558\n",
      "9  -1.338043  0.231763 -0.002284  0.734685  0.185805 -0.305282\n"
     ]
    }
   ],
   "source": [
    "# STEP 7. Compute mean and standard deviation for each parameter\n",
    "\n",
    "bootstrap_summary = pd.DataFrame({'mean': bootstrap_sample_coefs.mean(), 'std': bootstrap_sample_coefs.std(ddof=1)})\n",
    "print(bootstrap_summary)\n",
    "print(bootstrap_sample_coefs)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "a334d4fa-6804-4701-b98b-7d447581bf69",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "               mean       std\n",
      "Intercept -1.327371  0.010017\n",
      "x1         0.232018  0.002619\n",
      "x2        -0.004880  0.003324\n",
      "x3         0.729014  0.011309\n",
      "x4         0.165079  0.011594\n",
      "x2_sq     -0.304516  0.004015\n",
      "                            OLS Regression Results                            \n",
      "==============================================================================\n",
      "Dep. Variable:                      y   R-squared:                       0.529\n",
      "Model:                            OLS   Adj. R-squared:                  0.529\n",
      "Method:                 Least Squares   F-statistic:                     1573.\n",
      "Date:                Wed, 17 Sep 2025   Prob (F-statistic):               0.00\n",
      "Time:                        22:28:02   Log-Likelihood:                -5015.8\n",
      "No. Observations:                7000   AIC:                         1.004e+04\n",
      "Df Residuals:                    6994   BIC:                         1.008e+04\n",
      "Df Model:                           5                                         \n",
      "Covariance Type:            nonrobust                                         \n",
      "==============================================================================\n",
      "                 coef    std err          t      P>|t|      [0.025      0.975]\n",
      "------------------------------------------------------------------------------\n",
      "const         -1.3462      0.016    -82.353      0.000      -1.378      -1.314\n",
      "x1             0.2270      0.006     38.714      0.000       0.216       0.239\n",
      "x2            -0.0049      0.006     -0.835      0.404      -0.017       0.007\n",
      "x3             0.7457      0.021     36.372      0.000       0.706       0.786\n",
      "x4             0.1685      0.020      8.244      0.000       0.128       0.209\n",
      "x2_sq         -0.2996      0.004    -70.543      0.000      -0.308      -0.291\n",
      "==============================================================================\n",
      "Omnibus:                        0.634   Durbin-Watson:                   1.978\n",
      "Prob(Omnibus):                  0.728   Jarque-Bera (JB):                0.646\n",
      "Skew:                          -0.023   Prob(JB):                        0.724\n",
      "Kurtosis:                       2.990   Cond. No.                         8.31\n",
      "==============================================================================\n",
      "\n",
      "Notes:\n",
      "[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.\n",
      "CONCLUSION\n",
      "The bootstrap results very closely match the estimates from our original OLS validation training model. On average, the bootstrap means were off by about 0.01-0.02. The standard deviations from bootstrapping are small and approximate the standard error values from the original regression, indicating high precision in coefficients and good reliability in our  standard errors. Overall, this makes us more confident in the original results provided from validation and indicates that our model is less susceptible to sampling variability, performing consistently across resamples. This makes sense, because there was only a tiny amount of a noise added when simulating the data. Not only did our model generalize well with new test data without evidence of overfitting, but it also demonstrated stability across resampling. \n"
     ]
    }
   ],
   "source": [
    "# STEP 8. What can you say about the coefficients in STEP #4 when looking at STEP #7?\n",
    "print(bootstrap_summary)\n",
    "print(results.summary()) \n",
    "\n",
    "print('''CONCLUSION\n",
    "The bootstrap results very closely match the estimates from our original OLS validation training model. On average, the bootstrap means were off by about 0.01-0.02. The standard deviations from bootstrapping are small and approximate the standard error values from the original regression, indicating high precision in coefficients and good reliability in our  standard errors. Overall, this makes us more confident in the original results provided from validation and indicates that our model is less susceptible to sampling variability, performing consistently across resamples. This makes sense, because there was only a tiny amount of a noise added when simulating the data. Not only did our model generalize well with new test data without evidence of overfitting, but it also demonstrated stability across resampling. ''')"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
