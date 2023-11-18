#!/usr/bin/env python
# coding: utf-8

# ## Part IIB
# 
# ### Including necessary library files needed for execution of this task

# In[1]:


import pandas as pd
import pulp
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt 
import math
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
from keras.utils import np_utils
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
data = pd.read_csv("C://Users//44776//Desktop//MLF_GP1_CreditScore.csv")


# ###  Exploratory Data Analysis

# In[2]:


print(f"The dataset has {data.shape[0]} rows and {data.shape[1]} columns")


# In[3]:


data.describe()


# ### Printing type of variables in the dataset

# In[4]:


data.dtypes


# ### Checking Null values 

# In[5]:


# Finding Missing Values in the dataset
print("Column\t\tNumber of NaN values")
print("-"*40)
for col in data.columns:
    print("{:<20}{:<10}".format(col,     data[col].isna().sum()))


# ### Splitting dataset into numerical and categorical type and printing columns accordingly

# In[6]:


num_features = data.select_dtypes(exclude="object").columns
cat_features = data.select_dtypes(include="object").columns


# In[7]:


num_features


# In[8]:


cat_features


# ###  Visualizing the data
# 
# ###  Scatter Plot between Sales/Revenues  vs  Gross Margin

# In[9]:


# Create a scatter plot of Sales/Revenues vs Gross Margin
plt.scatter(data['Sales/Revenues'], data['Gross Margin'])
plt.xlabel('Sales/Revenues')
plt.ylabel('Gross Margin')
plt.title('Sales/Revenues vs Gross Margin')
plt.show()


# ### Histogram showing the frequency of values in column "Total MV"

# In[10]:


sns.histplot(data=data, x="Total MV", bins=5,color="green")


# ###  Analysis of Target columns  "Rating" and "InvGrd"

# In[11]:


print("Categories in'Rating' variable:",end=" " )
print(data['Rating'].unique())


# In[12]:


print(len(data['Rating'].unique()))


# In[13]:


# Plot the Rating and InvGrd columns

fig, ax = plt.subplots()
ax.scatter(data['Rating'], data['InvGrd'])
ax.set_xlabel('Rating')
ax.set_ylabel('InvGrd')
ax.set_title('Credit Ratings vs. Investment Grade')
plt.show()


# In[14]:


sns.countplot(x=data["InvGrd"])


# ###  From the above plot it clearly states the dataset is unbalanced.

# In[15]:


sns.countplot(x=data["Rating"])


# ### Training and Testing Split
# 
# - Split the dataset into a training set and a test set using train_test_split from the scikit-learn library.
# - This code loads the dataset into a pandas DataFrame, drops the target variables "Rating" and "InvGrd", and splits the dataset into training and test sets with an 80:20 ratio.
# - The independent features (X)  has 1700 rows and 26 columns.

# In[16]:


# Split the dataset into a training set and a test set
X = data.drop(['InvGrd','Rating'], axis=1)
y = data['InvGrd']

X_train,X_test,y_train,y_test= train_test_split(X,y, test_size=0.2, random_state=42)


# In[17]:


X.shape


# In[18]:


y.shape


# ###  Data Balancing
# 
# - This code is performing oversampling using RandomOverSampler from the imblearn library. The dataset is being balanced by generating synthetic samples for the minority class (InvGrd '1') so that it has the same number of samples as the majority class (InvGrd '0').
# 
# - Before oversampling, the code is printing the counts of InvGrd '1' and '0' in the original training set. Then, RandomOverSampler is instantiated and fit to the training set to generate synthetic samples. The code is then printing the shape of the resampled training data, as well as the counts of InvGrd '1' and '0' after oversampling.
# 
# - By oversampling the minority class, we can balance the dataset and improve the performance of machine learning models trained on this data.

# In[19]:


print("Before OverSampling, counts of InvGrd '1': {}".format(sum(y_train == 1)))
print("Before OverSampling, counts of InvGrd '0': {} \n".format(sum(y_train == 0)))
  
import imblearn
from imblearn.over_sampling import RandomOverSampler


ros = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

  
print('After OverSampling, the shape of train_X: {}'.format(X_train_resampled.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_resampled.shape))
  
print("After OverSampling, counts of InvGrd '1': {}".format(sum(y_train_resampled == 1)))
print("After OverSampling, counts of InvGrd '0': {}".format(sum(y_train_resampled == 0)))


# ###  Linear Regression with Ridge and Lasso
# 
# -  The cross_val_score function  performs cross-validation on the training data (X_train_resampled and y_train_resampled). It calculates the negative mean squared error (scoring="neg_mean_squared_error") and sets the number of folds for cross-validation to 5 (cv=5).
# - The results of the cross-validation are stored in the mse variable. Then, the code calculates the mean of the MSE scores (mean_mse) using NumPy's mean function.
# - Overall, this code is used to evaluate the performance of the linear regression model using cross-validation and mean squared error as the evaluation metric.
# - The models are trained using resampled dataset.
# - The fitted model is tested using original test dataset.
# - The prediction and the evaluation metrics are shown.
# - Overall the Ridge and Lasso tuning improved the performance of the model.

# In[20]:


#Linear Regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
lin_model=LinearRegression()
mse=cross_val_score(lin_model,X_train_resampled,y_train_resampled,scoring="neg_mean_squared_error",cv=5)
print(mse)
mean_mse=np.mean(mse)
print(mean_mse)


# In[21]:


lin_model=LinearRegression().fit(X_train_resampled,y_train_resampled)


# In[22]:


# Evaluate the performance of the models
y_pred_linear = lin_model.predict(X_test)
y_pred_linear[y_pred_linear >= 0.5] = 1
y_pred_linear[y_pred_linear < 0.5] = 0
recall_linear = recall_score(y_test, y_pred_linear)
f1_linear = f1_score(y_test, y_pred_linear)


# In[23]:


print("Linear Regression Results:")
print('----------------------------------')
print("Recall:", recall_linear)
print("F1 score:", f1_linear)


# ###  Ridge Regularization
# 
# ### The implementation of a linear regression approach with Ridge (L1)  regularization to predict whether a firm is in an investment grade or not is done with following steps:
# 
# - Train a linear regression model with Ridge regularization using the resampled  training set. The  Ridge function is taken from the scikit-learn library to implement Ridge regularization.
# - The parameters chosen to smooth the model is 'alpha'.The observation of this model  shows ,higher the value of alpha better the negative mean squared error.
# - The performance of the models are evaluated using the original test set. The metrics chosen are  recall, and F1 score to evaluate the models performance.
# - The selected best parameter and best score are printed.
# 

# In[24]:


#Ridge
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
ridge=Ridge()
params={'alpha':[0.1, 1, 10, 100,200,500,1000]}                    
ridge_regressor=GridSearchCV(ridge,params,scoring="neg_mean_squared_error",cv=5)
ridge_regressor.fit(X_train_resampled,y_train_resampled)


# In[25]:


print(ridge_regressor.best_params_)
print(ridge_regressor.best_score_)


# In[26]:


y_pred_ridge = ridge_regressor.predict(X_test)
y_pred_ridge[y_pred_ridge >= 0.5] = 1
y_pred_ridge[y_pred_ridge < 0.5] = 0
recall_ridge = recall_score(y_test, y_pred_ridge)
f1_ridge = f1_score(y_test, y_pred_ridge)

print("Ridge regularization:")
print('----------------------------------')
print("Recall:", recall_ridge)
print("F1 score:", f1_ridge)


# ###  Lasso Regularization
#  
# ### The implementation of  a linear regression approach with  Lasso (L2) regularization to predict whether a firm is in an investment grade or not is done with following steps:
# 
# - Train a linear regression model with Lasso regularization using the resampled training set. The scikit-learn library contains the necessary Lasso function to implement Lasso regularization.
# - The parameters chosen tune the data are 'alpha','max_iter' and 'tol'.
# - The best values are obtained as 'alpha': 0.01, 'max_iter': 500, 'tol': 0.01 in this model.
# - The performance of the models are evaluated using the original test set. The metrics chosen are  recall, and F1 score to evaluate the models' performance.
# 
# 

# In[27]:


from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
lasso=Lasso()
params={'alpha':[0.01, 0.1, 1, 10, 100],
                'max_iter': [100, 500, 1000],
        'tol': [1e-2, 1e-3, 1e-4]}
lasso_regressor=GridSearchCV(lasso,params,scoring="neg_mean_squared_error",cv=5)
lasso_regressor.fit(X_train_resampled,y_train_resampled)


# In[28]:


print(lasso_regressor.best_params_)
print(lasso_regressor.best_score_)


# In[29]:


y_pred_lasso = lasso_regressor.predict(X_test)
y_pred_lasso[y_pred_lasso >= 0.5] = 1
y_pred_lasso[y_pred_lasso < 0.5] = 0
recall_lasso = recall_score(y_test, y_pred_lasso)
f1_lasso = f1_score(y_test, y_pred_lasso)


# In[30]:


print("Lasso regularization:")
print('----------------------------------')
print("Recall:", recall_lasso)
print("F1 score:", f1_lasso)


# ###  Evaluation of Linear,Ridge and Lasso
# 
# - The three models are evaluated with the prediction rate and the results are shown using Classification Report and Confusion matrix .
# - The confusion matrix tells us how well the model is performing in terms of correctly classifying positive and negative instances.
# 
# - The confusion matrix is of size 2 x 2, as there are two classes: investment grade (positive class) and non-investment grade (negative class).
# 
# - The rows correspond to the true class labels, while the columns correspond to the predicted class labels.
# 
# - Each cell of the matrix represents the number of instances that belong to a particular true class and were classified as a particular predicted class.
# 
# - In this particular confusion matrix of Linear Model, the values are as follows:
# 
#      - 42 true negatives (TN): The model correctly predicted 42 instances as negative.
#      - 42 false positives (FP): The model incorrectly predicted 42 instances as positive.
#      - 72 false negatives (FN): The model incorrectly predicted 72 instances as negative.
#      - 184 true positives (TP): The model correctly predicted 184 instances as positive.
#      
# - In the confusion matrix of Ridge Linear Model, the values are as follows:
# 
#      - 32 true negatives (TN): The model correctly predicted 32 instances as negative.
#      - 52 false positives (FP): The model incorrectly predicted 52 instances as positive.
#      - 46 false negatives (FN): The model incorrectly predicted 46 instances as negative.
#      - 210 true positives (TP): The model correctly predicted 210 instances as positive.
# 
# - In the confusion matrix of Lasso Linear Model, the values are as follows:
# 
#     - 29 true negatives (TN): The model correctly predicted 29 instances as negative.
#     - 55 false positives (FP): The model incorrectly predicted 55instances as positive.
#     - 48 false negatives (FN): The model incorrectly predicted 48 instances as negative.
#     - 208 true positives (TP): The model correctly predicted 208 instances as positive.
# 

# In[31]:


from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(y_test, y_pred_linear)
cm2 = confusion_matrix(y_test, y_pred_ridge)
cm3 = confusion_matrix(y_test, y_pred_lasso)

from sklearn.metrics import classification_report, confusion_matrix

print("----------------CLASSIFICATION REPORT OF LOGISTIC REGRESSION MODEL--------------------")
print(classification_report(y_test, y_pred_linear))
print("----------------CLASSIFICATION REPORT OF RIDGE LOGISTIC REGRESSION MODEL--------------------")
print(classification_report(y_test, y_pred_ridge))
print("----------------CLASSIFICATION REPORT OF LASSO LOGISTIC REGRESSION MODEL--------------------")
print(classification_report(y_test, y_pred_lasso))

print("----------Confusion Matrix of Linear Model------------------")
print(cm1 ,"\n")
print("----------Confusion Matrix of Ridge Linear Model------------")
print(cm2,"\n")
print("----------Confusion Matrix of Lasso Linear Model------------")
print(cm3)


# ### Graphical View of Linear,Ridge and Lasso Analysis
# 
# - The visualization and the results shows that the two models Linear Ridge and Linear Lasso are similar in prediction.

# In[32]:


# ROC curve for ridge and lasso regression models
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc

fpr_ridge, tpr_ridge, _ = roc_curve(y_test, y_pred_ridge)
fpr_lasso, tpr_lasso, _ = roc_curve(y_test, y_pred_lasso)
roc_auc_ridge = auc(fpr_ridge, tpr_ridge)
roc_auc_lasso = auc(fpr_lasso, tpr_lasso)
plt.plot(fpr_ridge, tpr_ridge, label='ROC curve for ridge (area = %0.2f)' % roc_auc_ridge)
plt.plot(fpr_lasso, tpr_lasso, label='ROC curve for lasso (area = %0.2f)' % roc_auc_lasso)
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve for ridge and lasso regression models')
plt.legend(loc="lower right")
plt.show()


# In[33]:


# Precision-recall curve for ridge and lasso regression models
precision_ridge, recall_ridge, _ = precision_recall_curve(y_test, y_pred_ridge)
precision_lasso, recall_lasso, _ = precision_recall_curve(y_test, y_pred_lasso)
prc_auc_ridge = auc(recall_ridge, precision_ridge)
prc_auc_lasso = auc(recall_lasso, precision_lasso)
plt.plot(recall_ridge, precision_ridge, label='Precision-recall curve for ridge (area = %0.2f)' % prc_auc_ridge)
plt.plot(recall_lasso, precision_lasso, label='Precision-recall curve for lasso (area = %0.2f)' % prc_auc_lasso)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-recall curve for ridge and lasso regression models')
plt.legend(loc="lower right")
plt.show()


# ### Pairplot visualizing the predictions using Linear,Ridge and Lasso

# In[34]:


results_df = pd.DataFrame({'y_test': y_test,
                           'y_pred_linear': y_pred_linear,
                           'y_pred_ridge': y_pred_ridge,
                           'y_pred_lasso': y_pred_lasso})


# In[35]:


sns.pairplot(results_df)


# ###  Model Fitting using Imbalanced  Target column 'InvGrd'
# 
# - From the available y count,in InvGrd column,we have
#       - 1    1287
#       - 0     413   
#              
# - The interpretation of Linear Model  values:
# 
#     - 6 True negatives (TN): The model correctly predicted 6 samples as negative when they were actually negative.
#     - 78 False positives (FP): The model incorrectly predicted 78 samples as positive when they were actually negative.
#     - 1 False negatives (FN): The model incorrectly predicted 1 sample as negative when they were actually positive.
#     - 255 True positives (TP) :The model correctly predicted 255 samples as positive when they were actually positive.
#      
# - In the confusion matrix of Ridge Linear Model, the values are as follows:
# 
#      - 1 true negatives (TN): The model correctly predicted 1 instances as negative.
#      - 83 false positives (FP): The model incorrectly predicted 83 instances as positive.
#      - 1 false negatives (FN): The model incorrectly predicted 1 instances as negative.
#      - 255 true positives (TP): The model correctly predicted 255 instances as positive.
# 
# - In the confusion matrix of Lasso Linear Model, the values are as follows:
# 
#     -  0 true negatives (TN): The model correctly predicted 0 instances as negative.
#     - 84 false positives (FP): The model incorrectly predicted 84 instances as positive.
#     - 0 false negatives (FN): The model incorrectly predicted 0 instances as negative.
#     - 256 true positives (TP): The model correctly predicted 256 instances as positive.
# 
# - From the observed values,it is evident that the model can predict True values more correctly than False values because of the imbalanced dataset.So working with balanced models will be more perfect.
# 

# In[36]:


# Split the dataset into X and y to check if the dataset is balanced or not
X = data.drop(['InvGrd','Rating'], axis=1)
y = data['InvGrd']


y.value_counts()


X_train,X_test,y_train,y_test= train_test_split(X,y, test_size=0.2, random_state=42)


#Linear Regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
lin_model=LinearRegression()
mse=cross_val_score(lin_model,X_train,y_train,scoring="neg_mean_squared_error",cv=5)
print(mse)
mean_mse=np.mean(mse)
print(mean_mse)


lin_model=LinearRegression().fit(X_train,y_train)


# Evaluate the performance of the models
y_pred_linear = lin_model.predict(X_test)
y_pred_linear[y_pred_linear >= 0.5] = 1
y_pred_linear[y_pred_linear < 0.5] = 0
recall_linear = recall_score(y_test, y_pred_linear)
f1_linear = f1_score(y_test, y_pred_linear)


print("Linear Regression Results:")
print('----------------------------------')
print("Recall:", recall_linear)
print("F1 score:", f1_linear)



#Ridge
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
ridge=Ridge()
params={'alpha':[0.1, 1, 10, 100,200,500,1000]}                    
ridge_regressor=GridSearchCV(ridge,params,scoring="neg_mean_squared_error",cv=5)
ridge_regressor.fit(X_train,y_train)


print(ridge_regressor.best_params_)
print(ridge_regressor.best_score_)


y_pred_ridge = ridge_regressor.predict(X_test)
y_pred_ridge[y_pred_ridge >= 0.5] = 1
y_pred_ridge[y_pred_ridge < 0.5] = 0
recall_ridge = recall_score(y_test, y_pred_ridge)
f1_ridge = f1_score(y_test, y_pred_ridge)

print("Ridge regularization:")
print('----------------------------------')
print("Recall:", recall_ridge)
print("F1 score:", f1_ridge)


from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
lasso=Lasso()
params={'alpha':[0.01, 0.1, 1, 10, 100],
                'max_iter': [100, 500, 1000],
        'tol': [1e-2, 1e-3, 1e-4]}
lasso_regressor=GridSearchCV(lasso,params,scoring="neg_mean_squared_error",cv=5)
lasso_regressor.fit(X_train,y_train)


print(lasso_regressor.best_params_)
print(lasso_regressor.best_score_)

y_pred_lasso = lasso_regressor.predict(X_test)
y_pred_lasso[y_pred_lasso >= 0.5] = 1
y_pred_lasso[y_pred_lasso < 0.5] = 0
recall_lasso = recall_score(y_test, y_pred_lasso)
f1_lasso = f1_score(y_test, y_pred_lasso)



print("Lasso regularization:")
print('----------------------------------')
print("Recall:", recall_lasso)
print("F1 score:", f1_lasso)




print("----------------CLASSIFICATION REPORT OF LOGISTIC REGRESSION MODEL--------------------")
print(classification_report(y_test, y_pred_linear))
print("----------------CLASSIFICATION REPORT OF RIDGE LOGISTIC REGRESSION MODEL--------------------")
print(classification_report(y_test, y_pred_ridge))
print("----------------CLASSIFICATION REPORT OF LASSO LOGISTIC REGRESSION MODEL--------------------")
print(classification_report(y_test, y_pred_lasso))




cm1 = confusion_matrix(y_test, y_pred_linear)
cm2 = confusion_matrix(y_test, y_pred_ridge)
cm3 = confusion_matrix(y_test, y_pred_lasso)
print("----------Confusion Matrix of Linear Model------------------")
print(cm1 ,"\n")
print("----------Confusion Matrix of Ridge Linear Model------------")
print(cm2,"\n")
print("----------Confusion Matrix of Lasso Linear Model------------")
print(cm3)


# ## Logistic regression Model
# 
# - Similar to  Linear model,the Logistic Regression models with Ridge and Lasso is evalauated and analysed with the prediction results. 

# In[37]:


from sklearn.linear_model import LogisticRegression

log_model = LogisticRegression()


# In[38]:


from sklearn.model_selection import cross_val_score
mse=cross_val_score(log_model,X_train_resampled,y_train_resampled,scoring="neg_mean_squared_error",cv=5)
print(mse)
mean_mse=np.mean(mse)
print(mean_mse)


# In[39]:


log_model.fit(X_train_resampled, y_train_resampled)


# In[40]:


# Evaluate the performance of the Logistic  model
y_pred_log = log_model.predict(X_test)
y_pred_log[y_pred_log >= 0.5] = 1
y_pred_log[y_pred_log < 0.5] = 0
recall_logistic = recall_score(y_test, y_pred_log)
f1_logistic = f1_score(y_test, y_pred_log)


# In[41]:


print("Logistic Regression:")
print('----------------------------------')
print("Recall:", recall_logistic)
print("F1 score:", f1_logistic)


# ##  Logistic regression model with Ridge regularization and  Lasso regularization
# 
# - The LogisticRegression class from scikit-learn's linear_model module to train the logistic regression models with Ridge and Lasso regularization. 
# - The penalty parameter is set to 'l2' for Ridge regularization and 'l1' for Lasso regularization.
# - The C parameter controls the regularization strength, and the solver parameter is set to 'liblinear' for Lasso regularization. - We also use the recall and F1 score function from scikit-learn's metrics module to evaluate the accuracy of the models.
# - The training is done with  resampled dataset and evaluation done  based on the original testing data.
# - Finally, the score metrics of both models are shown.

# In[42]:


# Train a logistic regression model with Ridge regularization

ridge_logreg = LogisticRegression(penalty='l2',max_iter=100,C=10)
from sklearn.model_selection import cross_val_score
mse=cross_val_score(ridge_logreg,X_train_resampled,y_train_resampled,scoring="neg_mean_squared_error",cv=5)
print(mse)
mean_mse=np.mean(mse)
print(mean_mse)


# In[43]:



ridge_logreg.fit(X_train_resampled, y_train_resampled)

# Evaluate the Ridge logistic regression model
y_pred = ridge_logreg.predict(X_test)


# In[44]:


# Evaluate the performance of the Ridge  model
y_pred_ridgelog = ridge_logreg.predict(X_test)
y_pred_ridgelog[y_pred_ridgelog >= 0.5] = 1
y_pred_ridgelog[y_pred_ridgelog < 0.5] = 0
recall_ridgelog = recall_score(y_test, y_pred_ridgelog)
f1_ridgelog = f1_score(y_test, y_pred_ridgelog)


# In[45]:


print("Ridge Logistic Regression:")
print('----------------------------------')
print("Recall:", recall_ridgelog)
print("F1 score:", f1_ridgelog)


# In[46]:


# Train a logistic regression model with Lasso regularization
lasso_logreg = LogisticRegression(penalty='l1', C=10, max_iter=100,solver='liblinear')
from sklearn.model_selection import cross_val_score
mse=cross_val_score(lasso_logreg,X_train_resampled,y_train_resampled,scoring="neg_mean_squared_error",cv=5)
print(mse)
mean_mse=np.mean(mse)
print(mean_mse)


# In[47]:


lasso_logreg.fit(X_train_resampled, y_train_resampled)


# In[48]:


# Evaluate the Lasso logistic regression model
y_pred_lassolog = lasso_logreg.predict(X_test)
y_pred_lassolog[y_pred_lassolog >= 0.5] = 1
y_pred_lassolog[y_pred_lassolog < 0.5] = 0
recall_lassolog = recall_score(y_test, y_pred_lassolog)
f1_lassolog = f1_score(y_test, y_pred_lassolog)


# In[49]:


print("Lasso Logistic Regression:")
print('----------------------------------')
print("Recall:", recall_lassolog)
print("F1 score:", f1_lassolog)


# ###  Confusion Matrix of Balanced Dataset
# 
# - In this particular confusion matrix of Logistic Model, the values are as follows:
# 
#    - 42 true negatives (TN): The model correctly predicted 42 instances as negative.
#    - 52 false positives (FP): The model incorrectly predicted 52 instances as positive.
#    - 42 false negatives (FN): The model incorrectly predicted 42 instances as negative.
#    - 204 true positives (TP): The model correctly predicted 204 instances as positive.
#    
# - In the  confusion matrix of Ridge Logistic Model, the values are as follows:
# 
#    - 41 true negatives (TN): The model correctly predicted 41 instances as negative.
#    - 56 false positives (FP): The model incorrectly predicted 56 instances as positive.
#    - 43 false negatives (FN): The model incorrectly predicted 43 instances as negative.
#    - 200 true positives (TP): The model correctly predicted 200 instances as positive.
# - In the  confusion matrix of Lasso Logistic Model, the values are as follows:
# 
#    - 41 true negatives (TN): The model correctly predicted 41 instances as negative.
#    - 57 false positives (FP): The model incorrectly predicted 57 instances as positive.
#    - 43 false negatives (FN): The model incorrectly predicted 43 instances as negative.
#    - 199 true positives (TP): The model correctly predicted 199 instances as positive.

# In[50]:


from sklearn.metrics import classification_report, confusion_matrix

print("----------------CLASSIFICATION REPORT OF LOGISTIC REGRESSION MODEL--------------------")
print(classification_report(y_test, y_pred_log))
print("----------------CLASSIFICATION REPORT OF RIDGE LOGISTIC REGRESSION MODEL--------------------")
print(classification_report(y_test, y_pred_ridgelog))
print("----------------CLASSIFICATION REPORT OF LASSO LOGISTIC REGRESSION MODEL--------------------")
print(classification_report(y_test, y_pred_lassolog))




# In[51]:


cm1 = confusion_matrix(y_test, y_pred_log)
cm2 = confusion_matrix(y_test, y_pred_ridgelog)
cm3 = confusion_matrix(y_test, y_pred_lassolog)
print("----------Confusion Matrix of Logistic Model------------------")
print(cm1 ,"\n")
print("----------Confusion Matrix of Ridge Logistic Model------------")
print(cm2,"\n")
print("----------Confusion Matrix of Lasso Logistic Model------------")
print(cm3)


# ###  Graphical View of results of Logistic,Ridge and Lasso 

# In[52]:


results_df = pd.DataFrame({'y_test': y_test,
                           'y_pred_logistic': y_pred_log,
                           'y_pred_ridge': y_pred_ridgelog,
                           'y_pred_lasso': y_pred_lassolog})


# In[53]:


sns.pairplot(results_df)


# ###  Logistic Model with Imbalanced Target column 
# 
# ### Confusion Matrix of Original  Dataset
# 
# - In this particular confusion matrix of Logistic Model, the values are as follows:
# 
#    - 6 True Negative (TN): The model correctly predicted 6 instances as negative.
#    - 78 false positives (FP): The model incorrectly predicted 78 instances as positive.
#    - 2 false negatives (FN): The model incorrectly predicted 2 instances as negative.
#    - 254 True positives (TP): The model correctly predicted 254 instances as positive.
#    
# - In the  confusion matrix of Ridge Logistic Model, the values are as follows:
# 
#    - 8 True Negative (TN): The model correctly predicted 6 instances as negative.
#    - 76 false positives (FP): The model incorrectly predicted 78 instances as positive.
#    - 4 false negatives (FN): The model incorrectly predicted 2 instances as negative.
#    - 252 True positives (TP): The model correctly predicted 254 instances as positive.
# - In the  confusion matrix of Lasso Logistic Model, the values are as follows:
# 
#    - 10 True Negatives (TN): The model correctly predicted 10 instances as negative.
#    - 74 false positives (FP): The model incorrectly predicted 74 instances as positive.
#    - 4 false negatives (FN): The model incorrectly predicted 4 instances as negative.
#    - 252 True positives (TP): The model correctly predicted 252 instances as positive.
# - From the observed results,the prediction using balanced dataset stays perfect.

# In[54]:


from sklearn.linear_model import LogisticRegression
log_model = LogisticRegression()
from sklearn.model_selection import cross_val_score

mse=cross_val_score(log_model,X_train,y_train,scoring="neg_mean_squared_error",cv=5)
print(mse)
mean_mse=np.mean(mse)
print(mean_mse)


log_model.fit(X_train, y_train)


# Evaluate the performance of the Logistic  model
y_pred_log = log_model.predict(X_test)
y_pred_log[y_pred_log >= 0.5] = 1
y_pred_log[y_pred_log < 0.5] = 0
recall_logistic = recall_score(y_test, y_pred_log)
f1_logistic = f1_score(y_test, y_pred_log)

print("Logistic Regression:")
print('----------------------------------')
print("Recall:", recall_logistic)
print("F1 score:", f1_logistic)


# Train a logistic regression model with Ridge regularization

ridge_logreg = LogisticRegression(penalty='l2',max_iter=100,C=10)
from sklearn.model_selection import cross_val_score
mse=cross_val_score(ridge_logreg,X_train,y_train,scoring="neg_mean_squared_error",cv=5)
print(mse)
mean_mse=np.mean(mse)
print(mean_mse)


ridge_logreg.fit(X_train, y_train)

# Evaluate the Ridge logistic regression model
y_pred = ridge_logreg.predict(X_test)



# Evaluate the performance of the Ridge  model
y_pred_ridgelog = ridge_logreg.predict(X_test)
y_pred_ridgelog[y_pred_ridgelog >= 0.5] = 1
y_pred_ridgelog[y_pred_ridgelog < 0.5] = 0
recall_ridgelog = recall_score(y_test, y_pred_ridgelog)
f1_ridgelog = f1_score(y_test, y_pred_ridgelog)


print("Ridge Logistic Regression:")
print('----------------------------------')
print("Recall:", recall_ridgelog)
print("F1 score:", f1_ridgelog)


# Train a logistic regression model with Lasso regularization
lasso_logreg = LogisticRegression(penalty='l1', C=10, max_iter=100,solver='liblinear')
from sklearn.model_selection import cross_val_score
mse=cross_val_score(lasso_logreg,X_train,y_train,scoring="neg_mean_squared_error",cv=5)
print(mse)
mean_mse=np.mean(mse)
print(mean_mse)


lasso_logreg.fit(X_train, y_train)


# Evaluate the Lasso logistic regression model
y_pred_lassolog = lasso_logreg.predict(X_test)
y_pred_lassolog[y_pred_lassolog >= 0.5] = 1
y_pred_lassolog[y_pred_lassolog < 0.5] = 0
recall_lassolog = recall_score(y_test, y_pred_lassolog)
f1_lassolog = f1_score(y_test, y_pred_lassolog)


print("Lasso Logistic Regression:")
print('----------------------------------')
print("Recall:", recall_lassolog)
print("F1 score:", f1_lassolog)



from sklearn.metrics import classification_report, confusion_matrix

print("----------------CLASSIFICATION REPORT OF LOGISTIC REGRESSION MODEL--------------------")
print(classification_report(y_test, y_pred_log))
print("----------------CLASSIFICATION REPORT OF RIDGE LOGISTIC REGRESSION MODEL--------------------")
print(classification_report(y_test, y_pred_ridgelog))
print("----------------CLASSIFICATION REPORT OF LASSO LOGISTIC REGRESSION MODEL--------------------")
print(classification_report(y_test, y_pred_lassolog))



cm1 = confusion_matrix(y_test, y_pred_log)
cm2 = confusion_matrix(y_test, y_pred_ridgelog)
cm3 = confusion_matrix(y_test, y_pred_lassolog)
print("----------Confusion Matrix of Logistic Model------------------")
print(cm1 ,"\n")
print("----------Confusion Matrix of Ridge Logistic Model------------")
print(cm2,"\n")
print("----------Confusion Matrix of Lasso Logistic Model------------")
print(cm3)


# ###  Neural Networks
# 
# ### Data Preprocessing Steps
# 
# - Encoding of Rating column: The 'Rating' column of the dataset is encoded using a LabelEncoder to convert its string values into numerical values.
# 
# - Conversion of encoded Rating column to categorical variable: The encoded 'Rating' column is then converted to a categorical variable using np_utils.to_categorical function.
# 
# - Splitting of dataset: The dataset is then split into training and test sets using the train_test_split function from scikit-learn library. The features and target variables are also split separately for both the target variables (Rating and InvGrd).
# 
# - Scaling of data: The features of the dataset are standardized using StandardScaler from the scikit-learn library to ensure that each feature has a mean of 0 and standard deviation of 1. This step is performed to make sure that all the features contribute equally in the machine learning model.
# 
# 
# 

# In[55]:


data['Rating_encoded'] = label_encoder.fit_transform(data['Rating'])

#  Convert the encoded 'Rating' column to a categorical variable.
y = np_utils.to_categorical(data['Rating_encoded'])
y

# Independent and Dependent Features
X=data.drop(["InvGrd","Rating","Rating_encoded"],axis=1)
y_rating=y
y_invgrd=data['InvGrd']

# Split the dataset into training and test sets.
X_train, X_test, y_train_rating, y_test_rating, y_train_invgrd, y_test_invgrd = train_test_split(
    X, y_rating, y_invgrd, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ### Building and Training a Neural Networks Model
# 
# - Importing necessary libraries: The required libraries, including StandardScaler from scikit-learn and Sequential and Dense from Keras, are imported.
# 
# - Initializing a Sequential model: A Sequential model is initialized using the Sequential() function from Keras.
# 
# - Adding layers to the model: Three layers are added to the model using the add() function. The first layer has 64 neurons with 'relu' activation function and takes the input dimensions from the X_train dataset. The second layer has 32 neurons with 'relu' activation function. The third layer has 16 neurons with 'softmax' activation function, which is used to classify the output into multiple categories as we have 16 categories in Rating column.
# 
# - Compiling the model: The model is compiled using the compile() function. The loss function used is 'categorical_crossentropy', which is commonly used for multiclass classification problems. The optimizer used is 'adam', which is an efficient gradient descent optimization algorithm. The metrics used to evaluate the model are 'accuracy'.
# 
# - Fitting the model: The model is trained using the fit() function. The X_train and y_train_rating datasets are used as input, and the number of epochs, batch size, and verbose level are specified. The training process runs for 100 epochs with a batch size of 32, and the verbose level is set to 0 to suppress the output during training. 

# In[56]:


#  Build and train a Neural Networks model.
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train_rating, epochs=100, batch_size=32, verbose=0)


# ### Evaluating the Model's Performance on the Training and Test Set
# 
# - Evaluating the model on the training set: The evaluate() function is used to calculate the loss and accuracy of the trained model on the training set. The X_train and y_train_rating datasets are used as input, and the calculated accuracy is stored in the 'accuracy' variable.
# 
# - Printing the training set accuracy: The calculated accuracy is then printed on the console using the print() function.
# 
# - Evaluating the model on the test set: The evaluate() function is used again to calculate the loss and accuracy of the trained model on the test set. The X_test and y_test_rating datasets are used as input, and the calculated accuracy is stored in the 'accuracy' variable.
# 
# - Printing the test set accuracy: The calculated accuracy is then printed on the console using the print() function.

# In[57]:


# Evaluate the model's performance on the training and test set.

loss, accuracy = model.evaluate(X_train, y_train_rating)
print(f'Test set accuracy: {accuracy:.2f}')


loss, accuracy = model.evaluate(X_test, y_test_rating)
print(f'Test set accuracy: {accuracy:.2f}')


# ### Converting the Label Encoded Array to Class Labels
# 
# 
# - Generating predicted ratings for the test set: The predict() function is used to generate the predicted ratings for the test set. The X_test dataset is used as input, and the predicted ratings are stored in the 'rating_pred' variable.
# 
# - Converting the label encoded array to class labels: The argmax() function is used to convert the one-hot encoded array into a 1D array of class labels. This function returns the index of the highest value in each array, which corresponds to the predicted class label.
# 
# - Printing the resulting array of class labels: The resulting array of class labels is then printed on the console using the print() function.
# 
# - Converting the class labels to their original form: The inverse_transform() function of the LabelEncoder is used to convert the predicted class labels back to their original form. The 'rating_pred_labels' array is passed as input to this function, and the resulting array is stored in the 'rating_pred' variable. This step is performed to convert the numerical labels back to their original string values. 

# In[58]:


rating_pred=model.predict(X_test)

rating_pred

# Convert the one-hot encoded array to a 1D array of class labels.
rating_pred_labels = np.argmax(rating_pred, axis=1)

# Print the resulting array.
print(rating_pred_labels)

rating_pred = label_encoder.inverse_transform(rating_pred_labels)
rating_pred


# ### Predicting the Investment Grade Status of Firms Based on Predicted Ratings
# 
# 
# - Defining the set of investment grade ratings: The set of investment grade ratings is defined using a Python set. This set contains the credit ratings that are considered to be investment grade.
# 
# - Predicting the investment grade status of each firm: The invgrd_pred list is generated by iterating over the predicted credit ratings using a list comprehension. If a credit rating is present in the investment_grade_ratings set, then the corresponding firm is predicted to be investment grade; otherwise, it is predicted to be non-investment grade.
# 
# - Printing the predicted investment grade status: The predicted investment grade status for the test set is printed on the console using the print() function.
# 
# - Converting the predicted status to binary labels: The invgrd_pred list is further converted into binary labels, where 1 represents investment grade and 0 represents non-investment grade. This step is performed to convert the predicted investment grade status into a format that can be used for evaluation. 

# In[59]:


# Define the set of investment grade ratings.
investment_grade_ratings = {'Aaa', 'Aa1', 'Aa2', 'Aa3', 'A1', 'A2', 'A3', 'Baa1', 'Baa2', 'Baa3'}

# Step 3: Predict the investment grade status of each firm based on its predicted rating.
invgrd_pred = ['Investment Grade' if rating in investment_grade_ratings else 'Non-Investment Grade' for rating in rating_pred]

# Step 4: Print the predicted investment grade status for the test set.
print(invgrd_pred)

invgrd_pred= [1 if rating in investment_grade_ratings else 0 for rating in rating_pred]
invgrd_pred


# ### Visualizing the Relationship between Predicted Credit Rating and Investment Grade Status
# 
# - The given code creates a pairwise plot to visualize the relationship between the predicted credit rating and investment grade status for the test set.

# In[60]:


import seaborn as sns
import pandas as pd

# Create a dataframe with y_pred_rating and y_pred_invgrd
df = pd.DataFrame({'rating_pred_labels': rating_pred_labels, 'invgrd_pred': invgrd_pred})

# Create a pairwise plot
sns.pairplot(df, hue='invgrd_pred')

# Display the plot
plt.show()


# ### Computing Recall Score for Predicted Investment Grade Status
# 
# - The given code computes the recall score for the predicted investment grade status using the y_test_invgrd (actual investment grade status) and invgrd_pred (predicted investment grade status) arrays. 
# 
# -  Recall score is the ratio of true positive predictions to the total number of actual positive values. It measures the ability of the model to correctly identify positive instances that is identifying investment grade firms. 

# In[63]:


recall_neural = recall_score(y_test_invgrd, invgrd_pred)
print('Recall Score is :',recall_neural)


# ### Printing Classification Report and Confusion 
# 
# - The given code prints the classification report and confusion matrix for the predicted investment grade status using the y_test_invgrd (actual investment grade status) and invgrd_pred (predicted investment grade status) arrays.  

# ## Classification Report and Confusion Matrix

# In[61]:


#Printing the Classification report and confusion matrix

print("-----------------------------CLASSIFICATION REPORT-----------------------------")

print(classification_report(y_test_invgrd,invgrd_pred))

print("-----------------------------CONFUSION MATRIX----------------------------------")

print(confusion_matrix(y_test_invgrd,invgrd_pred))


# ### Creating and Displaying a DataFrame for Model Analysis
# 
# - The given code creates a dictionary named analysis_data that stores the accuracy and recall values for each model. 

# In[65]:


# Create a dictionary with the accuracy and recall values for each model
analysis_data = {'Model': ['Linear-Ridge', 'Linear-Lasso', 'Logistic-Ridge','Logistic-Lasso','Neural Network'],
        'Accuracy': [0.71,0.70,0.72,0.71,0.79],
        'Recall': [0.82,0.81,0.78,0.77,0.89]}

# Create a DataFrame from the dictionary
df = pd.DataFrame(analysis_data)

# Display the DataFrame
# Format the table with a border
styled_df = df.style.set_table_styles([{'selector': 'table', 'props': [('border', '2px solid black')]}])

# Display the formatted table
display(styled_df)


# ## Analysis of the Models
# 
# - Based on the results in the table, to predict an investgrade ,the Neural Network model performs the best, with an accuracy of 0.79 and a recall of 0.89. The Logistic Ridge model also performs relatively well, with an accuracy of 0.72 and a recall of 0.78. The Linear Ridge and Lasso, and Logistic Lasso models have similar accuracy and recall values, with slightly lower performance than the other two models.
