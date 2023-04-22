#!/usr/bin/env python
# coding: utf-8

# In[1]:


# loading the moduules 
import numpy as np 
import pandas as pd 


# In[2]:


import warnings
warnings.filterwarnings("ignore")


# In[3]:


#!pip install plotly
get_ipython().system('pip install plotly')


# In[4]:


import pandas as pd
import numpy as np
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
from IPython.display import Image
warnings.filterwarnings('ignore')
import os
import plotly.express as px
import sklearn
import matplotlib.pyplot as plt
import pandas as pd
import plotly
from plotly.offline import plot, iplot
import plotly.graph_objects as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
from plotly import tools
import plotly.tools as tls
init_notebook_mode(connected=True)


# In[5]:


get_ipython().system('pip install cufflinks')


# In[6]:


import cufflinks as cf
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)


# In[7]:


import plotly
plotly.offline.init_notebook_mode(connected=True)


# In[8]:


student=pd.read_csv('exams.csv',na_values=[',','?','#','@'])
student.head()


# In[9]:


student.shape


# #Display summary statistics

# In[10]:


student.describe() # summary to understand the data, prints the summary only for numeric attributes by default


# In[11]:


student.describe(include='all')


# #Getting information about the data

# In[12]:


student.info()


# #Display Columns

# In[13]:


student.columns


# In[14]:


student.isnull().sum()


# ### Recode the levels of target on  data ; yes=1 and no=0
# 

# In[15]:


# df['test preparation course'] = df['test preparation course'].apply(lambda x: 0 if x.strip()=='none' else 1)

student['test preparation course']=np.where(student['test preparation course']== 'none',0,1)


# #Identifying categorical attributes

# In[16]:


cat_attr=['gender', 'race/ethnicity', 'parental level of education', 'lunch','test preparation course']


# #Identifying numerical attributes

# In[17]:


num_attr=['reading score','writing score']


# #Convert all the attributes to appropriate type¶

# In[18]:


for col in cat_attr:
  student[cat_attr] = student[cat_attr].astype('category')


# In[19]:


student.dtypes


# In[20]:


student.info()


# In[21]:


student.gender.value_counts(normalize=True)*100


# In[22]:


from cufflinks.plotlytools import normalize
student['race/ethnicity'].value_counts(normalize=True)*100


# In[23]:


student['parental level of education'].value_counts(normalize=True)*100


# In[24]:


student['lunch'].value_counts(normalize=True)*100


# In[25]:


student['test preparation course'].value_counts(normalize=True)*100


# In[26]:


student['writing score'].value_counts()


# #Exploratory Data Analysis 
# 
# Let's explore the data!

# In[27]:


import plotly.express as px #importing plotly


# In[28]:


fig = px.scatter(student, x="reading score", y="math score",trendline="ols")
fig.show() #plotting scatter plot with "reading score","math score"


# In[29]:


fig = px.scatter(student, x="writing score", y="math score",trendline="ols")
fig.show() #plotting scatter plot with "writing score","math score"


# In[30]:


fig=px.histogram(student,x="math score", marginal = 'box')
fig.show()


# fig=px.histogram(student,x="writing score", marginal = 'box')
# fig.show()

# In[31]:


fig=px.histogram(student,x="reading score", marginal = 'box')
fig.show()


# In[32]:


fig = px.ecdf(student, x="math score", color="gender")
fig.show()


# In[33]:


tmp = student.groupby('gender').agg({'math score':'mean'}).reset_index()

fig = px.bar(tmp, y="math score", x="gender", text='math score')

fig.update_yaxes(title='avg_math score')

fig.update_traces(texttemplate='%{text:.2s}', textposition='auto')

fig.show()


# In[34]:


tmp = student.groupby('race/ethnicity').agg({'math score':'mean'}).reset_index()

fig = px.bar(tmp, y="math score", x="race/ethnicity", text='math score')

fig.update_yaxes(title='avg_math score')

fig.update_traces(texttemplate='%{text:.2s}', textposition='auto')

fig.show()


# In[35]:


tmp = student.groupby('test preparation course').agg({'math score':'mean'}).reset_index()

fig = px.bar(tmp, y="math score", x="test preparation course", text='math score')

fig.update_yaxes(title='avg_math score')

fig.update_traces(texttemplate='%{text:.2s}', textposition='auto')

fig.show()


# In[36]:


tmp = student.groupby('lunch').agg({'math score':'mean'}).reset_index()

fig = px.bar(tmp, y="math score", x="lunch", text='math score')

fig.update_yaxes(title='avg_math score')

fig.update_traces(texttemplate='%{text:.2s}', textposition='auto')

fig.show()


# In[37]:


fig = px.bar(student,x='lunch',y ='math score')
fig.show()


# In[38]:


student.columns


# In[39]:


fig = px.bar(student,x='gender',y ='math score')
fig.show()


# In[40]:



fig = px.bar(student, x='race/ethnicity', y='math score')
fig.show()


# In[41]:


fig = px.bar(student, x='parental level of education', y='math score')
fig.show()


# In[42]:


tmp = student.groupby('parental level of education').agg({'math score':'mean'}).reset_index()

fig = px.bar(tmp, y="math score", x="parental level of education", text='math score')

fig.update_yaxes(title='avg_math score')

fig.update_traces(texttemplate='%{text:.2s}', textposition='auto')

fig.show()


# # Data Preparation for model building 
# 
# Now that we've explored the data a bit, let's go ahead and split the data into training and testing sets.
# ** Set a variable X equal to the numerical & catgorical features of the customers and a variable y equal to the "math score" column. **

# In[43]:


student.head() #snapshot of clean data


# In[44]:


student.info()


# In[45]:


X = student.drop(['test preparation course'],axis=1) #independent attributes


# In[46]:


y = student['test preparation course'] #dependent attributes


# ## Spilit the data into Training and Testing Data

# In[47]:


import statsmodels.api as sm # import stats model o/p : R model
from sklearn.model_selection import train_test_split # importing neccessary modules


# In[48]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101,stratify= y) #train test split


# In[49]:


#print(len(y_train))
print('Y Target Size:', len(y_train))
print('X Train Size:', len(X_train))
print('X Test  Size:', len(X_test))
print('Y Target Size:', len(y_test))


# # Multiple Regression model with Categorical Variables - Dummification

# In[50]:


#Dummification 
X_train = pd.get_dummies(X_train,drop_first=True,dtype='int8')
X_test = pd.get_dummies(X_test,drop_first=True,dtype='int8')


# In[51]:


X_train.head() #snapshot of X train


# In[52]:


print(X_train.columns)
print("\n")
print(X_test.columns)
print("\n")
print("\n No. of columns in Train Data :{}".format(len(X_train.columns)))
print("\n No. of columns in Test Data :{}".format(len(X_test.columns))) 


# ## Standardizing the Numerical data 

# In[53]:


from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = StandardScaler() #importing Scaler


# In[54]:


scaler = StandardScaler()
X_train[num_attr] = scaler.fit_transform(X_train[num_attr])
X_test[num_attr] = scaler.transform(X_test[num_attr])

print(X_train.head()) 
#Scaling


# In[55]:


X_train.head()


# In[56]:


# adding the Constant term
X_train = sm.add_constant(X_train)
print(X_train.head())

X_test = sm.add_constant(X_test)
#X_test.head()


# ## Training the Model
# 
# Now its time to train our model on our training data!
# 
# 

# In[57]:


logit_model=sm.Logit(y_train,X_train)
result=logit_model.fit()
print(result.summary())


# In[58]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score,recall_score,precision_score

import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB 
from sklearn.naive_bayes import GaussianNB 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[59]:


logistic_model = LogisticRegression()

logistic_model.fit(X_train,y_train)


# In[60]:


train_preds = logistic_model.predict(X_train)
train_preds_prob=logistic_model.predict_proba(X_train)[:,1]
test_preds = logistic_model.predict(X_test)
test_preds_prob=logistic_model.predict_proba(X_test)[:,1]


# In[61]:


train_preds


# In[62]:


logistic_model.coef_


# # Errors metrics to evaluate the model
# 
# ### Confusion Matrix

# In[63]:


confusion_matrix(y_train,train_preds)


# In[64]:


train_accuracy_1= accuracy_score(y_train,train_preds)
train_recall_1= recall_score(y_train,train_preds)
train_precision_1= precision_score(y_train,train_preds)

test_accuracy_1= accuracy_score(y_test,test_preds)
test_recall_1= recall_score(y_test,test_preds)
test_precision_1= precision_score(y_test,test_preds)

print(train_accuracy_1)
print(train_recall_1)
print(train_precision_1)

print(test_accuracy_1)
print(test_recall_1)
print(test_precision_1)


# In[65]:


#Classification report
print(classification_report(y_train,train_preds))


# In[66]:


print(classification_report(y_test,test_preds))


# ### ROC and AUC

# In[67]:


fpr, tpr, threshold = roc_curve(y_train, train_preds_prob)
roc_auc = auc(fpr, tpr)


# In[68]:


get_ipython().run_line_magic('matplotlib', 'notebook')
# plt.figure()
plt.plot([0,1],[0,1],color='navy', lw=2, linestyle='--')
plt.plot(fpr,tpr,color='orange', lw=3, label='ROC curve (area = %0.2f)' % roc_auc)

plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend(loc="lower right")


# ### Manual inspection of threshold value

# In[69]:


roc_df = pd.DataFrame({'FPR':fpr, 'TPR':tpr, 'Threshold':threshold})

roc_df


# In[70]:


roc_df.sort_values('TPR',ascending=False,inplace=True)


# In[71]:


optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = threshold[optimal_idx]


# In[72]:


optimal_threshold


# In[73]:


custom_threshold = 0.345


## To get in 0-1 format vector (pandas Series)
final_pred_array = pd.Series([0 if x>custom_threshold else 1 for x in train_preds_prob])
final_pred_array.value_counts()

final_test_pred_array = pd.Series([0 if x>custom_threshold else 1 for x in test_preds_prob])
final_test_pred_array.value_counts()


# In[74]:


## To get True-False format vector (pandas Series)
final_pred = pd.Series(train_preds_prob > 0.345)
final_pred.value_counts()
final_test_pred=pd.Series(test_preds_prob > 0.345)


# In[75]:


print(classification_report(y_train,final_pred))


# In[76]:


print(classification_report(y_test,final_test_pred))


# In[77]:


train_accuracy= accuracy_score(y_train,final_pred)
train_recall= recall_score(y_train,final_pred)
print(train_accuracy)
print(train_recall)

test_accuracy= accuracy_score(y_test,final_test_pred)
test_recall= recall_score(y_test,final_test_pred)
print(test_accuracy)
print(test_recall)


# In[78]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


# ## RandomForestClassifier Model

# In[79]:


clf1 = RandomForestClassifier()


# In[80]:


clf1.fit(X_train,y_train)


# # Identify Right Error Metrics
# 

# ## Function to calculate required metrics

# In[81]:


def evaluate_model(act, pred):
    print("Confusion Matrix \n", confusion_matrix(act, pred))
    print("Accurcay : ", accuracy_score(act, pred))
    print("Recall   : ", recall_score(act, pred))
    print("Precision: ", precision_score(act, pred))    


# In[82]:


train_pred = clf1.predict(X_train)
test_pred = clf1.predict(X_test)


# In[83]:


print("--Train--")
evaluate_model(y_train, train_pred)
print("--Test--")
evaluate_model(y_test, test_pred)


# In[84]:


student.columns


# In[85]:


student['test preparation course'].value_counts()


# ## Up-sampling 
# 

# In[86]:


get_ipython().system('pip install imblearn')


# In[87]:


from imblearn import under_sampling, over_sampling
from imblearn.over_sampling import SMOTE


# ### Instantiate SMOTE

# In[88]:


smote=SMOTE(random_state=123)


# ### Fit Sample

# In[89]:


X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)


# In[90]:


np.unique(y_train, return_counts= True)


# In[91]:


np.unique(y_train_sm, return_counts= True)


# ## RandomForestClassifier with up-sample data

# ### Instantiate Model

# In[92]:


clf2=RandomForestClassifier()


# ### Train the model

# In[93]:


clf2.fit(X_train_sm, y_train_sm)


# ### Predict

# In[94]:


train_pred1 = clf2.predict(X_train_sm)
test_pred1 = clf2.predict(X_test)


# ### Evaluate

# In[95]:


print("--Train--")
evaluate_model(y_train_sm, train_pred1)
print("--Test--")
evaluate_model(y_test, test_pred1)


# ## Hyper-parameter tuning using Grid Search and Cross Validation

# ### Parameters to test

# In[96]:


param_grid = {"n_estimators" : [100,140,180],
              "max_depth" : [1,2,3],
              "max_features" : [2,4,6,10],
              "min_samples_leaf" : [1, 2, 3]}


# ### Instantiate Tree

# ### Instantiate GridSearchCV 

# In[97]:


clf_grid = GridSearchCV(clf2,param_grid,cv=2)


# ### Train DT using GridSearchCV

# In[98]:


clf_grid.fit(X_train_sm, y_train_sm)


# ### Best Params

# In[99]:


clf_grid.best_params_


# ### Predict 

# In[100]:


train_pred2= clf_grid.predict(X_train_sm)
test_pred2 = clf_grid.predict(X_test)


# In[101]:


print("--Train--")
evaluate_model(y_train_sm, train_pred2)
print("--Test--")
evaluate_model(y_test, test_pred2)


# In[102]:


importances = clf_grid.best_estimator_.feature_importances_
indices = np.argsort(importances)[::-1]
print(indices)

select = indices[0:10]
print(select)


# In[103]:


clf4 = RandomForestClassifier(max_depth=3, max_features=10,
                              min_samples_leaf=3, n_estimators=100)


# In[104]:


clf4.fit(X_train_sm.iloc[:,select],y_train_sm)


# In[105]:


train_pred3= clf4.predict(X_train_sm.iloc[:,select])
test_pred3= clf4.predict(X_test.iloc[:,select])


# In[106]:


print("--Train--")
evaluate_model(y_train_sm, train_pred3)
print("--Test--")
evaluate_model(y_test, test_pred3)


# In[ ]:





# In[ ]:




