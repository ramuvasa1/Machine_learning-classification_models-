# -*- coding: utf-8 -*-
"""Our Aim is to protect our customers without churning from our company( Telecommunications)_ML_.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1EtPltM0HY2sPjiXhQOjdSWXC0EvKDpsW
"""

!pip install pandas

from google.colab import drive
drive.mount('/content/drive')

import numpy as np
import pandas as pd
data1 = pd.read_csv('/content/drive/MyDrive/PROJECT/Telco_Churn_Details1.csv',na_values=["?",","])
data1.head()

data2= pd.read_csv('/content/drive/MyDrive/PROJECT/Telco_Churn_Details2.csv',na_values=["?",","])
data2.head()

data3= pd.read_csv('/content/drive/MyDrive/PROJECT/Telco_Customer_Call_Details1.csv',na_values=["?",","])
data3.head()

data4= pd.read_csv('/content/drive/MyDrive/PROJECT/Telco_Customer_Call_Details2.csv',na_values=["?",","])
data4.head()

telco=pd.concat([data1, data2],axis=0,ignore_index=True)

telco.head()

telco1=pd.concat([data3, data4],axis=0,ignore_index=True)

telco1

telco_cc= pd.merge(telco1,telco, on="Cust_ID")
telco_cc.head()

telco_cc['trainrows'].value_counts()

telco_cc['trainrows']=np.where(telco_cc['trainrows']== 'Yes',1,0)
telco_cc

telco_cc.shape

telco_cc.columns

telco_cc.dtypes

telco_cc.isnull().sum()

telco_cc.describe()

telco_cc.Churn.value_counts(normalize=True)*100

telco_cc= telco_cc.drop('Cust_ID',axis=1)
telco_cc

telco_cc['Churn']=np.where(telco_cc['Churn']== 'False.',0,1)
telco_cc['Churn'].value_counts()

print(telco_cc["X..customer.Service.Calls"].unique())

telco_cc.dtypes

cat_attr=['International.Plan', 'Voice.Mail.Plan','X..customer.Service.Calls']
telco_cc[cat_attr]= telco_cc[cat_attr].astype("category")

telco_cc.dtypes

num_attr=["Total.Day.Minutes","Total.Day.Calls","Total.Day.Charge", "Total.Eve.Minutes","Total.Eve.Calls",
"Total.Eve.Charge",            
"Total.Night.Minutes",         
"Total.Night.Calls",           
"Total.Night.Charge",          
"Total.Intl.Minutes",         
"Total.Intl.Calls",            
"Total.Intl.Charge",           
"X..Vmail.Messages"]

telco_cc.dtypes

telco_cc.isnull().sum()

telco_cc.head()

telco_cc.trainrows.value_counts()

#check correlation with pearson method
# columns shown here are selected by corr() since
telco_cc[num_attr].corr(method ='pearson')

import seaborn as sns
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(19, 10))

sns.heatmap(telco_cc[num_attr].corr(), ax=ax, annot=True)

telco_cc.head()

telco_cc.shape

train=telco_cc[telco_cc["trainrows"]==1]
train.head()

train.shape

val=telco_cc[telco_cc["trainrows"]==0]
val.head()

train.drop('trainrows',axis=1,inplace=True)
train

val.drop('trainrows',axis=1,inplace=True)
val

X=train.drop(["Churn"],axis=1)
y=train["Churn"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify= y, test_size = 0.3, random_state=124)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# Standardizing the numeric attributes in the train and test data:
from sklearn.preprocessing import StandardScaler,MinMaxScaler,OneHotEncoder

scaler = StandardScaler()
scaler.fit(X_train[num_attr])

X_train_num = pd.DataFrame(scaler.transform(X_train[num_attr]), columns=num_attr)
X_test_num = pd.DataFrame(scaler.transform(X_test[num_attr]), columns=num_attr)

X_train_num.head()

"""OneHotEncoder : Independent Categorical Attributes"""

ohe = OneHotEncoder()

ohe.fit(X_train[cat_attr])

columns_ohe = list(ohe.get_feature_names(cat_attr))
print(columns_ohe)

X_train_cat = ohe.transform(X_train[cat_attr])
X_test_cat  = ohe.transform(X_test[cat_attr])

X_train_cat = pd.DataFrame(X_train_cat.todense(), columns=columns_ohe)
X_test_cat  = pd.DataFrame(X_test_cat.todense(), columns=columns_ohe)

X_train_cat

"""Concatenate"""

X_train = pd.concat([X_train_num, X_train_cat], axis=1)
X_test = pd.concat([X_test_num, X_test_cat], axis=1)

print(X_train.shape, X_test.shape)

X_train.head()

X_test.head()

"""Function to calculate accuracy, recall, precision and F1 score"""

scores = pd.DataFrame(columns=['Model', 'Train_Accuracy', 'Train_Recall', 'Train_Precision', 'Train_F1_Score', 
                               'Test_Accuracy', 'Test_Recall', 'Test_Precision', 'Test_F1_Score'])

def get_metrics(train_actual, train_predicted, test_actual, test_predicted, model_description, dataframe):

    train_accuracy  = accuracy_score(train_actual, train_predicted)
    train_recall    = recall_score(train_actual, train_predicted, average="weighted")
    train_precision = precision_score(train_actual, train_predicted, average="weighted")
    train_f1score   = f1_score(train_actual, train_predicted, average="weighted")
    
    test_accuracy   = accuracy_score(test_actual, test_predicted)
    test_recall     = recall_score(test_actual, test_predicted, average="weighted")
    test_precision  = precision_score(test_actual, test_predicted, average="weighted")
    test_f1score    = f1_score(test_actual, test_predicted, average="weighted")

    dataframe       = dataframe.append(pd.Series([model_description, 
                                                  train_accuracy, train_recall, train_precision, train_f1score,
                                                  test_accuracy, test_recall, test_precision, test_f1score],
                                                 index=scores.columns ), 
                                       ignore_index=True)

    return(dataframe)

def classification_report_train_test(y_train, train_preds, y_test, test_preds):

    print('''
            =========================================
               CLASSIFICATION REPORT FOR TRAIN DATA
            =========================================
            ''')
    print(classification_report(y_train, train_preds, digits=2))

    print('''
            =========================================
               CLASSIFICATION REPORT FOR TEST DATA
            =========================================
            ''')
    print(classification_report(y_test, test_preds, digits=2))

"""Model Building

LogisticRegression
"""

from sklearn.linear_model import LogisticRegression

logistic_model = LogisticRegression()
logistic_model.fit(X_train,y_train)

train_preds = logistic_model.predict(X_train)
train_preds_prob=logistic_model.predict_proba(X_train)
test_preds = logistic_model.predict(X_test)
test_preds_prob=logistic_model.predict_proba(X_test)

train_preds

logistic_model.coef_

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,recall_score,precision_score,f1_score

"""Error Metrics"""

confusion_matrix(y_train,train_preds)

confusion_matrix(y_test,test_preds)

classification_report_train_test(y_train, train_preds, y_test, test_preds)

"""LogisticRegression using with parameters"""

logistic_model = LogisticRegression(class_weight="balanced")
logistic_model.fit(X_train,y_train)

train_preds = logistic_model.predict(X_train)
train_preds_prob=logistic_model.predict_proba(X_train)
test_preds = logistic_model.predict(X_test)
test_preds_prob=logistic_model.predict_proba(X_test)

train_preds

logistic_model.coef_

"""Error Metrics"""

confusion_matrix(y_train,train_preds)

confusion_matrix(y_test,test_preds)

classification_report_train_test(y_train, train_preds, y_test, test_preds)

logistic_model = LogisticRegression(penalty="l1",solver="liblinear")
logistic_model.fit(X_train,y_train)

train_preds = logistic_model.predict(X_train)
train_preds_prob=logistic_model.predict_proba(X_train)
test_preds = logistic_model.predict(X_test)
test_preds_prob=logistic_model.predict_proba(X_test)

train_preds

logistic_model.coef_

confusion_matrix(y_train,train_preds)

confusion_matrix(y_test,test_preds)

classification_report_train_test(y_train, train_preds, y_test, test_preds)

logistic_model = LogisticRegression(penalty="l1",solver="liblinear",class_weight="balanced")
logistic_model.fit(X_train,y_train)

train_preds = logistic_model.predict(X_train)
train_preds_prob=logistic_model.predict_proba(X_train)
test_preds = logistic_model.predict(X_test)
test_preds_prob=logistic_model.predict_proba(X_test)

train_preds

logistic_model.coef_

confusion_matrix(y_train,train_preds)

confusion_matrix(y_test,test_preds)

classification_report_train_test(y_train, train_preds, y_test, test_preds)

logistic_model = LogisticRegression(penalty="l1",solver="liblinear",class_weight="balanced",multi_class="ovr",max_iter=124)
logistic_model.fit(X_train,y_train)

train_preds = logistic_model.predict(X_train)
train_preds_prob=logistic_model.predict_proba(X_train)
test_preds = logistic_model.predict(X_test)
test_preds_prob=logistic_model.predict_proba(X_test)

train_preds

logistic_model.coef_

confusion_matrix(y_train,train_preds)

confusion_matrix(y_test,test_preds)

classification_report_train_test(y_train, train_preds, y_test, test_preds)

X_train.columns

np.round(logistic_model.coef_,3)

logistic_model = LogisticRegression(penalty="l2",solver="saga")
logistic_model.fit(X_train,y_train)

train_preds = logistic_model.predict(X_train)
train_preds_prob=logistic_model.predict_proba(X_train)
test_preds = logistic_model.predict(X_test)
test_preds_prob=logistic_model.predict_proba(X_test)

train_preds

logistic_model.coef_

confusion_matrix(y_train,train_preds)

confusion_matrix(y_test,test_preds)

classification_report_train_test(y_train, train_preds, y_test, test_preds)

np.round(logistic_model.coef_,3)

X_train.columns

logistic_model = LogisticRegression(penalty="l2",solver="saga",class_weight="balanced")
logistic_model.fit(X_train,y_train)

train_preds = logistic_model.predict(X_train)
train_preds_prob=logistic_model.predict_proba(X_train)
test_preds = logistic_model.predict(X_test)
test_preds_prob=logistic_model.predict_proba(X_test)

train_preds

logistic_model.coef_

confusion_matrix(y_train,train_preds)

confusion_matrix(y_test,test_preds)

classification_report_train_test(y_train, train_preds, y_test, test_preds)

logistic_model = LogisticRegression(penalty="l2",solver="saga",class_weight="balanced",multi_class="ovr",max_iter=124)
logistic_model.fit(X_train,y_train)

train_preds = logistic_model.predict(X_train)
train_preds_prob=logistic_model.predict_proba(X_train)
test_preds = logistic_model.predict(X_test)
test_preds_prob=logistic_model.predict_proba(X_test)

train_preds

logistic_model.coef_

confusion_matrix(y_train,train_preds)

confusion_matrix(y_test,test_preds)

classification_report_train_test(y_train, train_preds, y_test, test_preds)

import copy
data= copy.copy(telco_cc)

data.head()

data.columns

data.drop(['Total.Day.Minutes', 'Total.Day.Calls', 'Total.Day.Charge', 'Total.Eve.Calls', 'Total.Eve.Minutes','Total.Night.Minutes', 'Total.Night.Calls',
       'Total.Intl.Minutes', 'Total.Intl.Calls', 'trainrows'], axis=1, inplace=True)

data.head()

data.dtypes

data.isnull().sum()

numeric_list=['Total.Eve.Charge','Total.Night.Charge','Total.Intl.Charge','X..Vmail.Messages']
categorical_list=['International.Plan','Voice.Mail.Plan','X..customer.Service.Calls']

len(numeric_list)

len(categorical_list)

data = pd.get_dummies(columns=categorical_list, data = data, prefix=categorical_list, prefix_sep="_")
data.head()

X=data.drop(["Churn"],axis=1)
y=data["Churn"]

X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.30, random_state=123, stratify=y)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

"""Standard Scaler : Independent Numberic Attributes"""

scaler = StandardScaler()
X_train[numeric_list] = scaler.fit_transform(X_train[numeric_list])
X_test[numeric_list]=scaler.transform(X_test[numeric_list])

X_train.head()

y_train.head()

"""model building"""

logistic_model = LogisticRegression()
logistic_model.fit(X_train,y_train)

train_preds = logistic_model.predict(X_train)
train_preds_prob=logistic_model.predict_proba(X_train)
test_preds = logistic_model.predict(X_test)
test_preds_prob=logistic_model.predict_proba(X_test)

train_preds

logistic_model.coef_

confusion_matrix(y_train,train_preds)

confusion_matrix(y_test,test_preds)

classification_report_train_test(y_train, train_preds, y_test, test_preds)

logistic_model = LogisticRegression(class_weight="balanced")
logistic_model.fit(X_train,y_train)

train_preds = logistic_model.predict(X_train)
train_preds_prob=logistic_model.predict_proba(X_train)
test_preds = logistic_model.predict(X_test)
test_preds_prob=logistic_model.predict_proba(X_test)

train_preds

logistic_model.coef_

confusion_matrix(y_train,train_preds)

confusion_matrix(y_test,test_preds)

classification_report_train_test(y_train, train_preds, y_test, test_preds)

logistic_model = LogisticRegression(class_weight="balanced",multi_class="ovr",max_iter=124)
logistic_model.fit(X_train,y_train)

train_preds = logistic_model.predict(X_train)
train_preds_prob=logistic_model.predict_proba(X_train)
test_preds = logistic_model.predict(X_test)
test_preds_prob=logistic_model.predict_proba(X_test)

train_preds

logistic_model.coef_

confusion_matrix(y_train,train_preds)

confusion_matrix(y_test,test_preds)

classification_report_train_test(y_train, train_preds, y_test, test_preds)

logistic_model = LogisticRegression(penalty="l1",solver="liblinear")
logistic_model.fit(X_train,y_train)

train_preds = logistic_model.predict(X_train)
train_preds_prob=logistic_model.predict_proba(X_train)
test_preds = logistic_model.predict(X_test)
test_preds_prob=logistic_model.predict_proba(X_test)

train_preds

logistic_model.coef_

X_train.columns

confusion_matrix(y_train,train_preds)

confusion_matrix(y_test,test_preds)

classification_report_train_test(y_train, train_preds, y_test, test_preds)

logistic_model = LogisticRegression(penalty="l1",solver="liblinear",class_weight="balanced",multi_class="ovr",max_iter=124)
logistic_model.fit(X_train,y_train)

train_preds = logistic_model.predict(X_train)
train_preds_prob=logistic_model.predict_proba(X_train)
test_preds = logistic_model.predict(X_test)
test_preds_prob=logistic_model.predict_proba(X_test)

train_preds

logistic_model.coef_

confusion_matrix(y_train,train_preds)

confusion_matrix(y_test,test_preds)

classification_report_train_test(y_train, train_preds, y_test, test_preds)

"""Binary Classification using SVM"""

data2=copy.copy(train)
data2.head()

X=data2.drop(["Churn"],axis=1)
y=data2["Churn"]

from sklearn.model_selection import train_test_split
X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, stratify= y, test_size = 0.3, random_state=124)

print(X_train1.shape)
print(X_test1.shape)
print(y_train1.shape)
print(y_test1.shape)

# Standardizing the numeric attributes in the train and test data:
from sklearn.preprocessing import StandardScaler,MinMaxScaler,OneHotEncoder

s= StandardScaler()
s.fit(X_train1[num_attr])

X_train_num1= pd.DataFrame(s.transform(X_train1[num_attr]), columns=num_attr)
X_test_num1= pd.DataFrame(s.transform(X_test1[num_attr]), columns=num_attr)

X_train_num1.head()

o = OneHotEncoder()

o.fit(X_train1[cat_attr])

columns_ohe1 = list(ohe.get_feature_names(cat_attr))
print(columns_ohe1)

X_train_cat1= ohe.transform(X_train1[cat_attr])
X_test_cat1 = ohe.transform(X_test1[cat_attr])

X_train_cat1 = pd.DataFrame(X_train_cat1.todense(), columns=columns_ohe1)
X_test_cat1  = pd.DataFrame(X_test_cat1.todense(), columns=columns_ohe1)

X_train_cat1

X_train1 = pd.concat([X_train_num1, X_train_cat1], axis=1)
X_test1 = pd.concat([X_test_num1, X_test_cat1], axis=1)

print(X_train1.shape, X_test1.shape)

X_train1.head()

X_test1.head()

"""MODEL BUILDING

A. SVM (Linear and RBF Models)

Create a SVC classifier using a linear kernel
"""

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

linear_svm = SVC(kernel="linear",C=1)

linear_svm.fit(X=X_train1,y=y_train1)

train_predictions = linear_svm.predict(X_train1)
test_predictions = linear_svm.predict(X_test1)

def evaluate_model(act, pred):
    print("Confusion Matrix \n", confusion_matrix(act, pred))
    print("Accurcay : ", accuracy_score(act, pred))
    print("Recall   : ", recall_score(act, pred))
    print("Precision: ", precision_score(act, pred))
    print("F1_score : ", f1_score(act, pred))

### Train data accuracy
evaluate_model(y_train1, train_predictions)

### Test data accuracy
evaluate_model(y_test1, test_predictions)

linear_svm1= SVC(C=1000.0,kernel='linear',degree=3,gamma=0.1)

linear_svm1.fit(X=X_train1,y=y_train1)

train_predictions1 = linear_svm1.predict(X_train1)
test_predictions1 = linear_svm1.predict(X_test1)

### Train data accuracy
evaluate_model(y_train1, train_predictions1)

### Test data accuracy
evaluate_model(y_test1, test_predictions1)

"""Non Linear SVM (RBF)

Create an SVC object
"""

svc = SVC(kernel="rbf",gamma=0.01,C=10)
svc

svc.fit(X=X_train1, y=y_train1)

train_predictions2= svc.predict(X_train1)
test_predictions2= svc.predict(X_test1)

### Train data accuracy
evaluate_model(y_train1, train_predictions2)

### Test data accuracy
evaluate_model(y_test1, test_predictions2)

"""SVM with Grid Search for Paramater Tuning

Define param and instantiate GridSearchCV
"""

svc_grid = SVC()
 
param_grid = { 
                'C': [0.001, 0.01, 0.1, 1, 10,15,21,25 ,100,200 ],
                'gamma': [0.0001, 0.001, 0.01, 0.1, 1, 10,15, 25, 100], 
                'kernel':['linear','rbf','poly']
             }

svc_cv_grid = GridSearchCV(estimator = svc_grid, param_grid=param_grid, cv =3)

"""Fit the grid search model"""

svc_cv_grid.fit(X=X_train1, y=y_train1)

svc_cv_grid.best_params_

svc1 = SVC(kernel="rbf",gamma=0.01,C=100)
svc1

svc1.fit(X=X_train1, y=y_train1)

train_predictions3= svc1.predict(X_train1)
test_predictions3= svc1.predict(X_test1)

### Train data accuracy
evaluate_model(y_train1, train_predictions3)

### Test data accuracy
evaluate_model(y_test1, test_predictions3)

"""Model building

Decision Tree Model
"""

from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.metrics import confusion_matrix,accuracy_score,recall_score,precision_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import graphviz
import math

dtclf = DecisionTreeClassifier()

dtclf.fit(X_train1, y_train1)

importances = dtclf.feature_importances_
importances

indices = np.argsort(importances)[::-1]
ind_attr_names = X_train1.columns
pd.DataFrame([ind_attr_names[indices], np.sort(importances)[::-1]])

dtclf.classes_

# Decision Tree Graph explanation
dtclf2 = DecisionTreeClassifier(max_depth=2) 
dtclf2.fit(X_train1, y_train1)
dot_data2 = export_graphviz(dtclf2, 
                           feature_names=ind_attr_names,
                           class_names=['0', '1'], 
                           filled=True) 

graph2 = graphviz.Source(dot_data2) 
graph2

"""Predict"""

train_pred1 = dtclf.predict(X_train1)
test_pred1= dtclf.predict(X_test1)

### Train data accuracy
evaluate_model(y_train1, train_pred1)

### Test data accuracy
evaluate_model(y_test1, test_pred1)

"""SMOTE"""

smote = SMOTE(random_state=123)

X_train_sm, y_train_sm = smote.fit_resample(X_train1, y_train1)

X_train_sm.shape

y_train_sm.shape

print(pd.value_counts(y_train_sm, normalize=True)*100)

"""Decision Tree with up-sample data"""

dtclf2 = DecisionTreeClassifier()

dtclf2 = dtclf2.fit(X_train_sm, y_train_sm)

importances = dtclf2.feature_importances_
importances

indices = np.argsort(importances)[::-1]
pd.DataFrame([ind_attr_names[indices], np.sort(importances)[::-1]])

"""Predict"""

train_pred2=dtclf2.predict(X_train_sm)
test_pred2=dtclf2.predict(X_test1)

### Train data accuracy
evaluate_model(y_train_sm, train_pred2)
### Test data 
evaluate_model(y_test1, test_pred2)

"""Hyper-parameter tuning using Grid Search and Cross Validation"""

param_grid = {"criterion": ["gini", "entropy"],
              "min_samples_split": [2, 5],
              "max_depth": [None, 2],
              "min_samples_leaf": [1, 5]
             }

"""Instantiate Decision Tree"""

dtclf3 = DecisionTreeClassifier()

"""GridSearchCV"""

dtclf_grid = GridSearchCV(dtclf3,param_grid,cv=3)

"""Train"""

dtclf_grid.fit(X_train_sm, y_train_sm)

dtclf_grid.best_params_

"""Predict"""

train_pred3= dtclf_grid.predict(X_train_sm)
test_pred3 = dtclf_grid.predict(X_test1)

### Train data accuracy
evaluate_model(y_train_sm, train_pred3)
### Test data 
evaluate_model(y_test1, test_pred3)

"""
Building Decision Tree Model using Variable Importance"""

importances = dtclf_grid.best_estimator_.feature_importances_
importances

indices = np.argsort(importances)[::-1]
print(indices)

select = indices[0:5]
print(select)

"""Instantiate Model"""

dtclf3 = DecisionTreeClassifier(criterion='entropy',
 max_depth=None,
 min_samples_leaf=1,
 min_samples_split= 5)

"""Train the model"""

dtclf3 = dtclf3.fit(X_train_sm.values[:,select], y_train_sm)

"""Predict"""

train_pred4 = dtclf3.predict(X_train_sm.values[:,select])
test_pred4= dtclf3.predict(X_test1.values[:,select])

### Train data accuracy
evaluate_model(y_train_sm, train_pred4)
### Test data 
evaluate_model(y_test1, test_pred4)

param_grid1= {"criterion": ["gini", "entropy"],
              "min_samples_split": [1, 8],
              "max_depth": [None, 2],
              "min_samples_leaf": [4,5]
             }

dtclf4= DecisionTreeClassifier()

dtclf_grid1 = GridSearchCV(dtclf4,param_grid1,cv=5)

dtclf_grid1.fit(X_train_sm, y_train_sm)

dtclf_grid1.best_params_

train_pred5 = dtclf_grid1.predict(X_train_sm)
test_pred5= dtclf_grid1.predict(X_test1)

### Train data accuracy
evaluate_model(y_train_sm, train_pred5)
### Test data 
evaluate_model(y_test1, test_pred5)

importances1 = dtclf_grid1.best_estimator_.feature_importances_
importances1

indices1 = np.argsort(importances1)[::-1]
print(indices1)

select1 = indices[0:19]
print(select1)

dtclf4 = DecisionTreeClassifier(criterion= 'entropy',
 max_depth=None,
 min_samples_leaf= 5,
 min_samples_split=8)

dtclf4 = dtclf4.fit(X_train_sm.values[:,select1], y_train_sm)

train_pred6 = dtclf4.predict(X_train_sm.values[:,select1])
test_pred6= dtclf4.predict(X_test1.values[:,select1])

### Train data accuracy
evaluate_model(y_train_sm, train_pred6)
### Test data 
evaluate_model(y_test1, test_pred6)

"""RandomForestClassifier with up-sample data"""

from sklearn.ensemble import RandomForestClassifier

clf2=RandomForestClassifier()

"""Train the model"""

clf2.fit(X_train_sm, y_train_sm)

importances2 = clf2.feature_importances_
print(importances2)

indices2= np.argsort(importances2)[::-1]
print(indices2)

pd.DataFrame([ind_attr_names[indices2], np.sort(importances2)[::-1]])

"""Predict"""

train_pred7 = clf2.predict(X_train_sm)
test_pred7 = clf2.predict(X_test1)

### Train data accuracy
evaluate_model(y_train_sm, train_pred7)
### Test data 
evaluate_model(y_test1, test_pred7)

"""Hyper-parameter tuning using Grid Search and Cross Validation

Parameters to test
"""

param_grid2= {"n_estimators" : [50, 100],
              "max_depth" : [1,5],
              "max_features" : [3, 5],
              "min_samples_leaf" : [1, 2, 4]}

"""Instantiate Decision Tree"""

clf3 = RandomForestClassifier()

"""Instantiate GridSearchCV"""

clf_grid = GridSearchCV(clf3,param_grid2,cv=2)

"""Train DT using GridSearchCV"""

clf_grid.fit(X_train_sm, y_train_sm)

"""Best Params"""

clf_grid.best_params_

train_pred8= clf_grid.predict(X_train_sm)
test_pred8= clf_grid.predict(X_test1)

### Train data accuracy
evaluate_model(y_train_sm, train_pred8)
### Test data 
evaluate_model(y_test1, test_pred8)

param_grid3= {"n_estimators" : [50,100,150,200],
              "max_depth" : [None,2],
              "max_features" : [3,5],
              "min_samples_leaf" : [2,4]}

clf4 = RandomForestClassifier()

clf_grid1= GridSearchCV(clf4,param_grid3,cv=3)

clf_grid1.fit(X_train_sm, y_train_sm)

clf_grid1.best_params_

train_pred9= clf_grid1.predict(X_train_sm)
test_pred9= clf_grid1.predict(X_test1)

### Train data accuracy
evaluate_model(y_train_sm, train_pred9)
### Test data 
evaluate_model(y_test1, test_pred9)

