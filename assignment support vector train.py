# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 15:11:02 2022

@author: vaishnav
"""


#importing the data

import pandas as pd
from sklearn import metrics
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("C:\\anaconda\\New folder (2)\\SalaryData_Train.csv")
df
list(df)

df.info()
df.describe()

df.dtypes
#===========================================================================================================
#Finding the special characters in the data frame 

df.isin(['?']).sum(axis=0)
print(df[0:5])


df.native.value_counts()
df.native.unique()


df.workclass.value_counts()
df.workclass.unique()


df.occupation.value_counts()
df.occupation.unique()


df.sex.value_counts()

#===============================================================================================================

#visualisation
import seaborn as sns
import matplotlib.pyplot as plt
sns.pairplot(df)

t1 = pd.crosstab(index=df["education"],columns=df["workclass"])
t1.plot(kind='bar')

t2 = pd.crosstab(index=df["education"],columns=df["Salary"])
t2.plot(kind='bar')


t3 = pd.crosstab(index=df["sex"],columns=df["race"])
t3.plot(kind='bar')


t4 = pd.crosstab(index=df["maritalstatus"],columns=df["sex"])
t4.plot(kind='bar')

df["age"].hist()
df["educationno"].hist()
df["capitalgain"].hist()
df["capitalloss"].hist()
df["hoursperweek"].hist()


# Check Correlation amoung parameters
corr = df.corr()
fig, ax = plt.subplots(figsize=(8,8))
# Generate a heatmap
sns.heatmap(corr, cmap = 'magma', annot = True, fmt = ".2f")
plt.xticks(range(len(corr.columns)), corr.columns)

plt.yticks(range(len(corr.columns)), corr.columns)

plt.show()

sn = sns.jointplot(x = 'age', 
              y = 'hoursperweek',
              data = df, 
              kind = 'hex', 
              cmap= 'hot', 
              size=10)


sns.regplot(df.age, df['hoursperweek'], ax=sn.ax_joint, scatter=False, color='grey')



#==================================================================================================================================

from sklearn.preprocessing import LabelEncoder

df = df.apply(LabelEncoder().fit_transform)
df.head()

#==================================================================================================================================

drop_elements = ['education', 'native', 'Salary']

X = df.drop(drop_elements, axis=1)

y = df['Salary']

#==================================================================================================================================

#Data partition 

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=31)


#==================================================================================================================================
#support Vector mission

svc = SVC()
svc.fit(X_train, y_train)
# make predictions
prediction = svc.predict(X_test)
# summarize the fit of the model
print(metrics.classification_report(y_test, prediction))
print(metrics.confusion_matrix(y_test, prediction))

print("Accuracy:",metrics.accuracy_score(y_test, prediction))
print("Precision:",metrics.precision_score(y_test, prediction))
print("Recall:",metrics.recall_score(y_test, prediction))


#==================================================================================================================================


svc = SVC(kernel='rbf',gamma=12, C=1)
svc.fit(X_train, y_train)
# make predictions
prediction = svc.predict(X_test)
# summarize the fit of the model
print(metrics.classification_report(y_test, prediction))
print(metrics.confusion_matrix(y_test, prediction))

print("Accuracy:",metrics.accuracy_score(y_test, prediction))
print("Precision:",metrics.precision_score(y_test, prediction))
print("Recall:",metrics.recall_score(y_test, prediction))

#==================================================================================================================================

svc = SVC(kernel='poly',degree=3,gamma="scale")
svc.fit(X_train, y_train)
# make predictions
prediction = svc.predict(X_test)
# summarize the fit of the model
print(metrics.classification_report(y_test, prediction))
print(metrics.confusion_matrix(y_test, prediction))

print("Accuracy:",metrics.accuracy_score(y_test, prediction))
print("Precision:",metrics.precision_score(y_test, prediction))
print("Recall:",metrics.recall_score(y_test, prediction))


#==================================================================================================================================

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

logreg.fit(X_train,y_train)

#prediction
y_pred_test = logreg.predict(X_test)

print(metrics.classification_report(y_test, prediction))
print(metrics.confusion_matrix(y_test, prediction))

print("Accuracy:",metrics.accuracy_score(y_test, prediction))
print("Precision:",metrics.precision_score(y_test, prediction))
print("Recall:",metrics.recall_score(y_test, prediction))

#==================================================================================================================================

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train,y_train)

y_pred_test = classifier.predict(X_test)

print(metrics.classification_report(y_test, prediction))
print(metrics.confusion_matrix(y_test, prediction))

print("Accuracy:",metrics.accuracy_score(y_test, prediction))
print("Precision:",metrics.precision_score(y_test, prediction))
print("Recall:",metrics.recall_score(y_test, prediction))

#==================================================================================================================================




