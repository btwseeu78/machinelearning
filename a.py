import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1)
import statsmodels.api as sm
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
%matplotlib inline
train = pd.read_csv(r'C:/Users/achatte2/Documents/PythonScripts/train.csv')
test = pd.read_csv(r'C:/Users/achatte2/Documents/PythonScripts/test.csv')
full_data = train + test
train.head()
test.head()
train.isnull().sum()
test.isnull().sum()
train.groupby(by='Survived').mean()
train.Pclass.hist(by='Survived',bins=20,figsize=(10,12))
train.Pclass.hist('Survived',bins=20,figsize=(10,12))
train.Pclass.hist(bins=20,figsize=(10,12))
clear
pd.crosstab(train.Pclass,train.Survived.astype(bool)).plot(kind='bar')
train.corr()
train.Age.fillna(np.mean(train.Age),inplace=True)
test.Age.fillna(np.mean(train.Age),inplace=True)
train.describe
train.describe()
from scipy import interpolate
train.interpolate(inplace=True)
train.head()
test.isnull().sum()
test.Fare.fillna(np.mean(test.Fare),inplace=True)
train.columns.count()
train.columns
test.columns
y,X = dmatrices('Survived ~ Pclass + female + Age',train,return_type="dataframe")
y,X = dmatrices('Survived ~ Pclass +  Age',train,return_type="dataframe")
y.head()
 X.head()
y = np.ravel(y)
model = LogisticRegression()
model.fit(X,y)
model.score(X,y)
y.columns
y.columns()
y
clear
pred_train = model.predict(X)
pred_train.head()
pred_train
metrics.accuracy_score(y,pred_train[:,1])
metrics.accuracy_score(y,pred_train)
x_test = test
x_test
X
x_test
X
X.info
X.info()
X.head()
test.head()
test3 = test["Pclass,Age,Fare"]
test3 = test['Pclass','Age','Fare']
test.head()
pred = model.predict(test)
test.drop(labels=['PassengerId','SibSp' ,'Parch','Ticket','Cabin','Embarked'])
test.drop(labels=['PassengerId','SibSp' ,'Parch','Ticket','Cabin','Embarked'] axis=1,inplace=True)
test.drop(labels=['PassengerId','SibSp' ,'Parch','Ticket','Cabin','Embarked'],axis=1,inplace=True)
test.head()
test.drop("Name",axis=1,inplace=True)
pred_Test = model.predict(test)
test.head()
X
X.head()
y
y.head()
y.size
pred_Test = model.predict(test)
train_sex = pd.get_dummies(train['Sex'])
train2 = pd.concat([train, train_sex], axis=1)
test_sex = pd.get_dummies(test['Sex'])
test2 = pd.concat([test, test_sex], axis=1)
train3 = train2[['Survived', 'Pclass', 'female', 'Age', 'Family', 'Embarked']]
test3 = test2[['Pclass', 'female', 'Age', 'Family', 'Embarked']]
train['Family'] = train['SibSp'] + train['Parch'] + 1
test['Family'] = test['SibSp'] + test['Parch'] + 1
train3 = train2[['Survived', 'Pclass', 'female', 'Age', 'Family', 'Embarked']]
test3 = test2[['Pclass', 'female', 'Age', 'Family', 'Embarked']]

train3 = train2[['Survived', 'Pclass', 'female', 'Age', 'Family', 'Embarked']]
test3 = test2[['Pclass', 'female', 'Age', 'Family', 'Embarked']]

train3 = train2[['Survived', 'Pclass', 'female', 'Age','Embarked']]
test3 = test2[['Pclass', 'female', 'Age', 'Embarked']]

train3 = train2[['Survived', 'Pclass', 'female', 'Age',]]
test3 = test2[['Pclass', 'female', 'Age']]
, X = dmatrices('Survived ~ Pclass + female + Age,
                  train3, return_type="dataframe")
y, X = dmatrices('Survived ~ Pclass + female + Age',train3, return_type="dataframe")
y=np.ravel(y)
model =model.fit(X,y)
model.score(X,y)
model.predict(test3)
test.head()
model.predict(test3[:,1:])
test.columns
X.columns()
X
X/head()
X.head()
X.columns()
model.predict(test3)
model.predict(test3[:,1])
test3.head()
test3.columns
X.head()
X.Intercept.value_counts()
test3['Intercept'] = 1
test3['Intercept'] = 1
test3.head()
model.predict(test3)
mod = model.predict(test3)
y_test = mod.astype(int)
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": y_test
    })
submission.to_csv('titanic.csv', index=False)
test3.head
submission = pd.DataFrame({

        "Survived": y_test
    })
submission.to_csv('titanic.csv', index=False)
%history -f a.py
