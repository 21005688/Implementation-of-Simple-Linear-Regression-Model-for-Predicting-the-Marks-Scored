# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries and read the dataframe.
2. Assign hours to X and scores to Y.
3. Implement training set and test set of the dataframe.
4. Plot the required graph both for test data and training data.
5. Find the values of MSE , MAE and RMSE.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Deepika.j
RegisterNumber: 212221230016

import pandas as pd
import numpy as np
df=pd.read_csv('student_scores.csv')
print(df)
df.head()
df.tail()
X=df.iloc[:,:-1].values
Y=df.iloc[:,1].values
print(X,Y)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,Y_train)

Y_pred=reg.predict(X_test)
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error , mean_squared_error

plt.scatter(X_train,Y_train,color='green')
plt.plot(X_train,reg.predict(X_train),color='purple')
plt.title(' Training set (Hours Vs Scores)')
plt.xlabel('Hours')
plt.ylabel('Scores')

plt.scatter(X_test,Y_test,color='green')
plt.plot(X_test,reg.predict(X_test),color='purple')
plt.title(' Training set (Hours Vs Scores)')
plt.xlabel('Hours')
plt.ylabel('Scores')

mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print('RMSE = ',rmse)
/*
```

## Output:

![11111111](https://user-images.githubusercontent.com/94747031/204080521-0b643f16-4810-43ac-8013-45644f1e187e.png)

![222222](https://user-images.githubusercontent.com/94747031/204080535-1d9c9626-a2d0-41ca-b3da-a9d78a5f84c6.png)

![3333333](https://user-images.githubusercontent.com/94747031/204080538-c4be203c-3c17-4356-b073-d14cb060a261.png)

![444444444](https://user-images.githubusercontent.com/94747031/204080545-9d300ab5-66f2-46cb-a15f-0d9fee75e206.png)

![55555555](https://user-images.githubusercontent.com/94747031/204080555-68bba2cd-52ef-482a-b054-c39fe6cce4fc.png)

![66666](https://user-images.githubusercontent.com/94747031/204080588-2958afd9-b768-42c7-9fa4-cc7e1c36c447.png)

![77777](https://user-images.githubusercontent.com/94747031/204080591-b7d7bb85-6e97-4c16-9db4-bb0e4db7b7c9.png)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
