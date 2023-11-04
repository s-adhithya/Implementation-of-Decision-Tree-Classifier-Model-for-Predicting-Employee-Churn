# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn
## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs

2. Anaconda – Python 3.7 Installation / Jupyter notebook
## Algorithm
1.Prepare your data

2.Define your model

3.Define your cost function

4.Define your learning rate

5.Train your model

6.Evaluate your model

7.Tune hyperparameters

8.Deploy your model

## Program:
```
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Adhithya.S
RegisterNumber: 212222240003


import pandas as pd
data=pd.read_csv("Employee.csv")

data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```
## Output:
### Initial data set:
![image](https://github.com/s-adhithya/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113497423/7f052595-4781-43a8-8cc1-d5481b889ef6)


### Data info:
![image](https://github.com/s-adhithya/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113497423/6094b1fd-012c-4ac5-8f06-68a18df8f905)

### Optimization of null values:
![image](https://github.com/s-adhithya/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113497423/b3ebf44f-bd9b-48bb-8bcc-1867192af5d3)


### Assignment of x and y values:
![image](https://github.com/s-adhithya/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113497423/0fe536db-a09b-45b0-801f-3eeaf58b0689)

![image](https://github.com/s-adhithya/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113497423/a4e613d7-5f13-427c-8944-cb7fdfcbada8)


### Converting string literals to numerical values using label encoder:
![image](https://github.com/s-adhithya/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113497423/fec335ac-cd04-4f75-8d8b-822c02199d6b)


### Accuracy:
![image](https://github.com/s-adhithya/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113497423/3e50a96b-c10f-4fb9-8e5e-53e247869eb7)


### Prediction:
![image](https://github.com/s-adhithya/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113497423/d9dbf91e-bcd1-40b7-81c4-ffb482149c8d)

## Result:
Thus the program to implement the Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.


