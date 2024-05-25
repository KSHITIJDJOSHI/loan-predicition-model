# -*- coding: utf-8 -*-
"""loan prediction model.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1JZoqQZiulIWEZ8CNRIV9XiGAFYb5_dnS
"""

# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("train")
df.head()



df["Loan_Status"].value_counts()

y=df[df["Loan_Status"]=='Y'].sample(192)
n=df[df["Loan_Status"]=='N']

df=pd.concat([y,n],axis=0)

df

# null values in each column
#df.isnull().sum()

df["Gender"].value_counts()

# let star with filling null value by each column(gender column )
df["Gender"]=df["Gender"].fillna(np.random.choice(df["Gender"].dropna()))

# here we replace null value with mode
df["Married"]=df["Married"].fillna(df["Married"].mode()[0])

df["Dependents"].fillna(df["Dependents"].mode()[0],inplace=True)

df.Self_Employed.fillna(df["Self_Employed"].mode()[0],inplace=True)

#df["LoanAmount"].plot(kind='kde')

print(df.LoanAmount.dropna().median())
print(df.LoanAmount.mode())
print(df.LoanAmount.dropna().mean())

df["LoanAmount"].fillna(round(df.LoanAmount.mean(),0),inplace=True)

df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].mode()[0],inplace=True)

df['Credit_History'].fillna(df.Credit_History.mode()[0],inplace=True)

# all data are clean and start with scaling which convert categorical data into numerical

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

encoder=LabelEncoder()
scaler=MinMaxScaler()

#df.head()

df["Gender"]=encoder.fit_transform(df["Gender"])


df["Married"]=encoder.fit_transform(df["Married"])
df["Education"]=encoder.fit_transform(df["Education"])
df["Self_Employed"]=encoder.fit_transform(df["Self_Employed"])
df["Property_Area"]=encoder.fit_transform(df["Property_Area"])
df["Loan_Status"]=encoder.fit_transform(df["Loan_Status"])
df["Dependents"]=encoder.fit_transform(df["Dependents"])

df



df_final=df.drop(columns=["Loan_ID"])
df_final

y=df_final["Loan_Status"]
x=df_final.drop(columns=["Loan_Status"])
x

x=scaler.fit_transform(x)

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25,random_state=23,stratify=y)





#from sklearn.linear_model import LogisticRegression

#regressor=LogisticRegression()
#regressor.fit(xtrain,ytrain)
#predict=regressor.predict(xtest)

from sklearn.metrics import accuracy_score
#accuracy_score(ytest,predict)

#from sklearn.neighbors import KNeighborsClassifier

#classifier=KNeighborsClassifier()
#classifier.fit(xtrain,ytrain)
#predict1=classifier.predict(xtest)
#predict1

#accuracy_score(ytest,predict1)

from sklearn.svm import SVC
SVM=SVC(kernel='rbf',C=30,gamma='auto')
SVM.fit(xtrain,ytrain)
predict_x=SVM.predict(xtrain)

accuracy_score(ytrain,predict_x)

#from sklearn.model_selection import GridSearchCV
#clf=GridSearchCV(SVC(gamma='auto'),{"C":[1,10,20],
                                        #"kernel":['rbf','linear']},cv=5,return_train_score=False)
#clf.fit(xtrain,ytrain)
#clf.cv_results_

#clf.best_params_



from sklearn.svm import SVC
SVM=SVC()
SVM.fit(xtrain,ytrain)
predict2=SVM.predict(xtrain)

from sklearn.metrics import accuracy_score
accuracy_score(ytrain,predict2)

"""**TEST DATA**"""

#df2=pd.read_csv("/content/drive/MyDrive/test")
#df2.head()

#df2["Gender"]=df2["Gender"].fillna(np.random.choice(df2["Gender"].dropna(),))

#df2["Dependents"].fillna(df2["Dependents"].mode()[0],inplace=True)

#df2["Loan_Amount_Term"].fillna(df2["Loan_Amount_Term"].mode()[0],inplace=True)

#df2["LoanAmount"].fillna(round(df2.LoanAmount.mean(),0),inplace=True)

#df2.Self_Employed.fillna(df2["Self_Employed"].mode()[0],inplace=True)

#df2.Credit_History.fillna(df2.Credit_History.mode()[0],inplace=True)

#df2["Gender"]=encoder.fit_transform(df2["Gender"])
#df2["Married"]=encoder.fit_transform(df2["Married"])
#df2["Education"]=encoder.fit_transform(df2["Education"])
#df2["Self_Employed"]=encoder.fit_transform(df2["Self_Employed"])
##df2["Property_Area"]=encoder.fit_transform(df2["Property_Area"])
#df2["Dependents"]=encoder.fit_transform(df2["Dependents"])

#df2=df2.drop(columns=["Loan_ID"])

from sklearn.ensemble import RandomForestClassifier
RFC=RandomForestClassifier()
RFC.fit(xtrain,ytrain)
predict_rfc=RFC.predict(xtest)

from sklearn.metrics import f1_score
f1_score(ytest,predict_rfc)



#x2=scaler.fit_transform(df2)

#predict_test=SVM.predict(x2)

import pickle

pickle.dump(scaler,open('scaler.pkl','wb'))

pickle_model=pickle.load(open('scaler.pkl','rb'))

