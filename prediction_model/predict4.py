from fancyimpute import KNN
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
import fancyimpute
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from keras.models import load_model
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score


df1=pd.read_csv(filepath_or_buffer='../data/test_setmissing_rate_0.5missing_features20.csv',encoding='utf-8')
X=df1.drop(['MEMBER_NO','class'],axis=1)
X=X.fillna(X.mean())
X=np.array(X)
Y=df1['class']
# onehot=OneHotEncoder()
# Y=np.array(Y)
# Y=onehot.fit_transform(Y.reshape(-1,1))
# df2=pd.read_csv(filepath_or_buffer='../data/validate_set.csv',encoding='utf-8')

model=load_model(filepath='../model/MLTDAE0.5.h5')


imputation,classification=model.predict(x={'imputation_input':X})


result1=[]
for i in range(classification.shape[0]):
    temp=classification[i]
    max=temp[0]
    for j in range(len(temp)):
        if temp[j]>=max:
            max=temp[j]
            index=j
    if index==0:
        result1.append(4)
    elif index==1:
        result1.append(5)
    elif index==2:
        result1.append(6)


accuracy_score(y_true=df1['class'], y_pred=result1)

X=imputation
clf=LogisticRegression()
kf = StratifiedKFold(n_splits=10)
result2=[]
for train_index, test_index in kf.split(X, Y):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    clf.fit(X_train,Y_train)
    score=clf.score(X_test, Y_test)
    result2.append(score)
    print("score:%f" % (score))
print("平均score:%f",sum(result2)/len(result2))









