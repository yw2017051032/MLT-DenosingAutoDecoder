from fancyimpute import KNN
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
import fancyimpute
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


df1=pd.read_csv(filepath_or_buffer='../data/test_setmissing_rate_0.5missing_features20.csv',encoding='utf-8')
X=df1.drop(['MEMBER_NO','class'],axis=1)
Y=df1['class']

# df2=pd.read_csv(filepath_or_buffer='../data/validate_set.csv',encoding='utf-8')


#初始用KNN填补缺失值
X=KNN(k=3).complete(X)


clf=LogisticRegression()

kf = StratifiedKFold(n_splits=10)

result=[]
for train_index, test_index in kf.split(X, Y):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    clf.fit(X_train,Y_train)
    score=clf.score(X_test, Y_test)
    result.append(score)
    print("score:%f" % (score))

print("平均score:%f",sum(result)/len(result))



