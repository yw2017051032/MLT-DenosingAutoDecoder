import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB

clf=LogisticRegression()
clf2 = GaussianNB()

modelname='MLTDAE'
missing_rate=0.6
df1=pd.read_csv(filepath_or_buffer='../imputed-data/imputed-data-'+modelname+'-'+str(missing_rate)+'.csv',encoding='utf-8')
df2=pd.read_csv(filepath_or_buffer='../data/validate_set.csv',encoding='utf-8')


X=np.array(df1)
Y=np.array(df2['class'])

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

# sum(result[1:])/len(result[1:])