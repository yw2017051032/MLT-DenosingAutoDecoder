from fancyimpute import KNN
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
import fancyimpute
missing_rate=0.1
df1=pd.read_csv(filepath_or_buffer='../data/training_setmissing_rate_'+str(missing_rate)+'missing_features20.csv',encoding='utf-8',chunksize=10000)
df2=pd.read_csv(filepath_or_buffer='../data/validate_set.csv',encoding='utf-8')

result=pd.DataFrame()
for d1 in df1:
    #去掉属性'MEMBER_NO,class'
    d1=d1.drop(['MEMBER_NO','class'],axis=1)
    #记录下缺失值位置,返回类型为二元组，分别记录行号（由numpy数组表示）和列号（由numpy数组表示）
    loc1=np.where(np.isnan(d1))
    #初始用KNN填补缺失值
    d1= KNN(k=3).complete(d1)
    d1=pd.DataFrame(d1)
    result =result.append(d1,ignore_index=True)




x_test=df2.drop(['MEMBER_NO','class'],axis=1)
x_test=np.array(x_test)


# df=df1
# df=df.drop(['MEMBER_NO','class'],axis=1)
# df=df.fillna(df.mean())
# print('均值填补_rmse:',str(np.sqrt(metrics.mean_squared_error(x_test,np.array(df)))))

print('模型imputation_rmse:',str(np.sqrt(metrics.mean_squared_error(x_test,np.array(result)))))



result.to_csv(path_or_buf='../imputed-data/imputed-data-KNN-'+str(missing_rate)+'.csv',index=False,encoding='utf-8')

