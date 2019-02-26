import fancyimpute
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
    Imputer =fancyimpute.MICE(n_imputations=3)
    d1 = Imputer.complete(d1)

    # #初始用MIC填补缺失值
    # XY_completed = []
    # for i in range(5):
    #     Imputer=IterativeImputer(sample_posterior=True, random_state=i)
    #     d1=Imputer.fit_transform(d1)
    #     XY_completed.append(d1)
    # XY_completed_mean = np.mean(XY_completed, 0)
    d1=pd.DataFrame(d1)
    result =result.append(d1,ignore_index=True)




x_test=df2.drop(['MEMBER_NO','class'],axis=1)
x_test=np.array(x_test)


# df=pd.read_csv(filepath_or_buffer='../data/training_setmissing_rate_0.6missing_features20.csv',encoding='utf-8')
# df=df.drop(['MEMBER_NO','class'],axis=1)
# df=df.fillna(df.mean())
# print('均值填补_rmse:',str(np.sqrt(metrics.mean_squared_error(x_test,np.array(df)))))

print('模型imputation_rmse:',str(np.sqrt(metrics.mean_squared_error(x_test,np.array(result)))))

result.to_csv(path_or_buf='../imputed-data/imputed-data-MIC-'+str(missing_rate)+'.csv',index=False,encoding='utf-8')