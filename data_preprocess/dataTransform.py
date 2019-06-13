import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import time
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import random


def date_intervals1(date):
    start=datetime.datetime.strptime(date,'%Y/%m/%d')
    timestamp=time.time()
    end=time.strftime('%Y/%m/%d', time.localtime(timestamp)) #转换为时间结构体转换为时间字符串
    end=datetime.datetime.now()
    diff=end-start
    return diff.days


def parse_strdate(date):
    if date is not np.nan:
        date = datetime.datetime.strptime(date, '%Y/%m/%d')
        return date


def to_int(days):
    if  np.isnan(days):
        return np.nan
    else:
        return int(days)


def to_target(x):
    if x<0.5:
        return 0
    elif x>=0.5 and x<=0.9:
        return 1
    elif x>0.9:
        return 2
    else:
        return -1







#将数据进行标准化处理
def standardlization(series):
    mean=series.mean()
    std=series.std()
    for i in range(len(series)):
        if np.isnan(series[i]):
            series[i]=np.nan
        else:
            series[i] = (series[i] - mean) / std
        print(i)
    return series



df1=pd.read_csv(filepath_or_buffer='../data/air_data(complete).csv',encoding='utf-8')
df2=pd.read_csv(filepath_or_buffer='../data/air_data(incomplete).csv',encoding='utf-8')



#对时间类型Date进行处理，计算出时间间隔
df1['Days_to_FFP_DATE']=df1['LOAD_TIME'].apply(lambda x:parse_strdate(x))-df1['FFP_DATE'].apply(lambda x:parse_strdate(x))
df1['Days_to_FFP_DATE']=df1['Days_to_FFP_DATE'].apply(lambda x:x.days) #获取间隔天数
df1['Days_to_FIRST_FLIGHT_DATE']=df1['LOAD_TIME'].apply(lambda x:parse_strdate(x))-df1['FIRST_FLIGHT_DATE'].apply(lambda x:parse_strdate(x))
df1['Days_to_FIRST_FLIGHT_DATE']=df1['Days_to_FIRST_FLIGHT_DATE'].apply(lambda x:x.days)
df1['target']=df1['L1Y_Flight_Count']/df1['P1Y_Flight_Count'].apply(lambda x:to_target(x))
temp1=df1.drop('MEMBER_NO',axis=1)
temp1=temp1.drop('FFP_DATE',axis=1)
temp1=temp1.drop('FIRST_FLIGHT_DATE',axis=1)
temp1=temp1.drop('GENDER',axis=1)
temp1=temp1.drop('target',axis=1)
temp1=temp1.drop('LOAD_TIME',axis=1)
temp1=temp1.drop('WORK_CITY',axis=1)
temp1=temp1.drop('WORK_PROVINCE',axis=1)
temp1=temp1.drop('WORK_COUNTRY',axis=1)
temp1=temp1.drop('LAST_FLIGHT_DATE',axis=1)


#对数字型属性进行归一化处理
scaler = MinMaxScaler()
# min_max_scaler =MinMaxScaler()
temp1=scaler.fit_transform(temp1)
joblib.dump(scaler,'../model/scaler.pkl')
temp1=pd.DataFrame(temp1)



#将class标签进行onehot编码
# one_hot = OneHotEncoder()
# df1['class']=one_hot.fit_transform(np.array(df1['FFP_TIER']).reshape(-1,1))
# joblib.dump(one_hot,'../model/OneHotEncoder.pkl')
df1['class']=df1['target']



#将性别转为-1和1
mapping = {"男":-1,"女":1}
df1['GENDER']=df1['GENDER'].map(mapping)




#将字符型类别变量数字化，并进行归一化。
labelEncoder1=LabelEncoder()
min_max_scaler1 =MinMaxScaler()
temp=pd.concat([df1['WORK_CITY'],df2['WORK_CITY']],axis=0,ignore_index=True)
temp=temp.fillna('NAN')
temp=labelEncoder1.fit_transform(temp.values)
joblib.dump(labelEncoder1,'../model/labelEncoder1.pkl')
df1['WORK_CITY']=labelEncoder1.transform(df1['WORK_CITY'])
min_max_scaler1.fit(temp.reshape(-1,1))#.values将df转为np数组
joblib.dump(min_max_scaler1,'../model/min_max_scaler1.pkl')
df1['WORK_CITY']=min_max_scaler1.transform(df1['WORK_CITY'].values.reshape(-1,1))




#将字符型类别变量工作省份数字化，并进行归一化。
labelEncoder2=LabelEncoder()
min_max_scaler2 =MinMaxScaler()
temp=pd.concat([df1['WORK_PROVINCE'],df2['WORK_PROVINCE']],axis=0)
temp=temp.fillna('NAN')
temp=labelEncoder2.fit_transform(temp.values)
joblib.dump(labelEncoder2,'../model/labelEncoder2.pkl')
df1['WORK_PROVINCE']=labelEncoder2.transform(df1['WORK_PROVINCE'])
min_max_scaler2.fit(temp.reshape(-1,1))#.values将df转为np数组
joblib.dump(min_max_scaler2,'../model/min_max_scaler2.pkl')
df1['WORK_PROVINCE']=min_max_scaler2.transform(df1['WORK_PROVINCE'].values.reshape(-1,1))



#将字符型类别变量工作国家数字化，并进行归一化。
labelEncoder3=LabelEncoder()
min_max_scaler3 =MinMaxScaler()
temp=pd.concat([df1['WORK_COUNTRY'],df2['WORK_COUNTRY']],axis=0)
temp=temp.fillna('NAN')
temp=labelEncoder3.fit_transform(temp.values)
joblib.dump(labelEncoder3,'../model/labelEncoder3.pkl')
df1['WORK_COUNTRY']=labelEncoder3.transform(df1['WORK_COUNTRY'])
min_max_scaler3.fit(temp.reshape(-1,1))#.values将df转为np数组
joblib.dump(min_max_scaler3,'../model/min_max_scaler3.pkl')
df1['WORK_COUNTRY']=min_max_scaler3.transform(df1['WORK_COUNTRY'].values.reshape(-1,1))



newdf=pd.concat([df1['MEMBER_NO'],temp1,df1['WORK_CITY'],df1['WORK_PROVINCE'],df1['WORK_COUNTRY'],df1['GENDER'],df1['class']],axis=1,ignore_index=False)
newdf.to_csv(path_or_buf='../data/validate_set.csv',index=False,encoding='utf-8')

























