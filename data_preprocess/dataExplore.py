#coding="utf-8"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random


df=pd.read_csv(filepath_or_buffer='../data/air_data.csv',encoding='utf-8')

result1=df.describe()                   #整个数据集探索性分析
result2=df.isnull().any()              #列级别的判断，只要该列有为空或者NA的元素，就为True，否则False
result3=df.isnull()                    #元素级别的判断，把对应的所有元素的位置都列出来，元素为空或者NA就显示True，否则就是False

df1=df[df.isnull().values==True]   #可以只显示存在缺失值的行列，清楚的确定缺失值的位置。
df1.to_csv(path_or_buf='../data/air_data(incomplete).csv',index=False,encoding='utf-8')

df2=df.dropna(axis=0)                   #去掉缺失记录，保留完整数据集
df2.to_csv(path_or_buf='../data/air_data(complete).csv',index=False,encoding='utf-8')









#对完整数据集模拟随机完全缺失
df3=pd.read_csv(filepath_or_buffer='../data/air_data(complete).csv',encoding='utf-8')
size=(int)(58258*0.1)
#利用Python中的randomw.sample()函数实现生产区间[A，B]范围内不重复的随机数组，表示从[A,B]间随机生成N个数，结果以列表返回
sampleList=random.sample(range(0,58258),size)
sampleList.sort()
np.random.randint(0,44,size=1)
for i in sampleList:
    df3.iloc[i,np.random.randint(1,44,size=1)]=np.nan #df.loc()函数根据元素的选取条件来选取对应的数据集，df.iloc()函数，它是基于索引位来选取数据集，0:4就是选取 0，1，2，3这四行，需要注意的是这里是前闭后开集合

df3.to_csv(path_or_buf='../data/air_data(training).csv',index=False,encoding='utf-8')#生产用于构建降噪自编码器的数据集


