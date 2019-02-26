import random
import pandas as pd
import numpy as np

#对完整数据集模拟随机完全缺失
df3=pd.read_csv(filepath_or_buffer='../data/validate_set.csv',encoding='utf-8')
df3=pd.read_csv(filepath_or_buffer='../data/testing_set.csv',encoding='utf-8')
missing_rate=0.5
size=(int)(df3.shape[0]*missing_rate)
#利用Python中的randomw.sample()函数实现生产区间[A，B]范围内不重复的随机数组，表示从[A,B]间随机生成N个数，结果以列表返回
sampleList=random.sample(range(0,df3.shape[0]),size)
sampleList.sort()

# for i in sampleList:
#     df3.iloc[i,np.random.randint(1,41,size=1)]=np.nan #df.loc()函数根据元素的选取条件来选取对应的数据集，df.iloc()函数，它是基于索引位来选取数据集，0:4就是选取 0，1，2，3这四行，需要注意的是这里是前闭后开集合


for i in sampleList:
    max=21
    num=np.random.randint(1,max)
    print(i,num)
    for j in range(num):
        df3.iloc[i,np.random.randint(1,41)]=np.nan #df.loc()函数根据元素的选取条件来选取对应的数据集，df.iloc()函数，它是基于索引位来选取数据集，0:4就是选取 0，1，2，3这四行，需要注意的是这里是前闭后开集合







# df3.to_csv(path_or_buf='../data/training_set'+'missing_rate_'+str(missing_rate)+'missing_features'+str(max-1)+'.csv',index=False,encoding='utf-8')#生产用于构建降噪自编码器的训练集
df3.to_csv(path_or_buf='../data/test_set'+'missing_rate_'+str(missing_rate)+'missing_features'+str(max-1)+'.csv',index=False,encoding='utf-8')#生产用于构建降噪自编码器的训练集