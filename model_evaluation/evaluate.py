import pandas as pd
import numpy as np
from sklearn import metrics

def my_mean_square_error(src,des):
    if not isinstance(src,pd.core.frame.DataFrame) or not isinstance(des,pd.core.frame.DataFrame):
        raise TypeError('原数据集必须为DataFrame类型')
    (src_rows,src_columns)=src.shape
    (des_rows,des_columns)=des.shape
    if src_rows==des_rows and src_columns==des_columns:
        sum1=0;
        for i in range(src_rows):
            sum2=0;
            for j in range(src_columns):
                x_obs=src[i][j]
                x_hat=des[i][j]
                temp=x_obs-x_hat
                sum2+=temp
            sum1+=sum2

        return sum1

def  mean_square_error(src,des):
    if not isinstance(src, pd.core.frame.DataFrame) or not isinstance(des, pd.core.frame.DataFrame):
        raise TypeError('原数据集必须为DataFrame类型')
    (src_rows, src_columns) = src.shape
    (des_rows, des_columns) = des.shape
    if src_rows == des_rows and src_columns == des_columns:
      return  metrics.mean_squared_error(src,des)


def  root_mean_square_error(src,des):
    if not isinstance(src, pd.core.frame.DataFrame) or not isinstance(des, pd.core.frame.DataFrame):
        raise TypeError('原数据集必须为DataFrame类型')
    (src_rows, src_columns) = src.shape
    (des_rows, des_columns) = des.shape
    if src_rows == des_rows and src_columns == des_columns:
      return  np.sqrt(metrics.mean_squared_error(src,des))





