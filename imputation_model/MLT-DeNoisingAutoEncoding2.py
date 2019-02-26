from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras import layers
from keras.models import Model
from keras import backend as K
import keras
import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics

def my_loss(input,output):

    return K.mean(K.square(output-input))



df1=pd.read_csv(filepath_or_buffer='../data/training_setmissing_rate_0.6missing_features20.csv',encoding='utf-8')
df=df1
df2=pd.read_csv(filepath_or_buffer='../data/validate_set.csv',encoding='utf-8')
df3=pd.read_csv(filepath_or_buffer='../data/testing_set.csv',encoding='utf-8')


#去掉属性'MEMBER_NO'
df1=df1.drop('MEMBER_NO',axis=1)
#记录下缺失值位置,返回类型为二元组，分别记录行号（由numpy数组表示）和列号（由numpy数组表示）
loc1=np.where(np.isnan(df1))
#初始将缺失值补零
df1=df1.fillna(df1.mean())

df2=df2.drop('MEMBER_NO',axis=1)

#记录下缺失值位置,返回类型为二元组，分别记录行号（由numpy数组表示）和列号（由numpy数组表示）
loc3=np.where(np.isnan(df3))
df3=df3.drop('MEMBER_NO',axis=1)
#初始将缺失值补零
df3=df3.fillna(0)

onehot=OneHotEncoder()
x_train=df1.drop('class',axis=1)
x_train=np.array(x_train)
y_train=df1['class']
y_train=onehot.fit_transform(y_train.values.reshape(-1,1))
y_train=y_train.todense()
x_train_imputation=np.concatenate((x_train,y_train),axis=1)



x_test=df2.drop('class',axis=1)
x_test=np.array(x_test)
y_test=df2['class']
y_test=onehot.transform(y_test.values.reshape(-1,1))
y_test=y_test.todense()
x_test_imputation=np.concatenate((x_test,y_test),axis=1)


x_validate=df3.drop('class',axis=1)
x_validate=np.array(x_validate)
y_validate=df3['class']
y_validate=onehot.transform(y_validate.values.reshape(-1,1))
y_validate=y_validate.todense()
x_validate_imputation=np.concatenate((x_validate,y_validate),axis=1)


# Parameters for denoising autoencoder
input_dim1 = 40
output_dim1=40
input_dim2 = 40
output_dim2=3
nb_hidden = 15
batch_size = 256
epochs=400


classification_input=Input(shape=(input_dim2,),name='classification_input')
hidden = Dense(15, activation='sigmoid')(classification_input)
classification_output = Dense(output_dim2, activation='softmax',name='classification_output')(hidden)


# Build autoencoder model
imputation_input = Input(shape=(input_dim1,),name='imputation_input')
encoded = Dense(40+7, activation='tanh')(imputation_input)
encoded = Dense(40+14, activation='tanh')(encoded )
encoded = Dense(40+21, activation='tanh')(encoded )
x1 = keras.layers.concatenate([encoded , classification_output],axis=1)
decoded = Dense(40+14, activation='tanh')(x1)
x2 = keras.layers.concatenate([decoded , classification_output],axis=1)
decoded = Dense(40+7, activation='tanh')(x2)
# imputation_output = Dense(40, activation='tanh',name='imputation_output')(decoded )
x3 = keras.layers.concatenate([decoded , classification_output],axis=1)
imputation_output = Dense(40, activation='sigmoid',name='imputation_output')(x3)



autoencoder = Model(input=[imputation_input,classification_input], output=[imputation_output,classification_output])
autoencoder.compile(loss='binary_crossentropy',loss_weights={'imputation_output':0.5,'classification_output':0.5}, optimizer='adadelta',metrics={'imputation_output':'mse'})
# autoencoder.summary()

# imputation=Model(input=[imputation_input], output=[imputation_output])
# classification=Model(input=[classification_input], output=[classification_output])


## 加一个early_stooping
early_stopping = keras.callbacks.EarlyStopping(
    monitor='loss',
    min_delta=0.0001,
    patience=5,
    verbose=0,
    mode='auto'
)

start = time.time()
# validation_data=({'imputation_input':x_validate_imputation,
#                                      'classification_input':x_validate},{'imputation_output':x_validate,
#                                      'classification_output':y_validate})
autoencoder.fit(x={'imputation_input':x_train,'classification_input':x_train},y={'imputation_output':x_test,'classification_output':y_test},
                nb_epoch=epochs,batch_size=batch_size,shuffle=True,verbose=1,callbacks=[early_stopping],
                validation_data=({'imputation_input':x_validate,'classification_input':x_validate},
                                 {'imputation_output':x_validate,'classification_output':y_validate}))


imputation,classification=autoencoder.predict(x={'imputation_input':x_train,'classification_input':x_train})


# classification_validate=classification.predict(x={'classification_input':x_validate})
# imputation_validate=imputation.predict(x={'imputation_input':x_validate_imputation})



df=df.drop(['MEMBER_NO','class'],axis=1)
df=df.fillna(df.mean())
print('均值填补_rmse:',str(np.sqrt(metrics.mean_squared_error(x_test,np.array(df)))))
for i,j in zip(loc1[0],loc1[1]):
    x_train[i][j]=imputation[i][j]
print('模型imputation_rmse:',str(np.sqrt(metrics.mean_squared_error(x_test,x_train))))

end = time.time()


#Evaluate
# evaluation = autoencoder.evaluate(x={'imputation_input':x_validate_imputation,
#                                      'classification_input':x_validate},
#                                   y={'imputation_output':x_validate,
#                                      'classification_output':y_validate},
#                                   batch_size=batch_size, verbose=1)
# print('val_loss: %.6f, val_mean_absolute_error: %.6f' % (evaluation[0], evaluation[1]))







print('耗时：' + str((end - start) / 60))




