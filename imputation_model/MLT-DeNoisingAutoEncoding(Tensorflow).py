import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics



def batch_iter(sourceData, batch_size, num_epochs, shuffle=True):
    sourceData = np.array(sourceData)  # 将sourceData转换为array存储
    data_size = len(sourceData)
    num_batches_per_epoch = int(len(sourceData) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = sourceData[shuffle_indices]
        else:
            shuffled_data = sourceData

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)

            yield shuffled_data[start_index:end_index]


def get_batch_data(batch_size,X,Y1,Y2):

    # 从tensor列表中按顺序或随机抽取一个tensor
    input_queue = tf.train.slice_input_producer([X,Y1,Y2], shuffle=True)

    x_batch,y1_batch,y2_batch = tf.train.batch(input_queue, batch_size=batch_size, num_threads=1, capacity=64)
    return x_batch,y1_batch,y2_batch




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




learning_rate = 0.5
epochs = 100
batch_size = 256


X = tf.placeholder("float", [batch_size, 40], name="X")
Y1 = tf.placeholder("float", [batch_size, 40], name="Y1")
Y2 = tf.placeholder("float", [batch_size, 3], name="Y2")

# Define the weights for the layers
initial_shared_layer_weights1 = np.random.rand(40,40+8)
initial_shared_layer_weights2 = np.random.rand(40+8,40+16)
initial_shared_layer_weights3 = np.random.rand(40+16,40+24)
initial_shared_layer_weights4 = np.random.rand(40+24,40+16)
initial_shared_layer_weights5 = np.random.rand(40+16,40+8)
initial_shared_layer_weights6 = np.random.rand(40+8,40)

initial_Y2_layer_weights1 = np.random.rand(40+24,40)
initial_Y2_layer_weights2 = np.random.rand(40,15)
initial_Y2_layer_weights3 = np.random.rand(15,3)



shared_layer_weights1= tf.Variable(initial_shared_layer_weights1, name="share_W1", dtype="float32")
b1 = tf.Variable(np.random.rand(48),name='bias1',dtype="float32")
shared_layer_weights2= tf.Variable(initial_shared_layer_weights2, name="share_W2", dtype="float32")
b2 = tf.Variable(np.random.rand(48),name='bias2',dtype="float32")
shared_layer_weights3= tf.Variable(initial_shared_layer_weights3, name="share_W3", dtype="float32")
b3 = tf.Variable(np.random.rand(48),name='bias3',dtype="float32")
shared_layer_weights4= tf.Variable(initial_shared_layer_weights4, name="share_W4", dtype="float32")
b4 = tf.Variable(np.random.rand(48),name='bias4',dtype="float32")
shared_layer_weights5= tf.Variable(initial_shared_layer_weights5, name="share_W5", dtype="float32")
b5 = tf.Variable(np.random.rand(48),name='bias5',dtype="float32")
shared_layer_weights6= tf.Variable(initial_shared_layer_weights6, name="share_W6", dtype="float32")
b6 = tf.Variable(np.random.rand(48),name='bias6',dtype="float32")

Y2_layer_weights1 = tf.Variable(initial_Y2_layer_weights1, name="share_Y1_1", dtype="float32")
Y2_layer_weights2 = tf.Variable(initial_Y2_layer_weights2, name="share_Y1_2", dtype="float32")
Y2_layer_weights3 = tf.Variable(initial_Y2_layer_weights3, name="share_Y1_3", dtype="float32")





# Construct the Layers with RELU Activations
shared_layer1 = tf.nn.tanh(tf.matmul(X,shared_layer_weights1)+b1)
shared_layer2 = tf.nn.tanh(tf.matmul(shared_layer1,shared_layer_weights2)+b2)
shared_layer3 = tf.nn.tanh(tf.matmul(shared_layer2,shared_layer_weights3)+b3)
shared_layer4 = tf.nn.tanh(tf.matmul(shared_layer3,shared_layer_weights4)+b4)
shared_layer5 = tf.nn.tanh(tf.matmul(shared_layer4,shared_layer_weights5)+b5)
Y1_layer = tf.nn.sigmoid(tf.matmul(shared_layer5,shared_layer_weights6)+b6)


Y2_layer_1 = tf.nn.tanh(tf.matmul(shared_layer3,Y2_layer_weights1))
Y2_layer_2 = tf.nn.tanh(tf.matmul(Y2_layer_1,Y2_layer_weights2))
Y2_layer_3 = tf.nn.sigmoid(tf.matmul(Y2_layer_2,Y2_layer_weights3))



# Calculate Loss
cross_entropy1 = -tf.reduce_mean(tf.reduce_sum(Y1 * tf.log(Y1_layer) + (1 -Y1) * tf.log(1 - Y1_layer), axis=-1))
cross_entropy2 = -tf.reduce_mean(tf.reduce_sum(Y2 * tf.log(Y2_layer_3) + (1 - Y2) * tf.log(1 - Y2_layer_3), axis=-1))
mse1 = tf.reduce_mean(tf.square(Y1 - Y1_layer))
mse2 = tf.reduce_mean(tf.square(Y2 -Y2_layer_3))
# Y2_Loss = tf.nn.l2_loss(Y2-Y2_layer)
Joint_Loss = cross_entropy1 + cross_entropy2


# optimisers
Optimiser = tf.train.AdamOptimizer().minimize(Joint_Loss)
Y1_op = tf.train.AdamOptimizer().minimize(cross_entropy1)
Y2_op = tf.train.AdamOptimizer().minimize(cross_entropy2)






x_batch,y1_batch,y2_batch = get_batch_data(batch_size=batch_size,X=x_train,Y1=x_test,Y2=y_test)
# Calculation (Session) Code
# ==========================

# open the session

# 创建session
with tf.Session() as sess:
    # 变量初始化
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)
    sess.run(tf.initialize_all_variables())
    total_batch = int(df1.shape[0]/ batch_size)
    try:
        for epoch in range(epochs):
            avg_cost = 0
            for i in range(total_batch):
                x,y1,y2 = sess.run([x_batch,y1_batch,y2_batch])
                _, joint_Loss = sess.run([Optimiser, Joint_Loss],feed_dict={
                                             "X:0": x,
                                             "Y1:0": y1,
                                             "Y2:0": y2
                                         })
            avg_cost+=joint_Loss
            print("Epoch:",(epoch + 1),"cost = ","{:.3f}".format(avg_cost))
    except tf.errors.OutOfRangeError:
            print("done")
    finally:
        coord.request_stop()
    coord.join(threads)




# print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))



