import tensorflow as tf
import pandas as pd
from tensorflow.contrib.keras.python.keras.backend import epsilon
import numpy as np
import math
from tf_test import tensor_test
anger_path="anger.csv"
fear_path="fear.csv"
joy_path="joy.csv"
sadness_path="sadness.csv"

train_ratio=0.6

df=pd.read_csv(anger_path)
df2=df.set_index("id")

labels=df2.loc[:,"score"].values


features=df2.loc[:, "0":"399"].values

np.random.seed(0)
np.random.shuffle(labels)
np.random.seed(0)
np.random.shuffle(features)



example_count=features.shape[0]
dim=features.shape[1]

train_count=math.ceil(example_count*train_ratio)
train_feat=features[:train_count,:]
train_labels=labels[:train_count]

test_feat=features[train_count:,:]
test_labels=labels[train_count:]

length=train_count
x=tf.placeholder(tf.float64, [length, dim], "features")
y_=tf.placeholder(tf.float64, [length], "labels")

def weight(shape):
    a=tf.truncated_normal(shape, stddev=0.1, dtype=tf.float64)
    return tf.Variable(a)
def bias(shape):
    a=tf.constant(0.1, shape=shape, dtype=tf.float64)
    return tf.Variable(a)

theta1=weight([dim,300])
theta2=weight([300, 125])
theta3=weight([125, 50])
theta4=weight([50, 25])
theta5=weight([25,1])

bias1=bias([300])
bias2=bias([125])
bias3=bias([50])
bias4=bias([25])
bias5=bias([1])

l1=tf.nn.relu(tf.matmul(x,theta1)+bias1)

keep_prob=tf.placeholder(tf.float64)
l1_drop=tf.nn.dropout(l1,keep_prob)
l2=tf.nn.relu(tf.matmul(l1_drop,theta2)+bias2)
l3=tf.nn.relu(tf.matmul(l2,theta3)+bias3)
l4=tf.nn.relu(tf.matmul(l3,theta4)+bias4)
sig=tf.sigmoid(tf.matmul(l4,theta5)+bias5)



sig2=tf.reshape(sig, [-1])
y2=tf.reshape(y_,[-1])
o2=length*tf.reduce_sum(tf.multiply(sig2, y2))
s2=o2-(tf.reduce_sum(sig2) * tf.reduce_sum(y2))
s21=tf.sqrt(((length * tf.reduce_sum(tf.square(sig2))) - tf.square(tf.reduce_sum(sig2)))*((length * tf.reduce_sum(tf.square(y2))) - tf.square(tf.reduce_sum(y2))))
 ## Example code. This line not meant to work.
s22=-1*tf.divide(s2,s21)




# cost=-tf.contrib.metrics.streaming_pearson_correlation(sig, y_)
saver=tf.train.Saver()
train_step=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-9).minimize(s22)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print(example_count)
    for i in range(60):
        sess.run(train_step, feed_dict={x: train_feat, y_: train_labels, keep_prob: 0.5})
        print('Epoch', i)
        
#         print('sig', sess.run(sig2, feed_dict={x: features, y_: labels, keep_prob: 0.5}))
        print('cost', sess.run(s22, feed_dict={x: train_feat, y_: train_labels, keep_prob: 0.5}))
    saver.save(sess, "anger.ckpt")
        
tensor_test(test_feat,test_labels,"anger.ckpt")
      