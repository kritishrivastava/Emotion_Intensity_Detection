import tensorflow as tf
import pandas as pd
from tensorflow.contrib.keras.python.keras.backend import epsilon
import numpy as np
import math


def tensor_test(test_feat, test_lab, model_path):
    length=len(test_lab)
    dim=test_feat.shape[1]
    
    x=tf.placeholder(tf.float64, [length, dim], "features")
    y_=tf.placeholder(tf.float64, [length], "labels")
    
    
    
    theta1=tf.get_variable("theta1", shape=[dim,300], dtype=tf.float64)
    theta2=tf.get_variable("theta2", shape=[300, 125], dtype=tf.float64)
    theta3=tf.get_variable("theta3", shape=[125, 50], dtype=tf.float64)
    theta4=tf.get_variable("theta4", shape=[50, 25], dtype=tf.float64)
    theta5=tf.get_variable("theta5", shape=[25,1], dtype=tf.float64)
    
    bias1=tf.get_variable("bias1", shape=[300], dtype=tf.float64)
    bias2=tf.get_variable("bias2", shape=[125], dtype=tf.float64)
    bias3=tf.get_variable("bias3", shape=[50], dtype=tf.float64)
    bias4=tf.get_variable("bias4", shape=[25], dtype=tf.float64)
    bias5=tf.get_variable("bias5", shape=[1], dtype=tf.float64)
    
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
    
    saver=tf.train.Saver({"theta1":theta1, "theta2": theta2, "theta3": theta3, "theta4": theta4, "theta5": theta5, "bias1": bias1, "bias2": bias2, "bias3": bias3, "bias4": bias4, "bias5": bias5})
    
    with tf.Session() as sess:
        saver.restore(sess, model_path)
        print("model restored")
        pearson=s22.eval(feed_dict={x: test_feat, y_: test_lab, keep_prob: 0.5})
        print(pearson)








        