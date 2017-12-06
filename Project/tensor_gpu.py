import tensorflow as tf
import pandas as pd
from tensorflow.contrib.keras.python.keras.backend import epsilon
import numpy as np
import math

import pickle as p

anger_path="anger.csv"
fear_path="fear.csv"
joy_path="joy.csv"
sadness_path="sadness.csv"

def neg_pearson(predicted_tensor, actual_tensor, length):
    predicted_tensor=tf.reshape(predicted_tensor, [-1])
    actual_tensor=tf.reshape(actual_tensor, [-1])
    num=(length*tf.reduce_sum(tf.multiply(predicted_tensor, actual_tensor)))-(tf.reduce_sum(predicted_tensor) * tf.reduce_sum(actual_tensor))
    denom=s21=tf.sqrt(((length * tf.reduce_sum(tf.square(predicted_tensor))) - tf.square(tf.reduce_sum(predicted_tensor)))*((length * tf.reduce_sum(tf.square(actual_tensor))) - tf.square(tf.reduce_sum(actual_tensor))))
    pearson=-1*tf.divide(num,denom)
    return pearson
    
    
def mse(predicted_tensor, actual_tensor, length):
    predicted_tensor=tf.reshape(predicted_tensor, [-1])
    actual_tensor=tf.reshape(actual_tensor, [-1])
    mse=tf.reduce_sum(tf.square(tf.subtract(predicted_tensor,actual_tensor)))/length
    return mse
    
def rmse(predicted_tensor, actual_tensor, length):
    predicted_tensor=tf.reshape(predicted_tensor, [-1])
    actual_tensor=tf.reshape(actual_tensor, [-1])
    rmse=tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(predicted_tensor,actual_tensor)))/length)
    return rmse       
    
def weight(shape):
    a=tf.truncated_normal(shape, stddev=0.1, dtype=tf.float64)
    return tf.Variable(a)
def bias(shape):
    a=tf.constant(0.1, shape=shape, dtype=tf.float64)
    return tf.Variable(a)





def train_validate_test(train_feat, train_labels, test_feat, test_labels, l1_neurons=300, l2_neurons=125, l3_neurons=50, l4_neurons=25, drop_prob=0.5, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-9, analytics=False, filename=""):
        
    train_cases=train_feat.shape[0]
    test_cases=test_feat.shape[0]
    dim=train_feat.shape[1]
    with tf.device('/device:GPU:0'):
        x=tf.placeholder(tf.float64, [None, dim], "features")
        y_=tf.placeholder(tf.float64, [None], "labels")
        
        
        theta1=weight([dim,l1_neurons])
        theta2=weight([l1_neurons, l2_neurons])
        theta3=weight([l2_neurons, l3_neurons])
        theta4=weight([l3_neurons, l4_neurons])
        theta5=weight([l4_neurons,1])
        
        bias1=bias([l1_neurons])
        bias2=bias([l2_neurons])
        bias3=bias([l3_neurons])
        bias4=bias([l4_neurons])
        bias5=bias([1])
        
        l1=tf.nn.relu(tf.matmul(x,theta1)+bias1)
        
        keep_prob=tf.placeholder(tf.float64)
        l1_drop=tf.nn.dropout(l1,keep_prob)
        l2=tf.nn.relu(tf.matmul(l1_drop,theta2)+bias2)
        l3=tf.nn.relu(tf.matmul(l2,theta3)+bias3)
        l4=tf.nn.relu(tf.matmul(l3,theta4)+bias4)
        sig=tf.sigmoid(tf.matmul(l4,theta5)+bias5)
        
        
        
        pearson_tr=neg_pearson(sig, y_, train_cases)
        
        pearson_te=neg_pearson(sig, y_, test_cases)
        
        
        mse_tr=mse(sig, y_, train_cases)
        mse_te=mse(sig,y_,test_cases)
    
        rmse_tr=rmse(sig, y_, train_cases)
        rmse_te=rmse(sig, y_, test_cases)
    
    
        train_step=tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, beta2=beta2, epsilon=epsilon).minimize(mse_tr)
    
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        sess.run(tf.initialize_all_variables())
        pearson_arr=[]
        for i in range(60):
            sess.run(train_step, feed_dict={x: train_feat, y_: train_labels, keep_prob: drop_prob})
#             print('Epoch', i)
            pearson_arr.append(-sess.run(pearson_tr, feed_dict={x: train_feat, y_: train_labels, keep_prob: drop_prob}))
    #         print('sig', sess.run(sig2, feed_dict={x: features, y_: labels, keep_prob: 0.5}))
#             print('mse', sess.run(mse_tr, feed_dict={x: train_feat, y_: train_labels, keep_prob: drop_prob}))
        test_mse=sess.run(mse_te, feed_dict={x: test_feat, y_: test_labels, keep_prob: drop_prob})
        test_pearson=-sess.run(pearson_te, feed_dict={x: test_feat, y_: test_labels, keep_prob: drop_prob})
        test_rmse=sess.run(rmse_te, feed_dict={x: test_feat, y_: test_labels, keep_prob: drop_prob})
        map={"pearson_tr_arr": pearson_arr, "test_mse": test_mse, "test_pearson": test_pearson, "test_rmse": test_rmse, "l1_neurons": l1_neurons, "l2_neurons": l2_neurons, "l3_neurons": l3_neurons, "l4_neurons": l4_neurons, "drop_prob": drop_prob, "learning_rate": learning_rate, "beta1": beta1, "beta2": beta2, "epsilon": epsilon}
        if(analytics):
            assert(filename!=""), "filename not available"
            with open(filename, 'wb') as handle:
                p.dump(map, handle, protocol=p.HIGHEST_PROTOCOL)
                print(filename, " saved")
        
            
#         print('test_mse', test_mse)
#         print('test_pearson', test_pearson)
#         print('test_rmse', test_rmse)
        
        
        return map
def k_fold_cv(dataset_path, filename, train_valid_split=0.8, k=10):
    
    assert (train_valid_split<1), "sum of train & validation split should be less than 1"
    df=pd.read_csv(dataset_path)
    df2=df.set_index("id")
    
    labels=df2.loc[:,"score"].values
    
    
    features=df2.loc[:, "0":"399"].values
    
    np.random.seed(0)
    np.random.shuffle(labels)
    np.random.seed(0)
    np.random.shuffle(features)
    
    
    
    example_count=features.shape[0]
    dim=features.shape[1]
    
    
    train_valid_count=math.ceil(example_count*train_valid_split)
    single_f_len=math.ceil(train_valid_count/k)
    
    train_v_feat=features[:train_valid_count,:]
    train_v_lab=labels[:train_valid_count]
    
    test_feat=features[train_valid_count:,]
    test_lab=labels[train_valid_count:]
    
     
    fold_length=[]
    for i in range(k):
        if(i==k-1):
            fold_length.append(train_valid_count-sum(fold_length))
        else:
            fold_length.append(single_f_len)
    
    
    l1_mse=[]        
    l1_arr=[(300-10*i) for i in range(10)]
    for l1 in l1_arr:
        print('l1', l1)
        mse_arr=[]
        for i in range(k):
            
            if(i==0):
                train_feat=train_v_feat[fold_length[0]:,:]
                train_labels=train_v_lab[fold_length[0]:]
                 
                valid_feat=train_v_feat[:fold_length[0],:]
                valid_labels=train_v_lab[:fold_length[0]]
                
                
                map=train_validate_test(train_feat, train_labels, valid_feat, valid_labels, l1_neurons=l1)
                mse_arr.append(map['test_mse'])
    
            elif(i==k-1):
                train_feat=train_v_feat[:sum(fold_length[:-1]),:]
                train_labels=train_v_lab[:sum(fold_length[:-1])]
                 
                valid_feat=train_v_feat[sum(fold_length[:-1]):,:]
                valid_labels=train_v_lab[sum(fold_length[:-1]):]
                
                map=train_validate_test(train_feat, train_labels, valid_feat, valid_labels, l1_neurons=l1)
                mse_arr.append(map['test_mse'])
                
            else:
                feat1=train_v_feat[:sum(fold_length[:i]),:]
                feat2=train_v_feat[sum(fold_length[:i+1]):,:]
                train_feat=np.concatenate((feat1,feat2),axis=0)
                
                lab1=train_v_lab[:sum(fold_length[:i])]
                lab2=train_v_lab[sum(fold_length[:i+1]):]
                train_labels=np.concatenate((lab1, lab2))
                
                valid_feat=features[sum(fold_length[:i]):sum(fold_length[:i+1]),:]
                valid_labels=train_v_lab[sum(fold_length[:i]):sum(fold_length[:i+1])]
                
                map=train_validate_test(train_feat, train_labels, valid_feat, valid_labels, l1_neurons=l1)
                mse_arr.append(map['test_mse'])
    
        
        l1_mse.append(np.mean(mse_arr))
    best_l1=l1_arr[np.argmin(l1_mse)]
    print('best_l1', best_l1)
    
    
    l2_mse=[]        
    l2_arr=[(best_l1-10*i) for i in range(10)]
    for l2 in l2_arr:
        print('l2', l2)
        mse_arr=[]
        if(l2<2):
            break
        for i in range(k):
            
            if(i==0):
                train_feat=train_v_feat[fold_length[0]:,:]
                train_labels=train_v_lab[fold_length[0]:]
                 
                valid_feat=train_v_feat[:fold_length[0],:]
                valid_labels=train_v_lab[:fold_length[0]]
                
                
                map=train_validate_test(train_feat, train_labels, valid_feat, valid_labels, l1_neurons=best_l1, l2_neurons=l2)
                mse_arr.append(map['test_mse'])
    
            elif(i==k-1):
                train_feat=train_v_feat[:sum(fold_length[:-1]),:]
                train_labels=train_v_lab[:sum(fold_length[:-1])]
                 
                valid_feat=train_v_feat[sum(fold_length[:-1]):,:]
                valid_labels=train_v_lab[sum(fold_length[:-1]):]
                
                map=train_validate_test(train_feat, train_labels, valid_feat, valid_labels, l1_neurons=best_l1, l2_neurons=l2)
                mse_arr.append(map['test_mse'])
                
            else:
                feat1=train_v_feat[:sum(fold_length[:i]),:]
                feat2=train_v_feat[sum(fold_length[:i+1]):,:]
                train_feat=np.concatenate((feat1,feat2),axis=0)
                
                lab1=train_v_lab[:sum(fold_length[:i])]
                lab2=train_v_lab[sum(fold_length[:i+1]):]
                train_labels=np.concatenate((lab1, lab2))
                
                valid_feat=features[sum(fold_length[:i]):sum(fold_length[:i+1]),:]
                valid_labels=train_v_lab[sum(fold_length[:i]):sum(fold_length[:i+1])]
                
                map=train_validate_test(train_feat, train_labels, valid_feat, valid_labels, l1_neurons=best_l1, l2_neurons=l2)
                mse_arr.append(map['test_mse'])
    
        
        l2_mse.append(np.mean(mse_arr))
    best_l2=l2_arr[np.argmin(l2_mse)]
    print('best_l2', best_l2)
    
    l3_mse=[]        
    l3_arr=[(best_l2-10*i) for i in range(10)]
    for l3 in l3_arr:
        print('l3', l3)
        mse_arr=[]
        if(l3<2):
            break
        for i in range(k):
            
            if(i==0):
                train_feat=train_v_feat[fold_length[0]:,:]
                train_labels=train_v_lab[fold_length[0]:]
                 
                valid_feat=train_v_feat[:fold_length[0],:]
                valid_labels=train_v_lab[:fold_length[0]]
                
                
                map=train_validate_test(train_feat, train_labels, valid_feat, valid_labels, l1_neurons=best_l1, l2_neurons=best_l2, l3_neurons=l3)
                mse_arr.append(map['test_mse'])
    
            elif(i==k-1):
                train_feat=train_v_feat[:sum(fold_length[:-1]),:]
                train_labels=train_v_lab[:sum(fold_length[:-1])]
                 
                valid_feat=train_v_feat[sum(fold_length[:-1]):,:]
                valid_labels=train_v_lab[sum(fold_length[:-1]):]
                
                map=train_validate_test(train_feat, train_labels, valid_feat, valid_labels, l1_neurons=best_l1, l2_neurons=best_l2, l3_neurons=l3)
                mse_arr.append(map['test_mse'])
                
            else:
                feat1=train_v_feat[:sum(fold_length[:i]),:]
                feat2=train_v_feat[sum(fold_length[:i+1]):,:]
                train_feat=np.concatenate((feat1,feat2),axis=0)
                
                lab1=train_v_lab[:sum(fold_length[:i])]
                lab2=train_v_lab[sum(fold_length[:i+1]):]
                train_labels=np.concatenate((lab1, lab2))
                
                valid_feat=features[sum(fold_length[:i]):sum(fold_length[:i+1]),:]
                valid_labels=train_v_lab[sum(fold_length[:i]):sum(fold_length[:i+1])]
                
                map=train_validate_test(train_feat, train_labels, valid_feat, valid_labels, l1_neurons=best_l1, l2_neurons=best_l2, l3_neurons=l3)
                mse_arr.append(map['test_mse'])
    
        
        l3_mse.append(np.mean(mse_arr))
    best_l3=l3_arr[np.argmin(l3_mse)]
    print('best_l3', best_l3)
    
    
    l4_mse=[]        
    l4_arr=[(best_l3-10*i) for i in range(10)]
    for l4 in l4_arr:
        print('l4', l4)
        mse_arr=[]
        if(l4<2):
            break
        for i in range(k):
            
            if(i==0):
                train_feat=train_v_feat[fold_length[0]:,:]
                train_labels=train_v_lab[fold_length[0]:]
                 
                valid_feat=train_v_feat[:fold_length[0],:]
                valid_labels=train_v_lab[:fold_length[0]]
                
                
                map=train_validate_test(train_feat, train_labels, valid_feat, valid_labels,l1_neurons=best_l1, l2_neurons=best_l2, l3_neurons=best_l3, l4_neurons=l4)
                mse_arr.append(map['test_mse'])
    
            elif(i==k-1):
                train_feat=train_v_feat[:sum(fold_length[:-1]),:]
                train_labels=train_v_lab[:sum(fold_length[:-1])]
                 
                valid_feat=train_v_feat[sum(fold_length[:-1]):,:]
                valid_labels=train_v_lab[sum(fold_length[:-1]):]
                
                map=train_validate_test(train_feat, train_labels, valid_feat, valid_labels, l1_neurons=best_l1, l2_neurons=best_l2, l3_neurons=best_l3, l4_neurons=l4)
                mse_arr.append(map['test_mse'])
                
            else:
                feat1=train_v_feat[:sum(fold_length[:i]),:]
                feat2=train_v_feat[sum(fold_length[:i+1]):,:]
                train_feat=np.concatenate((feat1,feat2),axis=0)
                
                lab1=train_v_lab[:sum(fold_length[:i])]
                lab2=train_v_lab[sum(fold_length[:i+1]):]
                train_labels=np.concatenate((lab1, lab2))
                
                valid_feat=features[sum(fold_length[:i]):sum(fold_length[:i+1]),:]
                valid_labels=train_v_lab[sum(fold_length[:i]):sum(fold_length[:i+1])]
                
                map=train_validate_test(train_feat, train_labels, valid_feat, valid_labels, l1_neurons=best_l1, l2_neurons=best_l2, l3_neurons=best_l3, l4_neurons=l4)
                mse_arr.append(map['test_mse'])
    
        
        l4_mse.append(np.mean(mse_arr))
    best_l4=l4_arr[np.argmin(l4_mse)]
    print('best_l4', best_l4)
    
    train_validate_test(train_v_feat, train_v_lab, test_feat, test_lab, l1_neurons=best_l1, l2_neurons=best_l2, l3_neurons=best_l3, l4_neurons=best_l4, analytics=True, filename=filename)

# k_fold_cv(fear_path, "fear.pkl", k=3)
# k_fold_cv(joy_path, "joy.pkl", k=3)
k_fold_cv(sadness_path, "sadness.pkl", k=3)


     