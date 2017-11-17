from __future__ import division
import math
from collections import defaultdict
import pickle
import operator


total=defaultdict(float)

anger=defaultdict(float)
fear=defaultdict(float)
joy=defaultdict(float)
sadness=defaultdict(float)


anger_r=defaultdict(float)
fear_r=defaultdict(float)
joy_r=defaultdict(float)
sadness_r=defaultdict(float)

total_tweets=0
total_anger_tweets=0
total_fear_tweets=0
total_joy_tweets=0
total_sadness_tweets=0

PATH_TO_TRAIN_DATA="EI-reg-English-Train/EI-reg-train.txt"
PATH_TO_TEST_DATA="EI-reg-English-Train/EI-reg-test.txt"

def bow_creator(filename):
    global total_tweets
    for line in open(filename):
        total_tweets+=1
        id, tweet, label, score = line.split("\t")
        score=float(score)
        tokenizer(tweet, label, score)
    with open('NBModel.pkl', 'w') as f:
        pickle.dump([total_tweets, total_anger_tweets, total_fear_tweets, total_joy_tweets, total_sadness_tweets, total, anger, fear, joy, sadness], f)

def tokenizer(string, label, score):
    global total_anger_tweets, total_fear_tweets, total_joy_tweets, total_sadness_tweets
    tokens=string.split()
    tokens_lc=map(lambda x: x.lower(), tokens)
    if label=="anger":
        total_anger_tweets+=1
        for token in tokens_lc:
            total[token]+=1.0
            anger[token]+=1.0
            anger_r[token]+=score
            
    elif label=="fear":
        total_fear_tweets+=1
        for token in tokens_lc:
            total[token]+=1.0
            fear[token]+=1.0
            fear_r[token]+=score
    elif label=="joy":
        total_joy_tweets+=1
        for token in tokens_lc:
            total[token]+=1.0
            joy[token]+=1.0
            joy_r[token]+=score
    elif label=="sadness":
        total_sadness_tweets+=1
        for token in tokens_lc:
            total[token]+=1.0
            sadness[token]+=1.0
            sadness_r[token]+=score

def simple_tokenizer(string):
    bow=defaultdict(float)
    tokens=string.split()
    tokens_lc=map(lambda x: x.lower(), tokens)
    for token in tokens_lc:
        bow[token]+=1.0
    return bow

def predict(tweet, alpha):
    bow=simple_tokenizer(tweet)
    scores=defaultdict(float)
    
    likeli_1=0
    likeli_2=0
    likeli_3=0
    likeli_4=0
    
    for keys in bow.iterkeys():
        likeli_1+=math.log10((anger[keys]+alpha)/(sum(anger.values())+len(total.keys())*alpha))
        likeli_3+=math.log10((joy[keys]+alpha)/(sum(joy.values())+len(total.keys())*alpha))
        likeli_4+=math.log10((sadness[keys]+alpha)/(sum(sadness.values())+len(total.keys())*alpha))
        likeli_2+=math.log10((fear[keys]+alpha)/(sum(fear.values())+len(total.keys())*alpha))
        
    scores["anger"]=likeli_1+math.log10(total_anger_tweets/total_tweets)
    scores["fear"]=likeli_2+math.log10(total_fear_tweets/total_tweets)
    scores["joy"]=likeli_3+math.log10(total_joy_tweets/total_tweets)
    scores["sadness"]=likeli_4+math.log10(total_sadness_tweets/total_tweets)
    
    return max(scores.iteritems(), key=operator.itemgetter(1))[0]

def dict_map(d_r, d_e):
    for keys in d_r.iterkeys():
        d_r[keys]=d_r[keys]/d_e[keys]
        assert (d_r[keys]<=1), "not less than 1" 

def predict_test_ds(filename, alpha):
    counter=0
    accurate=0
    for line in open(filename):
        counter+=1
        id, tweet, label, score = line.split("\t")
        predicted=predict(tweet, alpha)
        if predicted==label:
            accurate+=1
        print tweet, "\t", predicted, "\t", label, "\n"
    score=accurate/counter
    print "\n", "accuracy score: ", score
        
        
if __name__=="__main__":
    print "training...\n"
    bow_creator(PATH_TO_TRAIN_DATA)
    dict_map(anger_r,anger)
    dict_map(fear_r,fear)
    dict_map(joy_r,joy)
    dict_map(sadness_r, sadness)
    with open("reg_vals.pkl", 'w') as f:
        pickle.dump([anger_r, fear_r, joy_r, sadness_r], f)
    print "training done\n", "now testing"
#     predict_test_ds(PATH_TO_TEST_DATA, 0.05)
    
    
    
    