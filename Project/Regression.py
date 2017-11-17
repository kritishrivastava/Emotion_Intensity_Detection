from __future__ import division
import math
import operator
import pickle



PATH_TO_DATA1="EI-reg-English-Dev/2018-EI-reg-En-anger-dev.txt"
PATH_TO_DATA2="EI-reg-English-Dev/2018-EI-reg-En-fear-dev.txt"
PATH_TO_DATA3="EI-reg-English-Dev/2018-EI-reg-En-joy-dev.txt"
PATH_TO_DATA4="EI-reg-English-Dev/2018-EI-reg-En-sadness-dev.txt"

PATH_TO_TRAIND_DATA="EI-reg-English-Dev/2018-EI-reg-En-train-dev.txt"
PATH_TO_TESTD_DATA="EI-reg-English-Dev/2018-EI-reg-En-test-dev.txt"

PATH_TO_TRAIN1="EI-reg-English-Train/EI-reg-en_anger_train.txt"
PATH_TO_TRAIN2="EI-reg-English-Train/EI-reg-en_fear_train.txt"
PATH_TO_TRAIN3="EI-reg-English-Train/EI-reg-en_joy_train.txt"
PATH_TO_TRAIN4="EI-reg-English-Train/EI-reg-en_sadness_train.txt"

PATH_TO_TRAIN_DATA="EI-reg-English-Train/EI-reg-train.txt"
PATH_TO_TEST_DATA="EI-reg-English-Train/EI-reg-test.txt"
from collections import defaultdict

with open('NBModel.pkl') as f:
    total_tweets, total_anger_tweets, total_fear_tweets, total_joy_tweets, total_sadness_tweets, total, anger, fear, joy, sadness=pickle.load(f)


with open('reg_vals.pkl') as f2:
    anger_r, fear_r, joy_r, sadness_r=pickle.load(f2)
    


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

def predict_test_ds(filename, alpha):
    counter=0
    accurate=0
    sq_err=0
    for line in open(filename):
        counter+=1
        id, tweet, label, score = line.split("\t")
        predicted=predict(tweet, alpha)
        predscore=predict_reg_val(tweet, predicted)
        if predicted==label:
            accurate+=1
            sq_err+=(float(score)-predscore)**2
            print label, "\t", predscore, "\t", float(score)
            
            
#         print tweet, "\t", predicted, "\t", label, "\n"
    score=accurate/counter
    print "\n", "accuracy score: ", score, "\t", "MSE", "\t", sq_err/accurate

def predict_reg_val(tweet, predlabel):
    tokens=tweet.split();
    tokens=map(lambda y: y.lower(), tokens)
    if predlabel=="anger":
        sumval=0
        for token in tokens:
            sumval+=anger_r[token]
        predscore=sumval/len(tokens)
    elif predlabel=="fear":
        sumval=0
        for token in tokens:
            sumval+=fear_r[token]
        predscore=sumval/len(tokens)
    elif predlabel=="joy":
        sumval=0
        for token in tokens:
            sumval+=joy_r[token]
        predscore=sumval/len(tokens)
    elif predlabel=="sadness":
        sumval=0
        for token in tokens:
            sumval+=sadness_r[token]
        predscore=sumval/len(tokens)
    return predscore
if __name__=="__main__":
    predict_test_ds(PATH_TO_TEST_DATA, 0.05)
    