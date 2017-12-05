from __future__ import division
import math
import operator
import pickle
from numpy import corrcoef
import numpy as np
import csv
import emoji
from nltk.tokenize import RegexpTokenizer
import preprocessor as p
import pandas as pd




PATH_TO_DATA1="EI-reg-English-Dev/2018-EI-reg-En-anger-dev.txt"
PATH_TO_DATA2="EI-reg-English-Dev/2018-EI-reg-En-fear-dev.txt"
PATH_TO_DATA3="EI-reg-English-Dev/2018-EI-reg-En-joy-dev.txt"
PATH_TO_DATA4="EI-reg-English-Dev/2018-EI-reg-En-sadness-dev.txt"

PATH_TO_TRAIN1="EI-reg-English-Train/EI-reg-en_anger_train.txt"
PATH_TO_TRAIN2="EI-reg-English-Train/EI-reg-en_fear_train.txt"
PATH_TO_TRAIN3="EI-reg-English-Train/EI-reg-en_joy_train.txt"
PATH_TO_TRAIN4="EI-reg-English-Train/EI-reg-en_sadness_train.txt"

from word2vec import word2vecReaderUtils, word2vecReader
punctuation=['!',',', '.', ';', ':', '/', '-', '?', '\\', '\'', '\"']


model_path = "word2vec/word2vec_twitter_model.bin"
print("Loading the model, this can take some time...")
model = word2vecReader.Word2Vec.load_word2vec_format(model_path, binary=True)
def complex_tok(string):
    p.set_options(p.OPT.URL, p.OPT.EMOJI, p.OPT.RESERVED, p.OPT.MENTION)
    string=p.clean(string)
    tokenizer=RegexpTokenizer('\w+')
    raw=tokenizer.tokenize(string)
    tokens_lc=map(lambda x: x.lower(), raw)
    return tokens_lc
    
def simple_tokenizer(string):
    tokens=string.split()
    tokens_lc=map(lambda x: x.lower(), tokens)
    remove=[]
    for i in range(len(tokens_lc)):
        if((len(tokens_lc[i])>=2 and tokens_lc[i]=="rt") or (len(tokens_lc[i])>=3 and (tokens_lc[i][:4]=="http" or tokens_lc[i][:3]=="www"))):
            remove.append(tokens_lc[i])
        if(tokens_lc[i][0]=="#"):
            tokens_lc[i]=token[1:]
        if (tokens_lc[i][0]=='@' or (len(tokens_lc[i])>=3 and tokens_lc[i][-4:].decode('utf-8') in emoji.UNICODE_EMOJI)):
            remove.append(tokens_lc[i])
        if(tokens_lc[i][-1] in punctuation):
            tokens_lc[i]=tokens_lc[i][:-1]
        if(tokens_lc[i][0] in punctuation):
            tokens_lc[i]=tokens_lc[i][1:]
    for i in remove:
        tokens_lc.remove(i)
            
    return tokens_lc


def get_w2vec(data_path1, data_path2, filename):
    c1_list=['id', 'tweet', 'label', 'score']
    c2_list=[str(i) for i in range(400)]
    c1_list.extend(c2_list)
    df=pd.DataFrame(columns=c1_list)
    index=0
    for line in open(data_path1):
        id, tweet, label, score = line.split("\t")
        print tweet
        score=float(score)
        tokens_lc=complex_tok(tweet)
        count=0
        for t in tokens_lc:
            if (model.__contains__(t)):
                count+=1
        print tokens_lc
        data_arr=np.zeros((count, 400))
        for i in range(len(tokens_lc)):
            if(model.__contains__(tokens_lc[i])):
                data_arr[i:]=model.__getitem__(tokens_lc[i])
            
        mean_arr=np.mean(data_arr, axis=0)
        arr1=[id, tweet, label, score]
        arr1.extend(mean_arr)
        df.loc[index]=arr1
        index+=1
    for line in open(data_path2):
        id, tweet, label, score = line.split("\t")
        print tweet
        score=float(score)
        tokens_lc=complex_tok(tweet)
        count=0
        for t in tokens_lc:
            if (model.__contains__(t)):
                count+=1
        print tokens_lc
        data_arr=np.zeros((count, 400))
        for i in range(len(tokens_lc)):
            if(model.__contains__(tokens_lc[i])):
                data_arr[i:]=model.__getitem__(tokens_lc[i])
              
        mean_arr=np.mean(data_arr, axis=0)
        arr1=[id, tweet, label, score]
        arr1.extend(mean_arr)
        df.loc[index]=arr1
        index+=1
    
    df.to_csv(filename)

get_w2vec(PATH_TO_DATA1, PATH_TO_TRAIN1, "anger.csv")
get_w2vec(PATH_TO_DATA2, PATH_TO_TRAIN2, "fear.csv")
get_w2vec(PATH_TO_DATA3, PATH_TO_TRAIN3, "joy.csv")
get_w2vec(PATH_TO_DATA4, PATH_TO_TRAIN4, "sadness.csv")