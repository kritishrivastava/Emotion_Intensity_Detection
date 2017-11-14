import os
from sklearn.model_selection import train_test_split
import pandas
import preprocessor as p
import string
from nltk import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem.snowball import SnowballStemmer

# Setting parameters for preprocessing tweet text
p.set_options(p.OPT.MENTION,p.OPT.RESERVED)
stemmer = SnowballStemmer("english")
table = str.maketrans({key: None for key in string.punctuation})

def process_tweet(tweet):
    '''
    :param tweet: tweet text
    :return: processed tweet in the form of a string containing comma separated words
    '''
    tweet = str(tweet)
    tweet = p.clean(tweet)
    stop = stopwords.words('english') + list(string.punctuation)
    tweet = tweet.translate(table)
    words = [i for i in word_tokenize(tweet.lower()) if i not in stop]
    words = [item for item in words if not item.isdigit()]
    # words = [stemmer.stem(word) for word in words]
    words = ",".join(words)
    return words

def read_data(path):
    '''
    :param path: file path for the file to be read
    :return: dataframe of processed data
    '''
    data = pandas.read_table(path, sep="\t")
    data.columns = ["ID", "Tweet", "Emotion", "Intensity"]
    data['Words'] = data.apply(lambda row: process_tweet(row["Tweet"]), axis=1)
    return data

path = os.getcwd()
path = path + "\data\EI-reg-English-Dev\\2018-EI-reg-En-anger-dev.txt"
data = read_data(path)
tweet_train, tweet_test, intensity_train, intensity_test = train_test_split(data["Words"], data["Intensity"], test_size=0.33)
print(tweet_test)