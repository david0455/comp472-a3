# Model 1: NB-BOW-OV
#  all words appearing in the training set are used as features

import pandas as pd
import numpy as np
from collections import Counter


def bow_ov():
    print("hello")


# Calculates the frequency of each word within a string
def freq(str):
    aList = str.lower().split()
    counter = Counter(aList)
    return counter


# Calculates the frequency of each word within each tweet
def freqEachTweet(texts):
    templist = []
    for str in texts:
        templist.append(freq(str))
    return templist


# Calculates the frequency of each word in all tweets
def feqTotalTweets(texts):
    temp = ""
    for string in texts:
        temp += string.lower() + " "
    return freq(temp)


# Convert type Counter to type DataFrame
def toDataFrame(counter):
    df = pd.DataFrame.from_dict(counter, orient='index').reset_index()
    return df


# Get data from dataset
def getData(dataset, test):
    if test:
        file = pd.read_csv(dataset, sep='\t', header=None)
    else:
        file = pd.read_csv(dataset, sep='\t')
    return file.iloc[:, 0:3]


def filterVocab(counter):
    return Counter(elem for elem in counter.elements() if counter[elem] > 1)
