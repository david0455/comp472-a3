# Model 1: NB-BOW-OV
#  all words appearing in the training set are used as features

import pandas as pd
import numpy as np
from collections import Counter

def bow_ov():
    print("hello")

# Calculates the frequency of each word within a string
def freq(str):
    aList = str.split()
    counter = Counter(aList)
    return counter

# Calculates the frequency of each word within each tweet
def freqEachTweet(texts):
    templist = []
    for str in texts:
        print(type(str))
        templist.append(freq(str))
    return templist

# Calculates the frequency of each word in all tweets
def feqTotalTweets(texts):
    temp = ""
    for string in texts:
        temp += string + " "
    return freq(temp)

# Convert type Counter to type DataFrame
def toDataFrame(counter):
    df = pd.DataFrame.from_dict(counter, orient='index').reset_index()
    return df


training = pd.read_csv('./data/covid_training.tsv', sep='\t')
tweet_id = training.iloc[:, 0]
tweet_text = training.iloc[:, 1]
q1_label = training.iloc[:, 2]


test1 = training.iloc[0:3, 1]
total_words = feqTotalTweets(test1)
#print(total_words)
print()

test_row = training.iloc[0, 1]
#print(test_row)
print()

print(toDataFrame(freq(test_row)).T)
