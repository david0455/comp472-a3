# Multinomial Naive Bayes Classifier
# https://towardsdatascience.com/multinomial-naive-bayes-classifier-for-text-analysis-python-8dd6825ece67
# https://www.mygreatlearning.com/blog/multinomial-naive-bayes-explained/
# https://stackoverflow.com/questions/18702806/naive-bayes-classifier-explain-model-fitting-and-prediction-algorithms

# https://medium.com/@johnm.kovachi/implementing-a-multinomial-naive-bayes-classifier-from-scratch-with-python-e70de6a3b92e
# https://github.com/jmkovachi/sent-classifier


# operator library
# https://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary


 ############### NOTE 1 #################################

 #  |     "word1"    |     "word2"    | ... |     "word99"    | 
 #  |  P(word1 | h0) |  P(word2 | h0) | ... |  P(word99 | h0) |

 # likelihood = {word: probability}
 # e.g. likelihood["apple"] <== ["apple", 0.45]
    
##########

############## NOTE 2 ###################################

# the train_set should be formatted as follows:
# | tweetID |  tweet  | class | 
# | 0000001 |  hello |  yes  |

import pandas as pd
import operator as op
import numpy as np
from math import log10
from NB_BOW_OV import toDataFrame, feqTotalTweets, freq

smoothing = 0.01 

class NB_Classifier:

    def __init__(self):
        self.priors = { # Priors probabilities: ["yes": P(H1), "no" : P(H2)]
            "yes": 0, 
            "no": 0
        } 
        self.score = {
            "yes": 0,
            "no": 0
        }
        self.final_result = []
        self.likelihood_h0 = {}  # Dictionary of conditional probabilities for h1; see Note 1
        self.likelihood_h1 = {}  # Dictionary 
        self.vocabulary = None
    

    #TODO: Accuracy
    def getAccuracy(self, test_set):
        print("hello")


    #TODO: Precision
    def getPrecision(self, test_set):
        print("hello")
    

    #TODO: Recall
    def getRecall(self, test_set):
        print("hello")
    

    #TODO: F1-measure
    def getF1(self, test_set):
        print("hello")


    def fit_OV(self, train_set):
        tot_num_tweets = train_set.shape[0] # total number of instance/tweets
        all_tweets = train_set.iloc[:, 1]
        self.vocabulary = toDataFrame(feqTotalTweets(all_tweets)).to_numpy()
        for h in self.priors:
            num_tweet_c = (train_set['q1_label'] == h).sum()
            df_c = train_set[(train_set['q1_label'] == h)] # all instances of class h
            num_word_c = toDataFrame(feqTotalTweets(df_c.iloc[:, 1]))
            tot_num_word_c = num_word_c[0].sum() # total number of words in class h
            self.priors[h] = log10(num_tweet_c/tot_num_tweets)

            for f in self.vocabulary:
                word_f, _count = f
                temp_word = (num_word_c[num_word_c['index'] == word_f])
                if not temp_word.empty:
                    _word_c, count_c = temp_word.iloc[0]
                else:
                    count_c = 0
                if h == "yes":
                    self.likelihood_h0[word_f] = log10((count_c + smoothing)/(tot_num_word_c + smoothing*self.vocabulary.size)) # likelihood[0,0] = [class, word, probability]
                elif h == "no":
                    self.likelihood_h1[word_f] = log10((count_c + smoothing)/(tot_num_word_c + smoothing*self.vocabulary.size)) # likelihood[0,0] = [class, word, probability]


    def fit_FV(self, train_set):
        tot_num_tweets = train_set.shape[0] # total number of instance/tweets
        all_tweets = train_set.iloc[:, 1]
        self.vocabulary = toDataFrame(feqTotalTweets(all_tweets)).to_numpy()
        for h in self.priors:
            num_tweet_c = (train_set['q1_label'] == h).sum()
            df_c = train_set[(train_set['q1_label'] == h)] # all instances of class h
            num_word_c = toDataFrame(feqTotalTweets(df_c.iloc[:, 1]))
            tot_num_word_c = num_word_c[0].sum() # total number of words in class h
            self.priors[h] = log10(num_tweet_c/tot_num_tweets)

            for f in self.vocabulary:
                word_f, _count = f
                temp_word = (num_word_c[num_word_c['index'] == word_f])
                if not temp_word.empty:
                    _word_c, count_c = temp_word.iloc[0]
                else:
                    count_c = 0
                if h == "yes":
                    self.likelihood_h0[word_f] = log10((count_c + smoothing)/(tot_num_word_c + smoothing*self.vocabulary.size)) # likelihood[0,0] = [class, word, probability]
                elif h == "no":
                    self.likelihood_h1[word_f] = log10((count_c + smoothing)/(tot_num_word_c + smoothing*self.vocabulary.size)) # likelihood[0,0] = [class, word, probability]


    def predict(self, test_set):
        for f in test_set.to_numpy(): # numpy array of all instances
            _tweetID, tweet, _label = f # each instance has columns: tweetID, tweet, label
            headers = toDataFrame(freq(tweet)).to_numpy() # get the vocabulary for each individual tweets
            for h in self.priors:
                score = self.priors[h]
                for word in headers: 
                    voc, count = word 
                    if voc in self.vocabulary: # if the voc is in the list of the training vocabulary
                        if h == "yes":
                            score = score * (count*self.likelihood_h0[voc])
                        elif h == "no":
                            score = score * (count*self.likelihood_h1[voc])
                    else:
                        score += 0 # if word is not in vocabulary, ignore it
                self.score[h] = score
            result_class = max(self.score.items(), key=op.itemgetter(1))
            self.final_result.append(result_class)

        most_likely_class, final_score = zip(*self.final_result)

        test_set['predicted_class'] = most_likely_class
        test_set['score'] = np.array(final_score).round(2)
        # test_set['score'] = test_set['score'].round(decimals=2)

        self.getAccuracy(test_set)
        self.getPrecision(test_set)
        self.getRecall(test_set)
        self.getF1(test_set)

        print(test_set)

