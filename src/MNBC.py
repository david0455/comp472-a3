

 ############### NOTE 1 #################################

 #  |     "word1"    |     "word2"    | ... |     "word99"    | 
 #  |  P(word1 | h0) |  P(word2 | h0) | ... |  P(word99 | h0) |

 # likelihood = {word, probability}
 # e.g. likelihood["apple"] <== {"apple": 0.45}
    
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
        self.priors = { # Priors probabilities: ["yes": P(H0), "no" : P(H1)]
            "yes": 0, 
            "no": 0
        } 
        self.score = { # Score of each class when predicting
            "yes": 0,
            "no": 0
        }
        self.final_result = []
        self.likelihood_h0 = {}  # Dictionary of conditional probabilities for h1; see Note 1
        self.likelihood_h1 = {}  # Dictionary of conditional probabilities for h0
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

        # take only the "Tweet" column and calculate the frequency of ALL words among all tweets
        all_tweets = train_set.iloc[:, 1]
        self.vocabulary = toDataFrame(feqTotalTweets(all_tweets)).to_numpy() # vocabulary <-- [word, frequency]

        for h in self.priors:
            num_tweet_c = (train_set['q1_label'] == h).sum()            # Number of tweets/instance in class c
            df_c = train_set[(train_set['q1_label'] == h)]              # All instances of class c
            num_word_c = toDataFrame(feqTotalTweets(df_c.iloc[:, 1]))   # Get all tweets/instances of class c
            tot_num_word_c = num_word_c[0].sum()                        # Total number of words in class c
            
            self.priors[h] = log10(num_tweet_c/tot_num_tweets)          # Calculate the prior P(H) 

            for f in self.vocabulary:
                word_f, _count = f

                # Check if class c has the word in vocabulary(BOW)
                temp_word = (num_word_c[num_word_c['index'] == word_f]) 
                if not temp_word.empty:
                    _word_c, count_c = temp_word.iloc[0]    # Get the frequency of the word in class c
                else:
                    count_c = 0
                
                # Calculate conditional probabilities/likelihood for each word in each class
                if h == "yes":
                    self.likelihood_h0[word_f] = log10((count_c + smoothing)/(tot_num_word_c + smoothing*self.vocabulary.size)) # likelihood h0 <-- {word: probability}
                elif h == "no":
                    self.likelihood_h1[word_f] = log10((count_c + smoothing)/(tot_num_word_c + smoothing*self.vocabulary.size)) # likelihood h1 <-- {word: probability}


    # TODO: Need to change method names for NB_BOW_FV
    def fit_FV(self, train_set):
         tot_num_tweets = train_set.shape[0] # total number of instance/tweets

        # take only the "Tweet" column and calculate the frequency of ALL words among all tweets
        all_tweets = train_set.iloc[:, 1]
        self.vocabulary = toDataFrame(feqTotalTweets(all_tweets)).to_numpy() # vocabulary <-- [word, frequency]

        for h in self.priors:
            num_tweet_c = (train_set['q1_label'] == h).sum()            # Number of tweets/instance in class c
            df_c = train_set[(train_set['q1_label'] == h)]              # All instances of class c
            num_word_c = toDataFrame(feqTotalTweets(df_c.iloc[:, 1]))   # Get all tweets/instances of class c
            tot_num_word_c = num_word_c[0].sum()                        # Total number of words in class c
            
            self.priors[h] = log10(num_tweet_c/tot_num_tweets)          # Calculate the prior P(H) 

            for f in self.vocabulary:
                word_f, _count = f

                # Check if class c has the word in vocabulary(BOW)
                temp_word = (num_word_c[num_word_c['index'] == word_f]) 
                if not temp_word.empty:
                    _word_c, count_c = temp_word.iloc[0]    # Get the frequency of the word in class c
                else:
                    count_c = 0
                
                # Calculate conditional probabilities/likelihood for each word in each class
                if h == "yes":
                    self.likelihood_h0[word_f] = log10((count_c + smoothing)/(tot_num_word_c + smoothing*self.vocabulary.size)) # likelihood h0 <-- {word: probability}
                elif h == "no":
                    self.likelihood_h1[word_f] = log10((count_c + smoothing)/(tot_num_word_c + smoothing*self.vocabulary.size)) # likelihood h1 <-- {word: probability}


    def predict(self, test_set):
        for f in test_set.to_numpy():                       # Numpy array of all instances in test_set
            _tweetID, tweet, _label = f                     # Each instance has columns: tweetID, tweet, label
            headers = toDataFrame(freq(tweet)).to_numpy()   # Get the list words for each individual tweets
            for h in self.priors:
                score = self.priors[h]         # Initialize score with prior of the class
                for word in headers: 
                    voc, count = word          # Get a word from the tweet and its frequency 
                    if voc in self.vocabulary: # If the word is in the list(BOW) of the vocabulary(from training_set), add to the score
                        if h == "yes":
                            score = score * (count*self.likelihood_h0[voc]) 
                        elif h == "no":
                            score = score * (count*self.likelihood_h1[voc])
                    else:
                        score += 0  # If word is not in vocabulary, ignore it
                self.score[h] = score                                       # When we looped through all the words in the tweet, store the score to its class 
            result_class = max(self.score.items(), key=op.itemgetter(1))    # Get the argmax between the score of each class
            self.final_result.append(result_class)      # Append final result in list

        # Put results in dataframe
        most_likely_class, final_score = zip(*self.final_result)
        test_set['predicted_class'] = most_likely_class
        test_set['score'] = np.array(final_score).round(2)
        # test_set['score'] = test_set['score'].round(decimals=2)

        self.getAccuracy(test_set)
        self.getPrecision(test_set)
        self.getRecall(test_set)
        self.getF1(test_set)

        print(test_set)

