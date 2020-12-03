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

# the tain_set should be formatted as follows:
# | tweetID | word1 | word2 | word3| ... | class | 
# | 0000001 |   2   |   3   |  4   | ... |  yes  |


import operator as op
from math import log10
smoothing = 0.1 

class NB_Classifier:

    def __init__(self):
        self.priors = { # Priors probabilities: ["yes": P(H1), "no" : P(H2)]
            "yes": 0, 
            "no": 0
        } 
        self.likelihood_h0 = {}  # Dictionary of conditional probabilities for h0; see Note 1
        self.likelihood_h1 = {}  # Dictionary of conditional probabilities for h1; see Note 1
        self.train_set # dataframe; see Note 2
        self.vocabulary = list()
        

    def nb_fit(self, train_set):
        self.train_set = train_set 
        num_tweet_c = list() # number of instance that are part of that class 
        tot_num_tweets = train_set.length()  # total number of instance/tweets

        # TODO: GET all column names (i.e. vocabulary) from dataframe train_set into array
        self.vocabulary = self.train_set.getColumnName()

        for h in self.priors:
            num_tweet_c = (train_set.freq(h))
            self.priors[h] = log10(num_tweet_c/tot_num_tweets)

            #TODO: add smooting
            for f in self.vocabulary:
                if h == "yes":
                    self.likelihood_h0[f] = log10(f.freq()/num_tweet_c) # likelihood[0,0] = [class, word, probability]
                elif h == "no":
                    self.likelihood_h1[f] = log10(f.freq()/num_tweet_c) # likelihood[0,0] = [class, word, probability]


    def nb(self, test_set):
        for h in self.priors:
            score = self.priors[h]
            for f in test_set:
                # TODO: Check if tweet has a word in Vocabulary list (training_set)
                if f.contains(self.vocabulary):
                    score[h] *= self.likelihood_h0[f]

        result_class = max(score.iteritems(), key=op.itemgetter(1))[0]
        print(result_class)

    #TODO: Precision
    def precision(self):
        print("hello")
    
    #TODO: Recall
    def recall(self):
        print("hello")
    
    #TODO: F1-measure
    def F1(self):
        print("hello")

