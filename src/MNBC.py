# Multinomial Naive Bayes Classifier
# https://towardsdatascience.com/multinomial-naive-bayes-classifier-for-text-analysis-python-8dd6825ece67
# https://www.mygreatlearning.com/blog/multinomial-naive-bayes-explained/
# https://stackoverflow.com/questions/18702806/naive-bayes-classifier-explain-model-fitting-and-prediction-algorithms

# https://medium.com/@johnm.kovachi/implementing-a-multinomial-naive-bayes-classifier-from-scratch-with-python-e70de6a3b92e
# https://github.com/jmkovachi/sent-classifier




###### LIKELIHOOD

 #    |     "word1"    |     "word2"    | ... |     "word99"    | 
 # h0 |  P(word1 | h0) |  P(word2 | h0) | ... |  P(word99 | h0) |
 # h1 |  P(word1 | h1) |  P(word2 | h1) | ... |  P(word99 | h1) |

 # likelihood[0,0] = [class, word, probability]
 # e.g. likelihood[0,0] = ["yes", "apple", 0.45]
    
##########

from math import log10
smoothing = 0.1 

class NB_Classifier:

    def __init__(self):
        self.priors = { # Priors probabilities: ["yes": P(H1), "no" : P(H2)]
            "yes": 0, 
            "no": 0
        } 
        self.likelihood = list()

    def nb_fit(self, vocabulary):
        num_tweet_c = list() # number of instance that are part of that class 
        tot_num_tweets = vocabulary.length()  # total number of instance/tweets

        for h in self.priors:
            num_tweet_c = (vocabulary.freq(h))
            self.priors[h] = log10(num_tweet_c/tot_num_tweets)
            for f in vocabulary:
                self.likelihood.append([h, f, log10(f.freq()/num_tweet_c)]) # likelihood[0,0] = [class, word, probability]


    def nb(self, vocabulary):
        print("hello")