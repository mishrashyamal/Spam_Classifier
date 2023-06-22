from collections import Counter, defaultdict
import math,re
import pandas as pd
import numpy as np
spam = pd.read_csv("spam.csv", encoding = 'ISO-8859-1')

def tokenize(message):
    
    message = message.lower()                       # convert to lowercase
    all_words = re.findall("[a-z0-9']+", message)   # extract the words
    return set(all_words)                           # remove duplicates

def count_words(training_set):
   
    counts = defaultdict(lambda: [0, 0])
    for message, is_spam in training_set:
        for word in tokenize(message):
            counts[word][0 if is_spam else 1] += 1
    return counts

def word_probabilities(counts, total_spams, total_non_spams, k=0.5):
    
    return [(w,
             (spam + k) / (total_spams + 2 * k),
             (non_spam + k) / (total_non_spams + 2 * k))
             for w, (spam, non_spam) in counts.items()]

def spam_probability(word_probs, message, total_spams, total_non_spams, k = 0.5):

    
    message_words = tokenize(message)
    log_prob_if_spam = log_prob_if_not_spam = 0.0

    
    for word, total_spams, total_non_spams in word_probs:

       
        if word in message_words:
            log_prob_if_spam += math.log(total_spams)
            log_prob_if_not_spam += math.log(total_non_spams)

       
        else:
            log_prob_if_spam += math.log(1.0 - total_spams)
            log_prob_if_not_spam += math.log(1.0 - total_non_spams)

    prob_spam = math.exp(log_prob_if_spam)
    prob_ham = math.exp(log_prob_if_not_spam)
   
    
    return prob_spam / (prob_spam + prob_ham)

  def naiveBayes_classify(word_probs, message, total_spams, total_non_spams, k):
    
    spam_prob = spam_probability(word_probs, message, total_spams, total_non_spams, k)
    return "spam" if spam_prob > 0.5 else "ham"

spam.head()
spam.shape
spam['is_spam'] = spam['label'].map({'spam':1, 'ham':0})
spam.head()


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(spam['text'], spam['is_spam'], test_size = 0.2, random_state = 0)
y_test = list(y_test.map({0:'ham',1:'spam'}))
training_set = zip(X_train,y_train)
counts = count_words(training_set)
counts
total_spams = y_train.sum()
total_spams
total_non_spams = y_train.shape[0] - total_spams
total_non_spams
word_probs = word_probabilities(counts, total_spams, total_non_spams, k=0.5)
#just check if this works for any given text in the dataset.
naiveBayes_classify(word_probs, spam['text'][2], total_spams, total_non_spams, 0.5)
X_train.iloc[0]
X_test.iloc[0]
y_pred = []

for i in range(X_test.shape[0]):
    y_pred.append(naiveBayes_classify(word_probs, X_test.iloc[i], total_spams, total_non_spams, 0.5))
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

print("Accuracy score: ", accuracy_score(y_test, y_pred))
print("Recall score: ", recall_score(y_test, y_pred, average = 'weighted'))
print("Precision score: ", precision_score(y_test, y_pred, average = 'weighted'))
print("F1 score: ", f1_score(y_test, y_pred, average = 'weighted'))
