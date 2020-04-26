# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 14:34:01 2020

@author: Admin
"""

# import the library pandas
import pandas as pd
import pickle
messages=pd.read_csv("SMSSpamCollection.txt", sep="\t", names=["Label", "Message"])
messages.head(2)


#Data preprocessing and cleaning
import re
import nltk # natural language toolkit
nltk.download('stopwords')


from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer # for stemming 
ps = PorterStemmer()
corpus = []
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['Message'][i])    # replacing with spaces
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
    
    
    
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=2500)
X = cv.fit_transform(corpus).toarray()
pickle.dump(cv, open("transform.pkl", "wb"))

y=pd.get_dummies(messages['Label'])
y=y.iloc[:,1].values


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Training model using Naive bayes classifier

from sklearn.naive_bayes import MultinomialNB
clf= MultinomialNB()
clf.fit(X_train, y_train)
clf.score(X_test, y_test)
filename="nlp_model.pkl"
pickle.dump(clf, open(filename, "wb"))











