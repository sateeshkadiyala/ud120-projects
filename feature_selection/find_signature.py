#!/usr/bin/python

import pickle
import numpy
numpy.random.seed(42)
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn import tree



### The words (features) and authors (labels), already largely processed.
### These files should have been created from the previous (Lesson 10)
### mini-project.
words_file = "../text_learning/your_word_data.pkl"
authors_file = "../text_learning/your_email_authors.pkl"
word_data = pickle.load( open(words_file, "r"))
authors = pickle.load( open(authors_file, "r") )

from sklearn import model_selection
features_train, features_test, labels_train, labels_test = model_selection.train_test_split(word_data, authors, test_size=0.1, random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
features_train = vectorizer.fit_transform(features_train)
features_test  = vectorizer.transform(features_test).toarray()

features_train = features_train[:150].toarray()
labels_train   = labels_train[:150]


clf = tree.DecisionTreeClassifier()
clf.fit(features_train,  labels_train)
print len(features_train)

print "accuracy_score", accuracy_score(clf.predict(features_test), labels_test)

importances_list = list(clf.feature_importances_)

outliers = [outlier for outlier in importances_list if outlier > 0.2]

print "most important features/outliers", outliers

features = vectorizer.get_feature_names()

outlier_feature_names = [features[importances_list.index(outlier)] for outlier in outliers]

print "feature names", outlier_feature_names



