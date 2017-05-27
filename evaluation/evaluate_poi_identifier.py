#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

sort_keys = '../tools/python2_lesson14_keys.pkl'

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)


clf = DecisionTreeClassifier()


features_train, features_test, labels_train, labels_test = model_selection.train_test_split(features,
                                                                                            labels,
                                                                                             test_size=0.3,
                                                                                             random_state=42)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test, labels_test)

print "#####    Old    #####"
print "accuracy score", accuracy_score(labels_test, pred)

print "old confusion", confusion_matrix(labels_test, pred)

print "precision score", precision_score(labels_test, pred)

print "recall score", recall_score(labels_test, pred)

pois_in_test = list(pred).count(1)

print "POIs in test", pois_in_test

new_pred = [0] * len(labels_test)

print "#####    Modified    #####"
print "modified accuracy score", accuracy_score(labels_test, new_pred)

print "new confusion", confusion_matrix(labels_test, new_pred)

print "precision score", precision_score(labels_test, new_pred)

print "recall score", recall_score(labels_test, new_pred)

print "######   Experimental    ####"
predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]

print "precision score", precision_score(true_labels, predictions)

print "recall score", recall_score(true_labels, predictions)





