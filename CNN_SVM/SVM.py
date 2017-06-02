import os

import sklearn
from sklearn import cross_validation, grid_search
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.externals import joblib

def train_svm_classifer(features, labels, model_output_path):

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(features, labels, test_size=0.2)

    clf = svm.SVC(kernel='rbf', probability=True, C=10, gamma=0.001)

    clf.fit(X_train, y_train)

    y_predict=clf.predict(X_test)

    print("\nClassification report:")
    print(classification_report(y_test, y_predict))

    joblib.dump(clf, model_output_path)

