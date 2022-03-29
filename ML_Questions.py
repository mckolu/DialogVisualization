import QuestionExtraction
import re
import numpy as np
import pandas as pd
import time
import nltk
from collections import Counter
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.stem import PorterStemmer
porter_stemmer = PorterStemmer()
pd.set_option("display.max_columns", 20)
start = time.time()
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))
new_words = ["\"", ".\"", "\"?", ".'", "...", "okay","randomly","yeah", "like", "large", "also", "iv", "one", "two", "new", "previously", "okay"]
stop_words = stop_words.union(new_words)


def my_cool_preprocessor1(text):
    tokens = nltk.word_tokenize(text)
    stems = []
    for item in tokens:
        if item not in stop_words:
            stems.append(PorterStemmer().stem(item))
    return stems


def my_cool_preprocessor(text):
    text = text.lower()
    # text = re.sub("\\W", " ", text)  # remove special chars
    # text = re.sub("\\s+(in|the|all|for|and|on)\\s+", " _connector_ ", text)  # normalize certain words

    # stem words
    words = re.split("\\s+", text)
    stemmed_words = [porter_stemmer.stem(word=word) for word in words]
    # stemmed_words = [word for word in words if word not in stop_words]
    # stemmed_words = [porter_stemmer.stem(word=word) for word in words if word not in stop_words]
    return ' '.join(stemmed_words)


df_new1 = QuestionExtraction.ExtractData()
df_new = df_new1[['File', 'ExtractedQuestions', 'QuestionCode']]



X = np.asarray(df_new['ExtractedQuestions'].tolist())
y = np.asarray(df_new['QuestionCode'].tolist())
y = np.asarray([1 if i=="C" else 0 for i in y])


# tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=2000, preprocessor=my_cool_preprocessor, lowercase=True)
tfidf = CountVectorizer(ngram_range=(1,2), max_features=2000, preprocessor=my_cool_preprocessor, lowercase=True) # , preprocessor=my_cool_preprocessor ... can be added min_df=2. Also, preprocessor=my_cool_preprocessor def my_cool_preprocessor(text):

# # from sklearn.model_selection import train_test_split
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# #
# # X_train = tfidf.fit_transform(X_train)
# # X_test = tfidf.transform(X_test)
#
#
# ## KNN Model
# from sklearn.neighbors import KNeighborsClassifier
# neigh = KNeighborsClassifier(n_neighbors=3)
# # model = neigh.fit(X_train_tf, y_train)
# # y_pred = neigh.predict(X_test_tf)
#
# # # SCM Model
# from sklearn import svm
# clf_svm = svm.SVC()
# # model = clf.fit(X_train_tf, y_train)
# # y_pred = clf.predict(X_test_tf)
#
## Logistic Regression
from sklearn.linear_model import LogisticRegression
clf_lg = LogisticRegression(solver='liblinear') # class_weight={0:9,1:1}
# model = clf_lg.fit(X_train, y_train)
# y_pred = clf_lg.predict(X_test)

##  Neural Networks - Multi-layer Perceptron Model
from sklearn.neural_network import MLPClassifier
clf_nn = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
# clf_nn = MLPClassifier(solver='lbfgs', hidden_layer_sizes=[100], max_iter=2000, activation='logistic')
# model = clf.fit(X_train_tf, y_train)
# y_pred = clf.predict(X_test_tf)

## Naive Bayes Model
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
# model = nb.fit(X_train, y_train)
# y_pred = nb.predict(X_test)
# https://towardsdatascience.com/dont-sweat-the-solver-stuff-aea7cddc3451

## Random Forest
from sklearn.ensemble import RandomForestClassifier
clf_rf = RandomForestClassifier(max_depth=2, random_state=0)
# model = clf_rf.fit(X_train, y_train)
# y_pred = clf_rf.predict(X_test)


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
# model = gnb.fit(X_train, y_train)
# y_pred = gnb.predict(X_test)


from sklearn.model_selection import KFold
kf= KFold(n_splits=10, random_state=42, shuffle=True)

scores_accuracy = []
scores_precision = []
scores_recall = []
scores_f1 = []

y_testList=[]
y_predList=[]
conf_matrix_list_of_arrays= []
for train_index, test_index in kf.split(X, y):
   x_train, x_test = X[train_index], X[test_index]
   y_train, y_test = y[train_index], y[test_index]

   X_train = tfidf.fit_transform(x_train)
   X_test = tfidf.transform(x_test)

   model = clf_lg.fit(X_train, y_train)
   y_pred = clf_lg.predict(X_test)
   y_predList.extend(y_pred)
   y_testList.extend(y_test)

   score_accuracy = metrics.accuracy_score(y_test, y_pred)
   score_precision = metrics.precision_score(y_test, y_pred)
   score_recall = metrics.recall_score(y_test, y_pred)
   score_f1 = metrics.f1_score(y_test, y_pred)

   scores_accuracy.append(score_accuracy)
   scores_precision.append(score_precision)
   scores_recall.append(score_recall)
   scores_f1.append(score_f1)

   conf_matrix = metrics.confusion_matrix(y_test, y_pred)
   conf_matrix_list_of_arrays.append(conf_matrix)

   # print(metrics.classification_report(y_test, y_pred, labels=[1, 0]))


   # print(" ")
   # print(X_train.shape)
   # print(X_test.shape)

# print (" ")
# print("Mean Accuracy: ", sum(scores_accuracy)/len(scores_accuracy))
# print("Mean Precision: ", sum(scores_precision)/len(scores_precision))
# print("Mean Recall: ", sum(scores_recall)/len(scores_recall))
# print("Mean F1: ", sum(scores_f1)/len(scores_f1))
# print (" ")
# print ("confusion matrix mean: ")
# print (np.mean(conf_matrix_list_of_arrays, axis=0))

# print (" ")
# tn, fp, fn, tp = confusion_matrix(y_testList, y_predList).ravel()
# print("(tn, fp, fn, tp)")
# print(tn, fp, fn, tp)


# print(metrics.classification_report(y_testList, y_predList, labels=[1, 0]))


# print("-------------")
## Another way of getting the metrics for each class
# from sklearn.metrics import precision_recall_fscore_support as score
# precision, recall, fscore, support = score(y_test, y_pred)
# print('precision: {}'.format(precision))
# print('recall: {}'.format(recall))
# print('fscore: {}'.format(fscore))
# print('support: {}'.format(support))


# print ("False Positives")
# ccc= (x_test[(y_pred == 1) & (y_test == 0)])
# for each in ccc:
#     print(each)
# print (" ")
# print ("False negatives")
# print(x_test[(y_pred == 0) & (y_test == 1)])


# X_train_tf = tfidf.fit_transform(X_train)
# model = clf_lg.fit(X_train_tf, y_train)
#
# mySentence = tfidf.transform(["How do this"])
#
# from sklearn.externals import joblib
# joblib.dump(model, 'ML_model.pkl')
# joblib.dump(tfidf, 'tfidf.pkl')
#
# fittedModel = joblib.load('ML_model.pkl')
# print(fittedModel.predict(mySentence))

# https://kavita-ganesan.com/news-classifier-with-logistic-regression-in-python/#Feature-Representation
# https://www.quora.com/Why-does-TF-term-frequency-sometimes-give-better-F-scores-than-TF-IDF-does-for-text-classification
# https://towardsdatascience.com/dont-sweat-the-solver-stuff-aea7cddc3451