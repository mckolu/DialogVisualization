from sklearn.externals import joblib
import QuestionExtraction
from sklearn import metrics
import numpy as np

def classify_utterance(utt):
    # load the vectorizer
    loaded_vectorizer = joblib.load('tfidf.pkl')

    # load the model
    loaded_model = joblib.load('ML_model.pkl')

    # make a prediction
    return (loaded_model.predict(loaded_vectorizer.transform([utt]))[0])

# classify_utterance("How to do?")
# classify_utterance("What is this?")


def predict_test():
    # loaded_vectorizer = joblib.load('tfidf.pkl')  # load the vectorizer
    # loaded_model = joblib.load('ML_model.pkl')  # load the model

    df = QuestionExtraction.ExtractData()

    # print(df)

    y_test1 = np.asarray(df['QuestionCode'].tolist())
    y_test = np.asarray([1 if i == "C" else 0 for i in y_test1])

    y_pred = []
    for index, row in df.iterrows():
        Text = str(row["Text"])
        tag = classify_utterance(Text)

        # print(Text, ": ", tag)

        y_pred.append(tag)

    # score_accuracy = metrics.accuracy_score(y_test, y_pred)
    # score_precision = metrics.precision_score(y_test, y_pred)
    # score_recall = metrics.recall_score(y_test, y_pred)
    # score_f1 = metrics.f1_score(y_test, y_pred)

    print("")
    print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))
    print("Precision: ", metrics.precision_score(y_test, y_pred, average=None))
    print("Recall: ", metrics.recall_score(y_test, y_pred, average=None))
    print("F1: ", metrics.f1_score(y_test, y_pred, average=None))

    print(" ")


    print(metrics.confusion_matrix(y_test, y_pred))
    print("")

    print(metrics.classification_report(y_test, y_pred, labels=[1, 0]))

    # print ("False Positives")
    # ccc= ([(y_pred == 1) & (y_test == 0)])
    # for each in ccc:
    #     print(each)

    # print ("  ")
    # print ("False negatives")
    # print(x_test[(y_pred == 0) & (y_test == 1)])

predict_test()






