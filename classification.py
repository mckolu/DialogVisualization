import os
import keywordExtract
import re
import graphs
import timestamps
import numpy as np
import pandas as pd
# from sklearn.externals import joblib
import joblib
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
porter_stemmer = PorterStemmer()
import inflect
p = inflect.engine()
pd.set_option("display.max_columns", 10)
pd.set_option('display.width', 100)

def clean(text1):
    text = re.sub("[\(\[].*?[\)\]]", "", text1) # Change everything between [**]
    text = text.lower() \
        .replace("(())", "UNC") \
        .replace("...", " ") \
        .replace("(", "").replace(")", "").replace("-", " ") \
        .replace("student says partner's name", "PARTNERNAME") \
        .replace("student says own name", "OWNNAME") \
        .replace("student mentions partner's name", "PARTNERNAME") \
        .replace("mentions partner's name", "PARTNERNAME") \
        .replace("mentions another student's name", "NAME") \
        .replace("student mentions teacher name", "NAME") \
        .replace("<", "").replace(">", "").replace("[", "").replace("]", "")
    return text

def cleanAgain(text):
    tokenized = RegexpTokenizer(r'\w+').tokenize(clean(text))
    # cleanedTokens1 = ([each if each not in stop_words else "" for each in tokenized])
    cleanedTokens = ([p.number_to_words(each) if each.isdigit() else each.lower() for each in tokenized])
    # Convert digits to text
    # stemmed_words = [porter_stemmer.stem(word=word) for word in cleanedTokens]
    sentence = " ".join(cleanedTokens)
    return sentence

def classify_utterance(utt):
    loaded_vectorizer = joblib.load('tfidf.pkl') # load the vectorizer
    loaded_model = joblib.load('ML_model.pkl') # load the model
    return (loaded_model.predict(loaded_vectorizer.transform([utt]))[0]) # make a prediction



def data():
    dfs=list()
    import os
    dir_path = os.path.dirname(os.path.realpath(__file__))
    folder_path = dir_path + "/data/"
    # print(folder_path)

    fileNameList = os.listdir(folder_path)
    # fileNameList = ([x if x.endswith("xlsx", 4, 8) else "" for x in fileNameList1])
    if ('Icon\r') in fileNameList:
        fileNameList.remove('Icon\r')
    if ('.DS_Store') in fileNameList:
        fileNameList.remove('.DS_Store')


    for file in fileNameList:
        file_location = folder_path + '/' + file
        data_df1 = (pd.read_excel(file_location, sheet_name=0))
        df_import = data_df1[['Timestamp','TotalSecond', 'Role', 'Speaker', 'Text','QuestionCount', 'S1_Q_Count', 'S2_Q_Count']]

        # df_import['Speaker'] = df_import['Speaker'].replace('S1', 'S0new')
        # df_import['Speaker'] = df_import['Speaker'].replace('S2', 'S1')
        # df_import['Speaker'] = df_import['Speaker'].replace('S0new', 'S2')

        df_new = pd.DataFrame(columns=['GroupName', 'Cleaned','Tokenized','WordCount','S1_words','S1_words_Count','S2_words', 'S2_words_Count', 'Other_words', 'Other_words_Count'])

        S1_Q1_Count_List = []
        S1_Q0_Count_List = []
        for index, row in df_import.iterrows():
            cleanedText = (clean(str(row["Text"])))
            # cleanedTextList = str(row["Text"]).replace("...", ". ").replace("(", "").replace(")", "").replace("-", " ")
            from nltk.tokenize import sent_tokenize
            cleanedTextList1 = (sent_tokenize(cleanedText))

            total0 = 0
            total1 = 0
            if (row["Speaker"] == "S1"):
                for each in cleanedTextList1:
                    if str(each).endswith("?"):
                        qType = classify_utterance(each)
                        # print(each, ": ", qType)
                        if qType==1:
                            total1 = total1+1
                            # print(each, ": ", qType)
                        if qType==0:
                            total0 = total0+1
                            # print(each, ": ", qType)

                S1_Q1_Count_List.append(total1)
                S1_Q0_Count_List.append(total0)

            else:
                S1_Q1_Count_List.append(0)
                S1_Q0_Count_List.append(0)

            # print(row["Speaker"], cleanedTextList1, "0: ", total0)
            # print(row["Speaker"], cleanedTextList1, "1: ", total1)
            # print("----")

        # print(len(S1_Q1_Count_List))
        # print(len(S1_Q0_Count_List))
        # print(S1_Q0_Count_List)

            tokenized = RegexpTokenizer(r'\w+').tokenize(cleanedText)
            wordCount = len(tokenized)
            if (row["Speaker"] == "S1"):
                df_new = df_new.append({'GroupName': file, 'Cleaned': cleanedText, 'Tokenized': tokenized, 'WordCount': wordCount, 'S1_words':tokenized,'S2_words':"", 'S1_words_Count':wordCount, 'S2_words_Count':"", 'Other_words': "", 'Other_words_Count': ""},ignore_index=True)
            elif (row["Speaker"] == "S2"):
                df_new = df_new.append({'GroupName': file, 'Cleaned': cleanedText, 'Tokenized': tokenized, 'WordCount': wordCount, 'S1_words': "",'S2_words':tokenized, 'S1_words_Count':"", 'S2_words_Count': wordCount, 'Other_words': "", 'Other_words_Count': ""}, ignore_index=True)
            else:
                df_new = df_new.append({'GroupName': file, 'Cleaned': cleanedText, 'Tokenized': tokenized, 'WordCount': wordCount, 'Other_words': tokenized, 'Other_words_Count': wordCount}, ignore_index=True)

        df = pd.concat([df_import, df_new], axis=1)
        df["S1_Q1_Count"] = S1_Q1_Count_List
        df["S1_Q0_Count"] = S1_Q0_Count_List
        dfs.append(df)
        # df.to_csv(file+'.csv', index=False)

    return dfs
# print(data())
#
#
# def table(dfs):
#     table_df = pd.DataFrame()
#     for df in dfs:
#         SpeakersList = df['Speaker'].tolist()
#         QuestionsList = df['Question'].tolist()
#
#         S1_Turn = int(sum([1 if x=="S1" else 0 for x in SpeakersList]))
#         S2_Turn = int(sum([1 if x == "S2" else 0 for x in SpeakersList]))
#         Other_Turn = int(sum([0 if (x == "S2" or x == "S1") else 1 for x in SpeakersList]))
#
#         S1_Q = int(sum([1 if x == "S1Q" else 0 for x in QuestionsList]))
#         S2_Q = int(sum([1 if x == "S2Q" else 0 for x in QuestionsList]))
#         Other_Q = int(sum([0 if (x == "S2" or x == "S1" or x == "") else 1 for x in QuestionsList]))
#
#         GroupName = df['GroupName'][0]
#
#         S1_Total=(pd.to_numeric(df.S1_words_Count, errors='coerce').fillna(0).astype(np.int64)).sum()
#         S2_Total=(pd.to_numeric(df.S2_words_Count, errors='coerce').fillna(0).astype(np.int64)).sum()
#         Other =  (pd.to_numeric(df.Other_words_Count, errors='coerce').fillna(0).astype(np.int64)).sum()
#         WordCount_Total = (pd.to_numeric(df.WordCount, errors='coerce').fillna(0).astype(np.int64)).sum()
#
#         table_df = table_df.append({'GroupName': GroupName, 'S1_TurnCount': S1_Turn, 'S2_TurnCount':S2_Turn, 'Other_TurnCount': Other_Turn, 'Total_Word_Count': WordCount_Total,'S1_WordCount': S1_Total, 'S2_WordCount': S2_Total, 'Other_words_Count': Other, 'S1_Q_Count':S1_Q, 'S2_Q_Count':S2_Q, 'Other_Q_Count':Other_Q }, ignore_index=True)
#
#         # table_df.to_csv('Graphs/'+GroupName+'.csv', index=False)
#     return table_df
#
# # print(table(data()))
# # timestamps.temporal(data())
# # print (timestamps.temporal(data()))
# # graphs.graph_bokeh(timestamps.temporal(data()))

from flask import Flask, render_template, redirect, send_from_directory, request, jsonify
import json

app = Flask(__name__, static_url_path='/static', static_folder='static')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

@app.route('/')
def index():
    return graphs.graph_bokeh(timestamps.temporal(data()))

@app.route('/second', methods=['POST'])
def second():
    # Selected_Text = request.get_json()
    # It is a list
    Selected_Text = ''.join(request.get_json())
    # print("Selected_Text", Selected_Text)
    cleaned = cleanAgain(Selected_Text)
    # print("cleaned",cleaned)
    n = (keywordExtract.keywordExtraction(cleaned))
    # print("Rake:", n)

    # return json.dumps(Selected_Text)
    # return jsonify(result='Test Text')

    response = app.response_class(
        response=json.dumps(n),
        status=200,
        mimetype='application/json')
    return response


if __name__ == '__main__':
    app.run(host='localhost', debug=True, threaded=True)



print ("")
print ("")
print ("")
print ("")
print ("")

# end = time.time()
# print(end - start)




