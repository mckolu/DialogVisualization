from gensim.models import KeyedVectors
import os
import  nltk
import numpy as np
import re
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import sent_tokenize
import inflect # To convert the digits to text
from nltk.stem import PorterStemmer
porter_stemmer = PorterStemmer()
p = inflect.engine()
pd.set_option("display.max_columns", 10)
pd.set_option('display.width', 100)
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))
new_words = ["\"", ".\"", "\"?", ".'", "...", "okay","randomly","yeah", "like", "large", "also", "iv", "one", "two", "new", "previously", "okay"]
stop_words = stop_words.union(new_words)


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
    cleanedTokens = ([p.number_to_words(each) if each.isdigit() else each.lower() for each in tokenized]) # Convert digits to text
    stemmed_words = [porter_stemmer.stem(word=word) for word in cleanedTokens]

    sentence = " ".join(stemmed_words)
    return sentence


def ExtractData():
    folder_path = r"/Users/mehmetcelepkolu/Desktop/Google Drive/Engage UF/Transcription Files/ENGAGE Coded Transcriptions/"

    fileNameList = os.listdir(folder_path)
    if ('Icon\r') in fileNameList:
        fileNameList.remove('Icon\r')
    if ('.DS_Store') in fileNameList:
        fileNameList.remove('.DS_Store')

    df_List = []
    for file in fileNameList:
        file_location = folder_path + file
        data_df1 = (pd.read_excel(file_location, sheet_name=0))
        df_import = data_df1[['Speaker', 'Text', 'QuestionCode']]
        df_import = df_import.assign(File=file) # Create a File column with file name values

        df_List.append(df_import)
    all_dfs = pd.concat(df_List).reset_index()

    corpus=""
    for index, row in all_dfs.iterrows():
        cleanedTextList = str(row["Text"]).replace("...", ". ").replace("(", " ").replace(")", " ").replace("-", " ")
        cleanedTextList1 = (sent_tokenize(cleanedTextList))
        # # Check if one row contains multiple questions
        if len(cleanedTextList1) > 0:
            for each in cleanedTextList1:
                corpus = corpus + " " + str(each)
        else:
            corpus = corpus+" "+str(cleanedTextList1)

    corpus = (cleanAgain(corpus))
    return corpus

def w2v(corpus):
    from gensim.models import word2vec
    # tokenize sentences in corpus
    wpt = nltk.WordPunctTokenizer()
    tokenized_corpus = [wpt.tokenize(corpus)]

    # Set values for various parameters
    feature_size = 300  # Word vector dimensionality
    window_context = 5  # Context window size
    min_word_count = 1  # Minimum word count
    sample = 1e-3  # Downsample setting for frequent words

    w2v_model = word2vec.Word2Vec(tokenized_corpus, size=feature_size,
                                  window=window_context, min_count=min_word_count,
                                  sample=sample, iter=50)
    w2v_model.save("w2v_model.bin")
    print(w2v_model.wv.most_similar("beak"))
    # print(w2v_model.wv.similarity("clone", "beak"))
    # similarities = w2v_model.wv.most_similar(positive=['beak', 'for'], negative=['bird'])
    # print(similarities)


# def fastT(corpus):
#     from gensim.models import FastText
#     wpt = nltk.WordPunctTokenizer()
#     tokenized_corpus = [wpt.tokenize(corpus)]
#
#
#     model_ted = FastText(tokenized_corpus, size=100, window=5, min_count=5, workers=4, sg=1)
#
#     model_ted.wv.most_similar("bird")
#     # model.train(sentences=corpus, total_examples=len(corpus), epochs=10)  # train
#
#     # print(model.most_similar("length"))

# w2v(ExtractData())




model = KeyedVectors.load('w2v_model.bin')
# print(model.wv['becaus'] )

sentence1 = 'really good'
sentence2 = 'because'

def sentenceSimilarity(sent_1, sent_2):
    sentence1 = sent_1.lower().split(" ")
    sentence2 = sent_2.lower().split(" ")

    sentence1 = [porter_stemmer.stem(word=word) for word in sentence1]
    sentence2 = [porter_stemmer.stem(word=word) for word in sentence2]

    word_vectors = model.wv

    vector_1 = np.mean([model.wv[word] for word in (sentence1) if word in word_vectors], axis=0)
    vector_2 = np.mean([model.wv[word] for word in (sentence2) if word in word_vectors], axis=0)

    # vector_1 = np.mean([model.wv[word] for word in (sentence1)], axis=0)
    # vector_2 = np.mean([model.wv[word] for word in (sentence2)], axis=0)

    if np.isnan(np.min(vector_1)) or np.isnan(np.min(vector_2)):
        cos_similarity = 0.01

    else:
        cos_similarity = np.dot(vector_1, vector_2)/(np.linalg.norm(vector_1)* np.linalg.norm(vector_2))

    return (abs(cos_similarity))



# print(sentenceSimilarity("barcelona", "change sugar level"))


#Convert all vector representations of words in word2vec to real words
# index2word=model.wv.index2word

