import os
import numpy as np
import re
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import sent_tokenize
import inflect # To convert the digits to text
p = inflect.engine()
pd.set_option("display.max_columns", 200)
pd.set_option('display.width', 320)
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))
new_words = ["\"", ".\"", "\"?", ".'", "...", "okay","randomly","yeah", "like", "large", "also", "iv", "one", "two", "new", "previously", "okay"]
stop_words = stop_words.union(new_words)

pd.options.mode.chained_assignment = None

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
    cleanedTokens = ([p.number_to_words(each) if each.isdigit() else each for each in tokenized]) # Convert digits to text
    sentence = " ".join(cleanedTokens)
    return sentence

# file1 = open("MyFile.txt","w")

def ExtractData():
    folder_path = r"/Users/mehmetcelepkolu/Desktop/Google Drive/Engage UF/Transcription Files/ENGAGE Coded Transcriptions/"
    # folder_path = r"/Users/mehmetcelepkolu/Desktop/Google Drive/Celepkolu-Boyer Research Share/Conferences/EDM 2019/prototype3/QuestionTestData/"

    fileNameList = os.listdir(folder_path)
    print(fileNameList)
    if ('Icon\r') in fileNameList:
        fileNameList.remove('Icon\r')
    if ('.DS_Store') in fileNameList:
        fileNameList.remove('.DS_Store')
    # frames = []

    dfList = []
    for file in fileNameList:
        # print(file)
        file_location = folder_path + file
        data_df1 = (pd.read_excel(file_location, sheet_name=0))
        df_import = data_df1[['Speaker', 'Text', 'QuestionCode']]
        df_import = df_import.assign(File=file) # Create a File column with file name values

        df_filtered = df_import.loc[df_import['QuestionCode'].isin(['C','O'])]

    # frames.append(df_filtered)
    # df = pd.concat(frames).reset_index()

        aresults = []
        for index, row in df_filtered.iterrows():
            cleanedTextList = str(row["Text"]).replace("...", ". ").replace("(", "").replace(")", "").replace("-", " ")

            cleanedTextList1 = (sent_tokenize(cleanedTextList))
            # print(cleanedTextList1)
            result = [x for x in cleanedTextList1 if str(x).endswith("?")]
            # # Check if one row contains multiple questions
            if len(result)>0:
                # print(row["File"], result)
            #     # file1.write(row["File"] +": " + str(result) +"\n")
            #     # file1.write("\n")
                result1 = cleanAgain(result[0])
            else:
                result1 = "DROP THIS"
            # print(result1)

            aresults.append(result1)

        df_filtered['ExtractedQuestions'] = aresults

        df = df_filtered[['File', 'Speaker', 'Text', 'ExtractedQuestions','QuestionCode']]

        for index, row in df.iterrows():
            if row['ExtractedQuestions'] == "DROP THIS":
                # print(index)
                df.drop(df.index[1], inplace=True)
                df = df.reset_index(drop=True)
            else:
                continue

        df = df.drop(df[df['QuestionCode'] == "C"].sample(frac=.688, random_state=1).index) # Downsample -.681 is the best

        df_new = df[['File', 'ExtractedQuestions', 'QuestionCode']]

        doc = np.asarray(df_new['ExtractedQuestions'].tolist())

        df = df.reset_index()
        dfList.append(df)
    return (dfList)

# for each in ExtractData():
#     print(each)

# file1.close()

# end = time.time()
# print(end - start)


## TODO: Create machine learning functions that take df['ExtractedQuestions'] and df['QuestionCode'] and take care of the rest
