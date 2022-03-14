import os
import re
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import sent_tokenize
import inflect # To convert the digits to text
p = inflect.engine()
pd.set_option("display.max_columns", 10)
pd.set_option('display.width', 100)
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))
new_words = ["\"", ".\"", "\"?", ".'", "...", "okay", "randomly","yeah", "like", "large", "also", "iv", "one", "two", "new", "previously", "okay"]
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
    cleanedTokens = ([p.number_to_words(each) if each.isdigit() else each for each in tokenized]) # Convert digits to text
    sentence = " ".join(cleanedTokens)
    return sentence

# file1 = open("MyFile.txt","w")

def ExtractData():
    # folder_path = r"/Users/mehmetcelepkolu/Desktop/Google Drive/Celepkolu-Boyer Research Share/Conferences/EDM 2019/prototype3/data/"
    import os
    dir_path = os.path.dirname(os.path.realpath(__file__))
    folder_path = dir_path + "/data/"
    print(folder_path)

    fileNameList = os.listdir(folder_path)
    if ('Icon\r') in fileNameList:
        fileNameList.remove('Icon\r')
    if ('.DS_Store') in fileNameList:
        fileNameList.remove('.DS_Store')
    frames = []
    for file in fileNameList:
        print("fileee:", file)
        file_location = folder_path + file
        data_df1 = (pd.read_excel(file_location, sheet_name=0))
        df_import = data_df1[['Speaker', 'Text', 'QuestionCode']]
        df_import = df_import.assign(File=file) # Create a File column with file name values

        df_filtered = df_import.loc[df_import['QuestionCode'].isin(['C','O'])]
        frames.append(df_filtered)
    df = pd.concat(frames).reset_index()
    # print(df)

    aresults = []
    for index, row in df.iterrows():
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

        aresults.append(result1)

    df['ExtractedQuestions'] = aresults


    df = df[['File', 'Speaker', 'Text', 'ExtractedQuestions','QuestionCode']]

    df_dropThis = df[df['ExtractedQuestions'] == "DROP THIS"]

    df = df.drop(df_dropThis.index, axis=0)

    df = df.reset_index(drop=True)



    # for index, row in df.iterrows():
    #     if row['ExtractedQuestions'] == "DROP THIS":
    #         print(index, ": ",row['ExtractedQuestions'])


    # df = df.drop(df[df['QuestionCode'] == "C"].sample(frac=.695, random_state=1).index) # Downsample -.681=85.47% ---- -.689=85.31%

    # print(df.head(1000).to_string())


    # print(df)
    df.to_csv('filee.csv', index=False)

    return df


ExtractData()

# file1.close()

# end = time.time()
# print(end - start)



