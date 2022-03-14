import graphs
import os
import pandas as pd
import nltk
import numpy as np
from nltk.tokenize import RegexpTokenizer
import matplotlib.pyplot as plt
import openpyxl
from collections import Counter
from pandas import Series

def temporal(dfs):
    # counter = 1
    dSeconds_dfs=list()
    for data_df in dfs:
        # name = "Group_"+str(counter)
        # counter=counter+1

        name = data_df['GroupName'][0]

        print ("----------------------------"+name+"----------------------------")

        Speakers = ["S1", "S2"]
        TimeSeconds = (pd.to_numeric(data_df.TotalSecond, errors='coerce').fillna(0).astype(np.int64)).tolist()

        duration = 180  # seconds

        # Sentences=[]
        Time = [x for x in range(duration)]
        sumList = [0] * duration
        Student1_UtteranceList = [0] * duration
        Student2_UtteranceList = [0] * duration
        difference = [0] * duration

        # index = 0
        # for eachsecond in range(max(TimeSeconds) - duration):
        #     sentence = ""
        #     for line in TimeSeconds:
        #         if eachsecond == line:
        #             sentence = data_df['Cleaned'][index]
        #             index = index + 1
        #     Sentences.append(sentence)

        # Optimize the code for sliding window
        # https://stackoverflow.com/questions/38507672/summing-elements-in-a-sliding-window-numpy
        # In[334]: mydata
        # Out[334]: array([4, 2, 3, 8, -6, 10])
        #
        # In[335]: np.convolve(mydata, np.ones(3, dtype=int), 'valid')
        # Out[335]: array([9, 13, 5, 12])

        for eachsecond in range(max(TimeSeconds)-duration):
            sum = 0
            Student1_Utterance = 0
            Student2_Utterance = 0
            for m in range(eachsecond, eachsecond + duration):
                if m in TimeSeconds:
                    for s in range(len(TimeSeconds)):
                        if m==TimeSeconds[s]:
                            sum = sum + data_df['WordCount'].tolist()[s]
                            if data_df['Speaker'].tolist()[s] == Speakers[0]:
                                Student1_Utterance = Student1_Utterance + data_df['WordCount'].tolist()[s]
                            elif data_df['Speaker'].tolist()[s] == Speakers[1]:
                                Student2_Utterance = Student2_Utterance + data_df['WordCount'].tolist()[s]

            difference1 = (Student1_Utterance-Student2_Utterance)
            difference.append(difference1)
            sumList.append(sum)
            Student1_UtteranceList.append(Student1_Utterance)
            Student2_UtteranceList.append(Student2_Utterance)
            Time.append(eachsecond+duration)

        dataSeconds_df = pd.DataFrame()
        dataSeconds_df['Time'] = Time
        dataSeconds_df['Student1_UtteranceList'] = Student1_UtteranceList
        dataSeconds_df['Student2_UtteranceList'] = Student2_UtteranceList
        dataSeconds_df['Difference'] = difference
        dataSeconds_df['sumList'] = sumList
        # dataSeconds_df['Sentences'] = Sentences

        dSeconds_dfs.append(dataSeconds_df)

        # dataSeconds_df.to_csv('Graphs/'+name+'dataSeconds_df.csv', index=False)

    return (dSeconds_dfs, dfs)





