import W2V_API
import numpy as np
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))
new_words = ["\"", ".\"", "\"?", ".'", "...", "okay","oh", "yep","randomly","yah","yeah", "like", "large", "also", "iv", "one", "two", "new", "previously", "okay"]
stop_words = stop_words.union(new_words)
import pickle

def keywordExtraction(text):
    from rake_nltk import Metric, Rake
    from nltk.corpus import stopwords
    # r = Rake()  # Uses stopwords for english from NLTK, and all puntuation characters.Please note that "hello" is not included in the list of stopwords.

    r = Rake(ranking_metric=Metric.DEGREE_TO_FREQUENCY_RATIO)
    r = Rake(ranking_metric=Metric.WORD_DEGREE)
    r = Rake(ranking_metric=Metric.WORD_FREQUENCY)
    r= Rake(stop_words, min_length=1, max_length=3)
    # r = Rake("SmartStoplist.txt", min_length=1, max_length=2)

    a = r.extract_keywords_from_text(text)
    # b = r.get_ranked_phrases()
    b = r.get_ranked_phrases()[:25]
    c = r.get_ranked_phrases_with_scores()

    myList=b

    # return (myList[:15])
    evilList =[]
    for each in myList:
        for b in myList:
            sim = (W2V_API.sentenceSimilarity(each, b))
            if (each !=b) and (sim>0.9):
                # myList.remove(each)
                # break
                evilList.append(each)
                evilList.append(b)

                myList.remove(b)
                # print("This is to be removed: ", each, " ",b, sim)
            else: continue

    return (myList[:15])



    # new  = sorted(candidate.items(), key=lambda kv: kv[1])[-10:]
    # a = [each[0] for each in new]
    # print(candidate)

    # print(myList)



    # myList=b
    # x = {}
    # for each in myList:
    #     for i in range(len(myList)):
    #         if (each !=myList[i]):
    #             forDic = str(each+" & "+myList[i])
    #             sim = (W2V_API.sentenceSimilarity(each, myList[i]))
    #             # print(forDic, sim)
    #             z = {forDic:sim}
    #             x.update(z)
    #
    # new  = sorted(x.items(), key=lambda kv: kv[1])[-10:]
    # a = [each[0] for each in new]



# text = "Oh, shoot, yeah it is. How's it going? Um, the internet. It say trumpet started lagging again froze surprising. We've got the code on there. So, just... quickly, you know? Yeah. This is harder than you think it is. The way we did it. All right. So, we got to go back there, right? All right. Then, that. It's all good. We're going to live. It's okay. It's all not. It's not for a grade. It's okay. What did we make our variables again. Ah. I got you. What the heck? I'm not, like, really good at it. But, I'm pretty good at it. Yeah. There's our level. Then, like, all state. Oh, yeah. All state's, like... I think all state is, like, slightly higher I forget which one. Do you know Oh, yeah,  Oh, Amy's also there. Amy is also there. Amy Powers? Yeah. No. She does not... she's not playing. Yes, she does. Flute. Amy can play. And, so does plays trump... wait, no- Trombone. No, trombone. I was about to say trumpet. But, it's trombone. Oh, yeah. older brother plays trumpet. So. Nani. Okay, wave. Oh, my god. You type so slowly. I know, I'm not into You have to press okay. Just press okay. I know, but... press like, I press that- Smart one. Really. That amplitude. All right. What are the things? Oh, my god. No, what are the... what are the"
# print(keywordExtraction(text))

# Algorithm
# https://www.thinkinfi.com/2018/09/keyword-extraction-using-rake-in-python.html


# https://medium.com/analytics-vidhya/automated-keyword-extraction-from-articles-using-nlp-bfd864f41b34



