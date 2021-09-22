import os

import matplotlib
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from collections import Counter
import pandas as pd
import re
from matplotlib import pyplot as plt

matplotlib.rc('figure', figsize=(10, 5))
porter = PorterStemmer()
path = "output/"
FILE_PATH = "sms-spam-corpus.csv"
STOP_LIST = ["a", "the", "to", "in"]
spam = pd.read_csv(FILE_PATH, encoding='cp1251')
figuresCount = 0
hamWordsArr = []
spamWordsArr = []
spamWordsLengthArr = []
hamWordsLengthArr = []
hamSentenceLengthArr = []
hamSentenceArr = []
spamSentenceArr = []
spamSentenceLengthArr = []


def write(array, name):
    filePath = path + name + '.txt'
    with open(filePath, "w+") as f:
        for k, v in array.most_common():
            f.write("{}: {}\n".format(k, v))


def stemSentence(sentence):
    token_words = word_tokenize(sentence)
    stem_sentence = []
    for word in token_words:
        stem_sentence.append(porter.stem(word))
        stem_sentence.append(" ")
    return "".join(stem_sentence)


def wordFromStopListCheck(word):
    for stopWord in STOP_LIST:
        pattern = r'\b' + stopWord + r'\b'
        word = (re.sub(pattern, '', word))
    return word


def wordsArrayFill(sentance, messageType):
    for word in sentance.split(' '):
        if len(word.strip()) == 0: continue
        hamWordsArr.append(word) if messageType == 'ham' else spamWordsArr.append(word)


def wordsLengthArrayFill(sentance, messageType):
    for word in sentance.split(' '):
        if len(word.strip()) == 0: continue
        hamWordsLengthArr.append(len(word)) if messageType == 'ham' else spamWordsLengthArr.append(len(word))


def sentanceArrayFill(sentance, messageType):
    hamSentenceArr.append(sentance) if messageType == 'ham' else spamSentenceArr.append(sentance)


def sentanceLengthArrayFill(sentance, messageType):
    hamSentenceLengthArr.append(len(sentance)) if messageType == 'ham' else spamSentenceLengthArr.append(len(sentance))


def arrayElementsAverageLength(array):
    elementsLength = 0
    elementNumber = len(array)
    for element in array:
        elementsLength += len(element)
    return elementsLength / elementNumber


def normalize(list):
    counter = 0
    for val in list:
        counter += val

    for i, val in enumerate(list):
        list[i] = val / counter
    return list


def barBuild(firstData, secondData, thirdData=None, title=None, type=None):
    global figuresCount
    figuresCount += 1
    plt.figure(figuresCount)
    plt.suptitle(title)

    if type == 'bar':
        ax1 = plt.subplot(1, 2, 1)
        ax1.title.set_text("ham info")
        plt.bar(firstData[0], firstData[1])
        plt.legend(["words info"])
        plt.xticks(rotation=45)

        ax2 = plt.subplot(1, 2, 2)
        ax2.title.set_text("spam info")
        plt.bar(secondData[0], secondData[1])
        plt.legend(["words info"])
        plt.xticks(rotation=45)
    else:
        ax1 = plt.subplot(1, 2, 1)
        ax1.title.set_text("ham info")
        firstData.sort(reverse=True)
        plt.plot(firstData)
        plt.axhline(thirdData, color='r')
        plt.legend(['ham', 'agv size'])

        ax2 = plt.subplot(1, 2, 2)
        ax2.title.set_text("spam info")
        secondData.sort(reverse=True)
        plt.plot(secondData)
        plt.axhline(thirdData, color='r')
        plt.legend(['spam', 'agv size'])

    plt.savefig(path + title + '.png')


for value in spam.values:
    messageType = value[0]
    text = re.sub("[^a-zA-Z ]+", "", value[1]).lower()
    stemmedText = stemSentence(text)
    stemmedText = wordFromStopListCheck(stemmedText)
    wordsArrayFill(stemmedText, messageType)
    sentanceArrayFill(stemmedText, messageType)
    sentanceLengthArrayFill(stemmedText, messageType)
    wordsLengthArrayFill(stemmedText, messageType)

if not os.path.exists(path):
    os.mkdir(path)

counterHam = Counter(hamWordsArr)
counterSpam = Counter(spamWordsArr)

counterHamFirstTwenty = counterHam.most_common(20)
counterSpamFirstTwenty = counterSpam.most_common(20)

hamWordsAvgLength = arrayElementsAverageLength(hamWordsArr)
spamWordsAvgLength = arrayElementsAverageLength(spamWordsArr)
wordsAvgLength = (hamWordsAvgLength + spamWordsAvgLength) / 2

hamSentenceAvgLength = arrayElementsAverageLength(hamSentenceArr)
spamSentenceAvgLength = arrayElementsAverageLength(spamSentenceArr)
sentenceAvgLength = (hamSentenceAvgLength + spamSentenceAvgLength) / 2

namesHam, valuesHam = zip(*counterHamFirstTwenty)
namesSpam, valuesSpam = zip(*counterSpamFirstTwenty)

hamValues = normalize(list(valuesHam))
spamValues = normalize(list(valuesSpam))

barBuild([namesHam, hamValues], [namesSpam, spamValues], title="20 most common words", type='bar')
barBuild(hamWordsLengthArr, spamWordsLengthArr, wordsAvgLength, title="Words length")
barBuild(hamSentenceLengthArr, spamSentenceLengthArr, sentenceAvgLength, title="Sentence length")

plt.show()

write(counterHam, 'counterHam')
write(counterSpam, 'counterSpam')
