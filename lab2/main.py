import os
import matplotlib
from nltk.stem import PorterStemmer
from collections import Counter
import pandas as pd
import re
import tkinter as tk
from tkinter import filedialog, simpledialog
from math import log

root = tk.Tk()
root.withdraw()

matplotlib.rc('figure', figsize=(10, 5))
porter = PorterStemmer()
path = "output/"
FILE_PATH = filedialog.askopenfilename()
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

def wordFromStopListCheck(word):
    for stopWord in STOP_LIST:
        pattern = r'\b' + stopWord + r'\b'
        word = (re.sub(pattern, '', word))
    return word


def sentNormalize(list):
    listSize = len(list)
    for i, val in enumerate(list):
        list[i] = i / listSize
    return list


def deepcopy_list(x):
    if isinstance(x, (str, bool, float, int)):
        return x
    else:
        return [deepcopy_list(y) for y in x]


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


def multiplyList(myList):
    result = 1
    for x in myList:
        result = result * x
    return result


def pMessage(message, category):
    text = re.sub("[^a-zA-Z ]+", "", message).lower()
    words = text.split(sep=' ')
    wordsCounter = counterHam
    if (category == 'spam'):
        wordsCounter = counterSpam
    p = 1
    noWordsCounter = 0

    for i in words:
        text = wordFromStopListCheck(text)
        if(text == " "):
            continue

        word = wordsCounter.get(i)
        if word is None:
            noWordsCounter += 1

    for i in words:
        word = wordsCounter.get(i)
        wordsSum = sum(list(wordsCounter.values()))

        if word is None:
            word = 1
        if noWordsCounter > 0:
            word += 1
            wordsSum += noWordsCounter
        p *= word / wordsSum
    return p


for value in spam.values:
    messageType = value[0]
    text = re.sub("[^a-zA-Z ]+", "", value[1]).lower()
    text = wordFromStopListCheck(text)
    wordsArrayFill(text, messageType)
    sentanceArrayFill(text, messageType)
    sentanceLengthArrayFill(text, messageType)
    wordsLengthArrayFill(text, messageType)

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

allValuesHam = list(counterHam.values())
allValuesSpam = list(counterSpam.values())

hamValues = normalize(list(valuesHam))
spamValues = normalize(list(valuesSpam))

hamValuesAll = normalize(allValuesHam)
spamValuesAll = normalize(allValuesSpam)

hamWordsCount = len(hamWordsArr)
spamWordsCount = len(spamWordsArr)
wordsCount = hamWordsCount + spamWordsCount

hamMessagesCount = len(hamSentenceArr)
spamMessagesCount = len(spamSentenceArr)
wordsMessages = hamMessagesCount + spamMessagesCount

pHam = log(hamMessagesCount / wordsMessages)
pSpam = log(spamMessagesCount / wordsMessages)

USER_INP = simpledialog.askstring(title="Input", prompt="Enter message:")

pWordHam = log(pMessage(USER_INP, 'ham'))
pWordSpam = log(pMessage(USER_INP, 'spam'))
hamRes = log(pHam * pWordHam)
spamRes = log(pSpam * pWordSpam)

print("P(ham) =", pHam)
print("P(spam) =", pSpam)
print("P(bodyText | ham) =", pWordHam)
print("P(bodyText | spam) =", pWordSpam)
print("Ham result = ", hamRes)
print("Ham result = ", spamRes)

if hamRes > spamRes:
    print("Message from ham category")
else:
    print("Message from spam category")

write(counterHam, 'counterHam')
write(counterSpam, 'counterSpam')
