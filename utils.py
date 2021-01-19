#############################################################################
### Търсене и извличане на информация. Приложение на дълбоко машинно обучение
### Стоян Михов
### Зимен семестър 2020/2021
##########################################################################
###
### Домашно задание 3
###
#############################################################################

import random

corpusSplitString = '@\n'
maxPoemLength = 10000
symbolCountThreshold = 100

def splitSentCorpus(fullSentCorpus, testFraction = 0.1):
    random.seed(42)
    random.shuffle(fullSentCorpus)
    testCount = int(len(fullSentCorpus) * testFraction)
    testSentCorpus = fullSentCorpus[:testCount]
    trainSentCorpus = fullSentCorpus[testCount:]
    return testSentCorpus, trainSentCorpus

def getAlphabet(corpus):
    symbols={}
    for s in corpus:
        for c in s:
            if c in symbols: symbols[c] += 1
            else: symbols[c]=1
    return symbols

def prepareData(corpusFileName, startChar, endChar, unkChar, padChar):
    file = open(corpusFileName,'r',encoding="utf8")
    poems = file.read().split(corpusSplitString)
    symbols = getAlphabet(poems)
    
    assert startChar not in symbols and endChar not in symbols and unkChar not in symbols and padChar not in symbols
    charset = [startChar,endChar,unkChar,padChar] + [c for c in sorted(symbols) if symbols[c] > symbolCountThreshold]
    char2id = { c:i for i,c in enumerate(charset)}
    
    corpus = []
    for i,s in enumerate(poems):
        if len(s) > 0:
            corpus.append( [startChar] + [ s[i] for i in range(min(len(s),maxPoemLength)) ] + [endChar] )

    testCorpus, trainCorpus  = splitSentCorpus(corpus, testFraction = 0.01)
    print('Corpus loading completed.')
    return testCorpus, trainCorpus, char2id
