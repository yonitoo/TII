#############################################################################
### Търсене и извличане на информация. Приложение на дълбоко машинно обучение
### Стоян Михов
### Зимен семестър 2020/2021
#############################################################################
###
### Упражнение 13
###
#############################################################################

import sys
import nltk
from nltk.corpus import PlaintextCorpusReader
import numpy as np
import torch
import random
import math


#############################################################
###  Визуализация на прогреса
#############################################################
class progressBar:
    def __init__(self ,barWidth = 50):
        self.barWidth = barWidth
        self.period = None
    def start(self, count):
        self.item=0
        self.period = int(count / self.barWidth)
        sys.stdout.write("["+(" " * self.barWidth)+"]")
        sys.stdout.flush()
        sys.stdout.write("\b" * (self.barWidth+1))
    def tick(self):
        if self.item>0 and self.item % self.period == 0:
            sys.stdout.write("-")
            sys.stdout.flush()
        self.item += 1
    def stop(self):
        sys.stdout.write("]\n")

def extractDictionary(corpus, limit=20000):
    pb = progressBar()
    pb.start(len(corpus))
    dictionary = {}
    for doc in corpus:
        pb.tick()
        for w in doc:
            if w not in dictionary: dictionary[w] = 0
        dictionary[w] += 1
    L = sorted([(w,dictionary[w]) for w in dictionary], key = lambda x: x[1] , reverse=True)
    if limit > len(L): limit = len(L)
    words = [ w for w,_ in L[:limit] ] + [unkToken] + [padToken]
    word2ind = { w:i for i,w in enumerate(words)}
    pb.stop()
    return words, word2ind

def splitSentCorpus(fullSentCorpus, testFraction = 0.1):
    random.seed(42)
    random.shuffle(fullSentCorpus)
    testCount = int(len(fullSentCorpus) * testFraction)
    testSentCorpus = fullSentCorpus[:testCount]
    trainSentCorpus = fullSentCorpus[testCount:]
    return testSentCorpus, trainSentCorpus


#############################################################
#######   Зареждане на корпуса
#############################################################

corpus_root = 'JOURNALISM.BG/C-MassMedia'
myCorpus = PlaintextCorpusReader(corpus_root, '.*\.txt')
startToken = '<s>'
endToken = '</s>'
unkToken = '<unk>'
padToken = '<pad>'

corpus = [ [startToken] + [w.lower() for w in sent] + [endToken] for sent in myCorpus.sents()]
words, word2ind = extractDictionary(corpus)

testCorpus, trainCorpus  = splitSentCorpus(corpus, testFraction = 0.01)

batchSize = 32
emb_size = 50
hid_size = 100

#device = torch.device("cpu")
#device = torch.device("cuda:0")
device = torch.device("cuda:1")

#################################################################
#### LSTM с пакетиране на партида
#################################################################

class LSTMLanguageModelPack(torch.nn.Module):
    def __init__(self, embed_size, hidden_size, word2ind, unkToken, padToken):
        super(LSTMLanguageModelPack, self).__init__()
        self.word2ind = word2ind
        self.unkTokenIdx = word2ind[unkToken]
        self.padTokenIdx = word2ind[padToken]
        self.lstm = torch.nn.LSTM(embed_size, hidden_size)
        self.embed = torch.nn.Embedding(len(word2ind), embed_size)
        self.projection = torch.nn.Linear(hidden_size,len(word2ind))
    
    def preparePaddedBatch(self, source):
        device = next(self.parameters()).device
        m = max(len(s) for s in source)
        sents = [[self.word2ind.get(w,self.unkTokenIdx) for w in s] for s in source]
        sents_padded = [ s+(m-len(s))*[self.padTokenIdx] for s in sents]
        return torch.t(torch.tensor(sents_padded, dtype=torch.long, device=device))

    def forward(self, source):
        X = self.preparePaddedBatch(source)
        E = self.embed(X[:-1])
        source_lengths = [len(s)-1 for s in source]
        outputPacked, _ = self.lstm(torch.nn.utils.rnn.pack_padded_sequence(E, source_lengths,enforce_sorted=False))
        output,_ = torch.nn.utils.rnn.pad_packed_sequence(outputPacked)

        Z = self.projection(output.flatten(0,1))
        Y_bar = X[1:].flatten(0,1)
        H = torch.nn.functional.cross_entropy(Z,Y_bar,ignore_index=self.padTokenIdx)
        return H


lm = LSTMLanguageModelPack(emb_size, hid_size, word2ind, unkToken, padToken).to(device)
optimizer = torch.optim.Adam(lm.parameters(), lr=0.01)

idx = np.arange(len(trainCorpus), dtype='int32')
np.random.shuffle(idx)

for b in range(0, len(idx), batchSize):
    batch = [ trainCorpus[i] for i in idx[b:min(b+batchSize, len(idx))] ]
    H = lm(batch)
    optimizer.zero_grad()
    H.backward()
    optimizer.step()
    if b % 10 == 0:
        print(b, '/', len(idx), H.item())

def perplexity(lm, testCorpus, batchSize):
    H = 0.
    c = 0
    for b in range(0,len(testCorpus),batchSize):
        batch = testCorpus[b:min(b+batchSize, len(testCorpus))]
        l = sum(len(s)-1 for s in batch)
        c += l
        with torch.no_grad():
            H += l * lm(batch)
    return math.exp(H/c)

#################################################################
####  Двупосочен LSTM с пакетиране на партида
#################################################################

class BiLSTMLanguageModelPack(torch.nn.Module):
    def __init__(self, embed_size, hidden_size, word2ind, unkToken, padToken, endToken):
        super(BiLSTMLanguageModelPack, self).__init__()
        self.word2ind = word2ind
        self.unkTokenIdx = word2ind[unkToken]
        self.padTokenIdx = word2ind[padToken]
        self.endTokenIdx = word2ind[endToken]
        self.hidden_size = hidden_size
        self.lstm = torch.nn.LSTM(embed_size, hidden_size, bidirectional=True)
        self.embed = torch.nn.Embedding(len(word2ind), embed_size)
        self.projection = torch.nn.Linear(2*hidden_size,len(word2ind))

    def preparePaddedBatch(self, source):
        device = next(self.parameters()).device
        m = max(len(s) for s in source)
        sents = [[self.word2ind.get(w,self.unkTokenIdx) for w in s] for s in source]
        sents_padded = [ s+(m-len(s))*[self.padTokenIdx] for s in sents]
        return torch.t(torch.tensor(sents_padded, dtype=torch.long, device=device))

    def forward(self, source):
        batch_size = len(source)
        X = self.preparePaddedBatch(source)
        E = self.embed(X)
        
        source_lengths = [len(s) for s in source]
        m = X.shape[0]
        outputPacked, _ = self.lstm(torch.nn.utils.rnn.pack_padded_sequence(E, source_lengths,enforce_sorted=False))
        
        output,_ = torch.nn.utils.rnn.pad_packed_sequence(outputPacked)
        output = output.view(m, batch_size, 2, self.hidden_size)
        t = torch.cat((output[:-2,:,0,:], output[2:,:,1,:]),2)
        Z = self.projection(t.flatten(0,1))

        Y_bar = X[1:-1].flatten(0,1)
        Y_bar[Y_bar==self.endTokenIdx] = self.padTokenIdx
        H = torch.nn.functional.cross_entropy(Z,Y_bar,ignore_index=self.padTokenIdx)
        return H

blm = BiLSTMLanguageModelPack(emb_size, hid_size, word2ind, unkToken, padToken, endToken).to(device)
optimizer = torch.optim.Adam(blm.parameters(), lr=0.01)

idx = np.arange(len(trainCorpus), dtype='int32')
np.random.shuffle(idx)

for b in range(0, len(idx), batchSize):
    batch = [ trainCorpus[i] for i in idx[b:min(b+batchSize, len(idx))] ]
    H = blm(batch)
    optimizer.zero_grad()
    H.backward()
    optimizer.step()
    if b % 10 == 0:
        print(b, '/', len(idx), H.item())

def perplexity(blm, testCorpus, batchSize):
    H = 0.
    c = 0
    for b in range(0,len(testCorpus),batchSize):
        batch = testCorpus[b:min(b+batchSize, len(testCorpus))]
        l = sum(len(s)-2 for s in batch)
        c += l
        with torch.no_grad():
            H += l * blm(batch)
    return math.exp(H/c)


#################################################################
#### LSTM класификатор на документи
#################################################################

class LSTMClassifier(torch.nn.Module):
    def __init__(self, langModel, classesCount):
        super(LSTMClassifier, self).__init__()
        self.langModel = langModel
        self.classProjection = torch.nn.Linear(langModel.lstm.hidden_size,classesCount)
    
    def forward(self, source):
        X = self.langModel.preparePaddedBatch(source)
        E = self.langModel.embed(X[:-1])
        source_lengths = [len(s)-1 for s in source]
        _, (h,_) = self.langModel.lstm(torch.nn.utils.rnn.pack_padded_sequence(E, source_lengths,enforce_sorted=False))
        
        Z = self.classProjection(torch.squeeze(h,dim=0))
        return Z

fileNames = myCorpus.fileids()

ecoCorpus = [ [startToken] + [w.lower() for w in myCorpus.words(f)] + [endToken] for f in fileNames if f.find('E-Economy'+'/')==0 ]
milCorpus = [ [startToken] + [w.lower() for w in myCorpus.words(f)] + [endToken] for f in fileNames if f.find('S-Military'+'/')==0 ]
polCorpus = [ [startToken] + [w.lower() for w in myCorpus.words(f)] + [endToken] for f in fileNames if f.find('J-Politics'+'/')==0 ]
culCorpus = [ [startToken] + [w.lower() for w in myCorpus.words(f)] + [endToken] for f in fileNames if f.find('C-Culture'+'/')==0 ]

testEcoCorpus, trainEcoCorpus = splitSentCorpus(ecoCorpus)
testMilCorpus, trainMilCorpus = splitSentCorpus(milCorpus)
testPolCorpus, trainPolCorpus = splitSentCorpus(polCorpus)
testCulCorpus, trainCulCorpus = splitSentCorpus(culCorpus)

trainClassCorpus = trainEcoCorpus + trainMilCorpus + trainPolCorpus + trainCulCorpus

trainY = np.concatenate((
                         np.ones(len(trainEcoCorpus),dtype='int32')*0,
                         np.ones(len(trainMilCorpus),dtype='int32')*1,
                         np.ones(len(trainPolCorpus),dtype='int32')*2,
                         np.ones(len(trainCulCorpus),dtype='int32')*3
                         ))

testY = np.concatenate((
                        np.ones(len(testEcoCorpus),dtype='int32')*0,
                        np.ones(len(testMilCorpus),dtype='int32')*1,
                        np.ones(len(testPolCorpus),dtype='int32')*2,
                        np.ones(len(testCulCorpus),dtype='int32')*3
                        ))

idx = np.arange(len(trainClassCorpus), dtype='int32')

classModel = LSTMClassifier(lm,4).to(device)
optimizer = torch.optim.Adam(classModel.parameters(), lr=0.01)

np.random.shuffle(idx)
for b in range(0, len(idx), batchSize):
    batch = [ trainClassCorpus[i] for i in idx[b:min(b+batchSize, len(idx))] ]
    target = torch.tensor(trainY[idx[b:min(b+batchSize, len(idx))]], dtype = torch.long, device = device)

    Z = classModel(batch)
    H = torch.nn.functional.cross_entropy(Z,target)

    optimizer.zero_grad()
    H.backward()
    optimizer.step()
    if b % 10 == 0:
        print(b, '/', len(idx), H.item())



testClassCorpus = [ testEcoCorpus, testMilCorpus, testPolCorpus, testCulCorpus ]

def gamma(s):
    with torch.no_grad():
        Z = classModel([s])
        return torch.argmax(Z[0]).item()


def testClassifier(testClassCorpus, gamma):
    L = [ len(c) for c in testClassCorpus ]
    pb = progressBar(50)
    pb.start(sum(L))
    classesCount = len(testClassCorpus)
    confusionMatrix = [ [0] * classesCount for _ in range(classesCount) ]
    for c in range(classesCount):
        for text in testClassCorpus[c]:
            pb.tick()
            c_MAP = gamma(text)
            confusionMatrix[c][c_MAP] += 1
    pb.stop()
    precision = []
    recall = []
    Fscore = []
    for c in range(classesCount):
        extracted = sum(confusionMatrix[x][c] for x in range(classesCount))
        if confusionMatrix[c][c] == 0:
            precision.append(0.0)
            recall.append(0.0)
            Fscore.append(0.0)
        else:
            precision.append( confusionMatrix[c][c] / extracted )
            recall.append( confusionMatrix[c][c] / L[c] )
            Fscore.append((2.0 * precision[c] * recall[c]) / (precision[c] + recall[c]))
    P = sum( L[c] * precision[c] / sum(L) for c in range(classesCount) )
    R = sum( L[c] * recall[c] / sum(L) for c in range(classesCount) )
    F1 = (2*P*R) / (P + R)
    print('=================================================================')
    print('Матрица на обърквания: ')
    for row in confusionMatrix:
        for val in row:
            print('{:4}'.format(val), end = '')
        print()
    print('Прецизност: '+str(precision))
    print('Обхват: '+str(recall))
    print('F-оценка: '+str(Fscore))
    print('Обща презизност: '+str(P))
    print('Общ обхват: '+str(R))
    print('Обща F-оценка: '+str(F1))
    print('=================================================================')
    print()


#################################################################
#### Двупосочен LSTM класификатор на документи
#################################################################

class BiLSTMClassifier(torch.nn.Module):
    def __init__(self, langModel, classesCount):
        super(BiLSTMClassifier, self).__init__()
        self.langModel = langModel
        self.classProjection = torch.nn.Linear(2*langModel.hidden_size,classesCount)
    
    def forward(self, source):
        batch_size = len(source)
        X = self.langModel.preparePaddedBatch(source)
        E = self.langModel.embed(X)
        source_lengths = [len(s) for s in source]
        _, (h,c) = self.langModel.lstm(torch.nn.utils.rnn.pack_padded_sequence(E, source_lengths,enforce_sorted=False))
        h = h.view(2,batch_size,self.langModel.hidden_size)
        
        Z = self.classProjection(torch.cat([h[0],h[1]],1))
        return Z

classModel = BiLSTMClassifier(blm,4).to(device)
optimizer = torch.optim.Adam(classModel.parameters(), lr=0.01)

idx = np.arange(len(trainClassCorpus), dtype='int32')
np.random.shuffle(idx)
for b in range(0, len(idx), batchSize):
    batch = [ trainClassCorpus[i] for i in idx[b:min(b+batchSize, len(idx))] ]
    target = torch.tensor(trainY[idx[b:min(b+batchSize, len(idx))]], dtype = torch.long, device = device)
    
    Z = classModel(batch)
    H = torch.nn.functional.cross_entropy(Z,target)
    
    optimizer.zero_grad()
    H.backward()
    optimizer.step()
    if b % 10 == 0:
        print(b, '/', len(idx), H.item())


#################################################################
#### Конволюционен класификатор на документи
#################################################################

class ConvolutionClassifier(torch.nn.Module):
    def __init__(self, embed, filterSize, filterCount, classesCount, word2ind, unkToken, padToken):
        super(ConvolutionClassifier, self).__init__()
        self.embed = embed
        self.word2ind = word2ind
        self.unkTokenIdx = word2ind[unkToken]
        self.padTokenIdx = word2ind[padToken]
        self.convolution = torch.nn.Conv1d(in_channels=embed.embedding_dim, out_channels=filterCount, kernel_size=filterSize)
        self.dropout = torch.nn.Dropout(0.5)
        self.classProjection = torch.nn.Linear(filterCount,classesCount)
    
    def preparePaddedBatch(self, source):
        device = next(self.parameters()).device
        m = max(len(s) for s in source)
        sents = [[self.word2ind.get(w,self.unkTokenIdx) for w in s] for s in source]
        sents_padded = [ s+(m-len(s))*[self.padTokenIdx] for s in sents]
        return torch.tensor(sents_padded, dtype=torch.long, device=device)
    
    def forward(self, source):
        X = self.preparePaddedBatch(source)
        
        E = torch.transpose(self.embed(X),1,2)
        ### Очаква се Е да е тензор с размер (batch_size, embed_size, max_sent_len)

        U,_ = torch.max(torch.relu(self.convolution(E)), dim=2)
        Z = self.classProjection(self.dropout(U))
        return Z

EMB = lm.embed

classModel = ConvolutionClassifier(EMB, 7, 400, 4, word2ind, unkToken, padToken).to(device)
optimizer = torch.optim.Adam(classModel.parameters(), lr=0.01, weight_decay=0.0002)

idx = np.arange(len(trainClassCorpus), dtype='int32')
classModel.train()
for epoch in range(10):
    np.random.shuffle(idx)
    for b in range(0, len(idx), batchSize):
        batch = [ trainClassCorpus[i] for i in idx[b:min(b+batchSize, len(idx))] ]
        target = torch.tensor(trainY[idx[b:min(b+batchSize, len(idx))]], dtype = torch.long, device = device)
    
        Z = classModel(batch)
        H = torch.nn.functional.cross_entropy(Z,target)
    
        optimizer.zero_grad()
        H.backward()
        optimizer.step()
        if b % 10 == 0:
            print(b, '/', len(idx), H.item())
classModel.eval()
testClassifier(testClassCorpus, gamma)



#################################################################
#### LSTM с посимволово влагане с КНН и пакетиране на партида
#################################################################

class CharEmbedding(torch.nn.Module):
    def __init__(self, word2ind, char_embed_size, word_embed_size, filter_size=5, dropoutrate=0.3, padding=1):
        super(CharEmbedding, self).__init__()
        self.word2ind = word2ind
        self.char_embed_size = char_embed_size
        self.word_embed_size = word_embed_size
        self.filter_size = filter_size
        self.dropoutrate = dropoutrate
        self.padding = padding

        alphabetSet = {c for w in word2ind for c in w}
        alphabet = ['§','`','~','№']+list(alphabetSet)
        self.char2id = {c:i for i, c in enumerate(alphabet) }
        self.char_pad = self.char2id['§']
        self.start_of_word = self.char2id['`']
        self.end_of_word = self.char2id['~']
        self.char_unk = self.char2id['№']

        self.CharEmbedding = torch.nn.Embedding(len(self.char2id),self.char_embed_size, padding_idx = self.char_pad)
        self.conv = torch.nn.Conv1d(char_embed_size, word_embed_size, filter_size, padding=padding)
        self.highway_proj = torch.nn.Linear(word_embed_size,word_embed_size)
        self.highway_gate = torch.nn.Linear(word_embed_size,word_embed_size)

        self.Dropout = torch.nn.Dropout(dropoutrate)

    def preparePaddedBatch(self, source):
        device = next(self.parameters()).device
        source_ids = [[ [self.start_of_word] + [self.char2id.get(c, self.char_unk) for c in w ] + [self.end_of_word] for w in s] for s in source]

        max_word_length = max(len(w) for s in source_ids for w in s )
        max_sent_len = max(len(s) for s in source_ids)
    
        sents_padded = []
        for sentence in source_ids:
            sent_padded = [ w + [self.char_pad]*(max_word_length-len(w)) for w in sentence ] + [[self.char_pad]*max_word_length] * (max_sent_len - len(sentence))
            sents_padded.append(sent_padded)

        return torch.transpose(torch.tensor(sents_padded, dtype=torch.long, device=device),0,1).contiguous()

    def forward(self, source):
        batch_size = len(source)
        X = self.preparePaddedBatch(source)
        X_emb = self.CharEmbedding(X).transpose(2,3)

        x_conv = self.conv(X_emb.flatten(0,1))
        x_conv_out0,_ = torch.max(torch.nn.functional.relu(x_conv),dim=2)
        x_conv_out = x_conv_out0.view((-1,batch_size,self.word_embed_size))

        x_proj = torch.nn.functional.relu(self.highway_proj(x_conv_out))
        x_gate = torch.sigmoid(self.highway_gate(x_conv_out))
        x_highway = x_gate * x_proj + (1 - x_gate) * x_conv_out

        output = self.Dropout(x_highway)
        return output


class CharCNNLSTMLanguageModelPack(torch.nn.Module):
    def __init__(self, word_embed_size, hidden_size, word2ind, unkToken, padToken, char_embed_size, filter_size=5, dropoutrate=0.3, padding=1):
        super(CharCNNLSTMLanguageModelPack, self).__init__()
        self.word2ind = word2ind
        self.unkTokenIdx = word2ind[unkToken]
        self.padTokenIdx = word2ind[padToken]

        self.charEmbedding = CharEmbedding(word2ind, char_embed_size, word_embed_size, filter_size, dropoutrate, padding)
        self.lstm = torch.nn.LSTM(word_embed_size, hidden_size)
        self.projection = torch.nn.Linear(hidden_size,len(word2ind))
    
    def preparePaddedBatch(self, source):
        device = next(self.parameters()).device
        m = max(len(s) for s in source)
        sents = [[self.word2ind.get(w,self.unkTokenIdx) for w in s] for s in source]
        sents_padded = [ s+(m-len(s))*[self.padTokenIdx] for s in sents]
        return torch.t(torch.tensor(sents_padded, dtype=torch.long, device=device))
    
    def forward(self, source):
        X = self.preparePaddedBatch(source)
        E = self.charEmbedding(source)
        source_lengths = [len(s)-1 for s in source]
        outputPacked, _ = self.lstm(torch.nn.utils.rnn.pack_padded_sequence(E, source_lengths,enforce_sorted=False))
        output,_ = torch.nn.utils.rnn.pad_packed_sequence(outputPacked)
        
        Z = self.projection(output.flatten(0,1))
        Y_bar = X[1:].flatten(0,1)
        H = torch.nn.functional.cross_entropy(Z,Y_bar,ignore_index=self.padTokenIdx)
        return H

lm = CharCNNLSTMLanguageModelPack(256, 256, word2ind, unkToken, padToken, 32).to(device)
optimizer = torch.optim.Adam(lm.parameters(), lr=0.001)

idx = np.arange(len(trainCorpus), dtype='int32')
np.random.shuffle(idx)

lm.train()
for b in range(0, len(idx), batchSize):
    batch = [ trainCorpus[i] for i in idx[b:min(b+batchSize, len(idx))] ]
    H = lm(batch)
    optimizer.zero_grad()
    H.backward()
    optimizer.step()
    if b % 10 == 0:
        print(b, '/', len(idx), H.item())
lm.eval()
perplexity(lm, testCorpus, batchSize)


#################################################################
####  Двупосочен LSTM с пакетиране на партида
#################################################################

class CharCNNBiLSTMLanguageModelPack(torch.nn.Module):
    def __init__(self, word_embed_size, hidden_size, word2ind, unkToken, padToken, endToken, char_embed_size, filter_size=5, dropoutrate=0.3, padding=1):
        super(CharCNNBiLSTMLanguageModelPack, self).__init__()
        self.word2ind = word2ind
        self.unkTokenIdx = word2ind[unkToken]
        self.padTokenIdx = word2ind[padToken]
        self.endTokenIdx = word2ind[endToken]
        self.hidden_size = hidden_size

        self.charEmbedding = CharEmbedding(word2ind, char_embed_size, word_embed_size, filter_size, dropoutrate, padding)
        self.lstm = torch.nn.LSTM(word_embed_size, hidden_size, bidirectional=True)
        self.projection = torch.nn.Linear(2*hidden_size,len(word2ind))
    
    def preparePaddedBatch(self, source):
        device = next(self.parameters()).device
        m = max(len(s) for s in source)
        sents = [[self.word2ind.get(w,self.unkTokenIdx) for w in s] for s in source]
        sents_padded = [ s+(m-len(s))*[self.padTokenIdx] for s in sents]
        return torch.t(torch.tensor(sents_padded, dtype=torch.long, device=device))
    
    def forward(self, source):
        batch_size = len(source)
        X = self.preparePaddedBatch(source)
        E = self.charEmbedding(source)

        source_lengths = [len(s) for s in source]
        m = X.shape[0]
        outputPacked, _ = self.lstm(torch.nn.utils.rnn.pack_padded_sequence(E, source_lengths,enforce_sorted=False))
        
        output,_ = torch.nn.utils.rnn.pad_packed_sequence(outputPacked)
        output = output.view(m, batch_size, 2, self.hidden_size)
        t = torch.cat((output[:-2,:,0,:], output[2:,:,1,:]),2)
        Z = self.projection(t.flatten(0,1))
        
        Y_bar = X[1:-1].flatten(0,1)
        Y_bar[Y_bar==self.endTokenIdx] = self.padTokenIdx
        H = torch.nn.functional.cross_entropy(Z,Y_bar,ignore_index=self.padTokenIdx)
        return H

blm = CharCNNBiLSTMLanguageModelPack(256, 256, word2ind, unkToken, padToken, endToken, 32).to(device)
optimizer = torch.optim.Adam(blm.parameters(), lr=0.001)

idx = np.arange(len(trainCorpus), dtype='int32')
np.random.shuffle(idx)

blm.train()
for b in range(0, len(idx), batchSize):
    batch = [ trainCorpus[i] for i in idx[b:min(b+batchSize, len(idx))] ]
    H = blm(batch)
    optimizer.zero_grad()
    H.backward()
    optimizer.step()
    if b % 10 == 0:
        print(b, '/', len(idx), H.item())



