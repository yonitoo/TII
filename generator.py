#############################################################################
### Търсене и извличане на информация. Приложение на дълбоко машинно обучение
### Стоян Михов
### Зимен семестър 2020/2021
#############################################################################
###
### Домашно задание 3
###
#############################################################################

import numpy as np
import torch

def generateText(model, char2id, startSentence, limit = 300, temperature = 0.7):
    # model е инстанция на обучен LSTMLanguageModelPack обект
    # char2id е речник за символите, връщащ съответните индекси
    # startSentence е началния низ стартиращ със символа за начало '{'
    # limit е горна граница за дължината на поемата
    # temperature е температурата за промяна на разпределението за следващ символ
    
    id2char = dict(enumerate(char2id))

    def predict(model, source, h=None):
        
        length = len(source) - 1
        X = model.preparePaddedBatch(source)
        E = model.embed(X)
        source_lengths = [len(s) for s in source]

        if h!=None:
            outputPacked, h = model.lstm(torch.nn.utils.rnn.pack_padded_sequence(E, source_lengths, enforce_sorted = False), h)
        else:
            outputPacked, h = model.lstm(torch.nn.utils.rnn.pack_padded_sequence(E, source_lengths, enforce_sorted = False))
        output,_ = torch.nn.utils.rnn.pad_packed_sequence(outputPacked)

        Z = model.projection(model.dropout(output.flatten(0,1)))
        p = torch.nn.functional.softmax(torch.div(Z, temperature), dim=1).data
        p, topChar = p.topk(32)
        topChar = topChar.numpy().squeeze()
        p = p[length]
        p = p.numpy().squeeze()
        if type(topChar[length]) is np.ndarray:
            t = np.random.choice(topChar[length], p = p / p.sum())
        else:
            t = np.random.choice(topChar, p = p / p.sum())
        return id2char[t],h 

    if(len(startSentence)==1):
        keys = list(char2id.keys())
        glavniBukvi = keys[51:79]
        startSentence+= np.random.choice(glavniBukvi)
    else:
        startSentence += " "
    result = startSentence[1:]
    startSentenceLen = len(result)
    chars  = [x for x in result]
    result = ""
    output, h = predict(model, chars)
    chars.append(output)
    model.eval()
    for i in range(limit):
        output, h = predict(model, chars[-startSentenceLen:], h)
        chars.append(output)

    for ch in chars:
        result += ch

    #t = result[0]
    #sz = 0

    #while not t == '}' and sz <= limit :
    #    p = torch.nn.functional.softmax(Z/temperature), Z = projection
    #    t = np.random.choice(P(t|result)) 
    #    result = result + t
    #    sz = sz + 1

    return result
