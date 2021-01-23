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

def generateText(model, char2id, startSentence, limit=1000, temperature=1.):
    # model е инстанция на обучен LSTMLanguageModelPack обект
    # char2id е речник за символите, връщащ съответните индекси
    # startSentence е началния низ стартиращ със символа за начало '{'
    # limit е горна граница за дължината на поемата
    # temperature е температурата за промяна на разпределението за следващ символ
    
    #result = startSentence[1:]
    int2char = dict(enumerate(char2id))
    def char_tensor (model, string):
        tensor = torch.zeros(len(string)).long()
        for c in range (len(string)):
            tensor[c] = char2id[string[c]]
            return tensor

    def predict(model, source, h=None):
        
        length = len(source)
        X = model.preparePaddedBatch(source)
        #   print(X)
        E = model.embed(X)
        source_lengths = [len(s) for s in source]
       # print(E)
        #print(source_lengths)
        if h!=None:
            outputPacked, h = model.lstm(torch.nn.utils.rnn.pack_padded_sequence(E, source_lengths,enforce_sorted=False), h)
        else:
            outputPacked, h = model.lstm(torch.nn.utils.rnn.pack_padded_sequence(E, source_lengths,enforce_sorted=False))
        output,_ = torch.nn.utils.rnn.pad_packed_sequence(outputPacked)

        Z = model.projection(model.dropout(output.flatten(0,1)))

        #print(Z)
        #Z = torch.div(Z,0.2)
        #print(Z)
        p = torch.nn.functional.softmax(Z/temperature, dim=1).data
        #print(p)
        p, top_ch = p.topk(32)
        top_ch = top_ch.numpy().squeeze()
        #print(top_ch)        
        #print(len(p[0]))
        p = p[length]
        #print(p)
        #print(top_ch)
        #print(len(p))
        p = p.numpy().squeeze()
        #print(len(p))
        #print(top_ch[length], p)
        t = np.random.choice(top_ch[length], p=p/p.sum())
        #print(char, int2char[char])
       # Y_bar = X.flatten(0,1)
        #print(Z.size(), Z)
       #Y_bar[Y_bar==model.endTokenIdx] = model.padTokenIdx
        #H = torch.nn.functional.cross_entropy(Z,Y_bar,ignore_index=model.padTokenIdx)
        return int2char[t],h 
    #print(char2id)
    startSentence+=" "
    result = startSentence[1:]
    startSentenceLen = len(result)
    chars  = [x for x in result]
    out, h = predict(model,chars)
    #print(h)
    chars.append(out)
    model.eval() # eval mode
    for x in range(limit):
        out, h = predict(model, chars[-startSentenceLen:],h)
        chars.append(out)
    return "".join(chars)

    #############################################################################
    ###  Тук следва да се имплементира генерацията на текста
    #############################################################################
    #### Начало на Вашия код.

    #t = result[0]
    #sz = 0

    #while not t == '}' and sz <= limit :
    #    p = torch.nn.functional.softmax(Z/temperature), Z = projection
    #    t = np.random.choice(P(t|result)) 
    #    result = result + t
    #    sz = sz + 1
    #
    #### Край на Вашия код
    #############################################################################

    return result
