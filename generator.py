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
    
    result = startSentence[1:]

    #############################################################################
    ###  Тук следва да се имплементира генерацията на текста
    #############################################################################
    #### Начало на Вашия код.

    t = result[0]
    sz = 0

    while not t == '}' and sz <= limit :
        #p = softmax(Z/temperature), Z = projection u13 : 490 sigmoid = softmax?
        #
        #t = np.random.choice(P(t|result)) 
        result = result + t
        sz = sz + 1

    #### Край на Вашия код
    #############################################################################

    return result
