#############################################################################
### Търсене и извличане на информация. Приложение на дълбоко машинно обучение
### Стоян Михов
### Зимен семестър 2020/2021
#############################################################################
###
### Домашно задание 3
###
#############################################################################

import torch

#################################################################
####  LSTM с пакетиране на партида
#################################################################

class LSTMLanguageModelPack(torch.nn.Module):
    def preparePaddedBatch(self, source):
        device = next(self.parameters()).device
        m = max(len(s) for s in source)
        sents = [[self.word2ind.get(w,self.unkTokenIdx) for w in s] for s in source]
        sents_padded = [ s+(m-len(s))*[self.padTokenIdx] for s in sents]
        return torch.t(torch.tensor(sents_padded, dtype=torch.long, device=device))
    
    def save(self,fileName):
        torch.save(self.state_dict(), fileName)
    
    def load(self,fileName):
        self.load_state_dict(torch.load(fileName))

    def __init__(self, embed_size, hidden_size, word2ind, unkToken, padToken, endToken, lstm_layers, dropout):
        super(LSTMLanguageModelPack, self).__init__()
        #############################################################################
        ###  Тук следва да се имплементира инициализацията на обекта
        ###  За целта може да копирате съответния метод от програмата за упр. 13
        ###  като направите добавки за повече слоеве на РНН и dropout
        #############################################################################
        #### Начало на Вашия код.

        self.word2ind    = word2ind
        self.unkTokenIdx = word2ind[unkToken]
        self.padTokenIdx = word2ind[padToken]
        self.endTokenIdx = word2ind[endToken]
        self.hidden_size = hidden_size
        self.lstm        = torch.nn.LSTM(embed_size, hidden_size, lstm_layers, bidirectional = True, dropout = dropout)
        self.embed       = torch.nn.Embedding(len(word2ind), embed_size)
        self.projection  = torch.nn.Linear(2 * hidden_size, len(word2ind))
        self.dropout     = torch.nn.Dropout(dropout)

        #### Край на Вашия код
        #############################################################################

    def forward(self, source):
        #############################################################################
        ###  Тук следва да се имплементира forward метода на обекта
        ###  За целта може да копирате съответния метод от програмата за упр. 13
        ###  като направите добавка за dropout
        #############################################################################
        #### Начало на Вашия код.

        batch_size = len(source)
        X = self.preparePaddedBatch(source)
        E = self.embed(X)
        
        source_lengths = [len(s) for s in source]
        m = X.shape[0]
        outputPacked, _ = self.lstm(torch.nn.utils.rnn.pack_padded_sequence(E, source_lengths,enforce_sorted=False))
        
        output,_ = torch.nn.utils.rnn.pad_packed_sequence(outputPacked)
        output = output.view(m, batch_size, 2, self.hidden_size)
        t = torch.cat((output[:-2,:,0,:], output[2:,:,1,:]),2)
        Z = self.projection(self.dropout(t.flatten(0,1)))

        Y_bar = X[1:-1].flatten(0,1)
        Y_bar[Y_bar==self.endTokenIdx] = self.padTokenIdx
        H = torch.nn.functional.cross_entropy(Z,Y_bar,ignore_index=self.padTokenIdx)
        return H
    
        #### Край на Вашия код
        #############################################################################

