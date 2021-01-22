import torch

corpusFileName = 'corpusPoems'
modelFileName = 'modelLSTM'
trainDataFileName = 'trainData'
testDataFileName = 'testData'
char2idFileName = 'char2id'

device = torch.device("cuda:0")
#device = torch.device("cpu")

batchSize = 32
char_emb_size = 32

hid_size = 512
lstm_layers = 2
dropout = 0.75

epochs = 3
learning_rate = 0.005

defaultTemperature = 0.4
