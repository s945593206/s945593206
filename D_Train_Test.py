import numpy as np
from C_LSTM import LSTM_Model
from C_BiLSTM import Bi_LSTM_Model
from C_BiLSTM_Attention import Bi_LSTM_Attention
from C_Transformer import Transformer
import sklearn.metrics as metrics
from tensorflow  import keras
import  re
import tensorflow as tf


###### Data config
weightMatrixPath = "./Data/weightMatrix.txt"
dataPath = "./Data/WashedData.txt"
embeddingSize = 256
batchSize = 64
maxWordsInOneSentence = 200
trainTestSplitRatio = 0.9
###### Training config
trainingOrTesting = "Test"
epoch = 6
saveParamPath = ""
####### Test config
testModeWeight = ".\\Bi_LSTM_Attention\\Bi_LSTM_Attention_0.7406181"


print("Loading weight matrix.")
vocabulary2idx = {}
weightMatrix = [np.zeros(shape=[embeddingSize],dtype=np.float32)]
with open(weightMatrixPath,"r",encoding="UTF-8") as wh:
    for i , line in enumerate(wh):
        oneLine = line.strip()
        word_vec = oneLine.split("\t")
        word = word_vec[0]
        vec = word_vec[1]
        vecS = vec.split(",")[0:-1]
        vocabulary2idx[word] = i + 1
        thisVec = []
        for num in vecS:
            thisVec.append(float(num))
        weightMatrix.append(np.array(thisVec,dtype=np.float32))
weightMatrix = np.array(weightMatrix,dtype=np.float32)
print("Loading completed.")
print("There are " + str(len(vocabulary2idx)) + " vocabularies in this file.")


############################
### Change model at here ###

#model = LSTM_Model(embeddingMatrix=weightMatrix,labelNumbers=1)
#model = Bi_LSTM_Model(embeddingMatrix=weightMatrix,labelNumbers=1)
model = Bi_LSTM_Attention(embeddingMatrix=weightMatrix,labelsNumber=1)
#model = Transformer(embeddingMatrix=weightMatrix,labelsNumber=1)

### Change model at here ###
############################



print("Loading Data.")
data = []
labels = []
with open(dataPath,"r",encoding="UTF-8") as rh:
    for line in rh:
        oneLine = line.strip()
        sentence_labels = oneLine.split("\t")
        if len(sentence_labels) > 1:
            if re.fullmatch(r"[01]", sentence_labels[1]) is not None:
                wordList = sentence_labels[0].split()
                thisSentence = []
                labels.append(float(sentence_labels[1]))
                #print(float(sentence_labels[1]))
                for word in wordList:
                    idx = vocabulary2idx[word]
                    thisSentence.append(idx)
                    if len(thisSentence) == maxWordsInOneSentence:
                        break
                if len(thisSentence) < maxWordsInOneSentence:
                    paddingZeros = maxWordsInOneSentence - len(thisSentence)
                    data.append(np.array(thisSentence + [0 for _ in range(paddingZeros)],dtype=np.int64))
                else:
                    data.append(np.array(thisSentence, dtype=np.int64))
lenData = len(data)
data = np.array(data,dtype=np.int64)
labels = np.array(labels, dtype=np.float32)
TrainData = data[0:int(lenData * trainTestSplitRatio),:]
TrainLabels = labels[0:int(lenData * trainTestSplitRatio)]
TestData = data[int(lenData * trainTestSplitRatio):,:]
TestLabels = labels[int(lenData * trainTestSplitRatio):]
print(TrainData)
print(TrainLabels)
print("The shape of weight matrix is :",weightMatrix.shape)
sampleWeight = []
for oneLabel in TrainLabels:
    if oneLabel == 0.:
        sampleWeight.append(0.5)
    else:
        sampleWeight.append(1.0)
sampleWeight = np.array(sampleWeight,dtype=np.float32)
print(sampleWeight)
#'binary_crossentropy'
if trainingOrTesting.lower() == "train":
    model.compile(optimizer=keras.optimizers.SGD(momentum=0.9,nesterov=True),
                  loss=tf.losses.BinaryCrossentropy(),
                  metrics=['acc'])
    valACC = 0.
    acc = []
    val_acc = []
    loss = []
    val_loss = []
    for e in range(epoch):
        print("epoch : ",e)
        if e >= 1:
            model.load_weights(saveParamPath + "_" + str(valACC))
        history = model.fit(TrainData, TrainLabels,
                                epochs=1,
                                batch_size=batchSize,
                                validation_data=(TestData, TestLabels),
                                sample_weight=sampleWeight
                                )
        valACC = history.history["val_acc"][0]
        ACC = history.history["acc"][0]
        LOSS = history.history["loss"][0]
        valLOSS = history.history['val_loss'][0]
        model.save_weights(saveParamPath + "_" + str(valACC))
        val_acc.append(valACC)
        acc.append(ACC)
        loss.append(LOSS)
        val_loss.append(valLOSS)
    import matplotlib.pyplot as plt
    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()

else:
    print("Predict.")
    model.load_weights(testModeWeight)
    predictNumpy = model.predict(TestData)
    predictSqueeze = np.squeeze(predictNumpy)
    predictLabels = np.zeros(shape=predictSqueeze.shape)
    predictLabels[np.where(predictSqueeze >= 0.5)] = 1
    confusionMatrix = metrics.confusion_matrix(TestLabels,predictLabels)
    acc = metrics.accuracy_score(y_true=TestLabels,y_pred=predictLabels)
    TP = confusionMatrix[0,0]
    FN = confusionMatrix[0,1]
    FP = confusionMatrix[1,0]
    TN = confusionMatrix[1,1]
    print("Recall is ",TP / (TP + FN) + 0.)
    print("Precision is ",TP / (TP + FP) + 0.)
    print("F1 is ",2 * TP / (2 * TP + FN + FP )  + 0.)
    print("Accuracy is ",acc)
















