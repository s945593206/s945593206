import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.python.keras import backend as K

l2Lambda = 5e-7
class Multi_Head_Attention(keras.Model):

    def __init__(self,inUnits,numberOfBlocks,outUnits):
        super().__init__()
        self.number = numberOfBlocks
        self.linerQ = [keras.layers.Dense(inUnits,kernel_regularizer=keras.regularizers.l2(l2Lambda)) for _ in range(numberOfBlocks)]
        self.linerK = [keras.layers.Dense(inUnits,kernel_regularizer=keras.regularizers.l2(l2Lambda)) for _ in range(numberOfBlocks)]
        self.linerV = [keras.layers.Dense(inUnits,kernel_regularizer=keras.regularizers.l2(l2Lambda)) for _ in range(numberOfBlocks)]
        self.attention = [keras.layers.Attention() for _ in range(numberOfBlocks)]
        self.outLiner = keras.layers.Dense(outUnits,kernel_regularizer=keras.regularizers.l2(l2Lambda))
        self.ln = keras.layers.LayerNormalization()
        self.dropout = keras.layers.Dropout(rate=0.2)

    def call(self, inputs, training=None, mask=None):
        QTensor = inputs[0]
        KTensor = inputs[1]
        VTensor = inputs[2]
        concatList = []
        for i in range(self.number):
            thisQTrans = self.linerQ[i](QTensor)
            thisVTrans = self.linerV[i](VTensor)
            thisKTrans = self.linerK[i](KTensor)
            thisAttentionT =self.attention[i]([thisQTrans,thisKTrans,thisVTrans])
            concatList.append(thisAttentionT)
        concatTensor = tf.concat(concatList,axis=-1)
        lnT = self.ln(concatTensor)
        outTensor = self.outLiner(lnT)
        return self.dropout(outTensor,training = training)


class FeedForward(keras.Model) :
    """
    In the call function,
    ### InputShape : [batchSize , concat(Si_1Tensor.shape[1], HiddenTensors.shape[1])]
    """

    def __init__(self,outputDim):
        super(FeedForward,self).__init__()
        self.dense0 = keras.layers.Dense(512,kernel_regularizer=keras.regularizers.l2(l2Lambda))
        self.bn0 = keras.layers.BatchNormalization()
        self.pr0 = keras.layers.PReLU()


        self.dense1 = keras.layers.Dense(256,kernel_regularizer=keras.regularizers.l2(l2Lambda))
        self.bn1 = keras.layers.BatchNormalization()
        self.pr1 = keras.layers.PReLU()


        self.dense2 = keras.layers.Dense(outputDim,kernel_regularizer=keras.regularizers.l2(l2Lambda))
        self.dropout = keras.layers.Dropout(rate=0.2)


    def call(self, inputs, training=None, mask=None):
        x = self.dense0(inputs)
        x = self.bn0(x,training=training)
        x = self.pr0(x)

        x = self.dense1(x)
        x = self.bn1(x,training = training)
        x = self.pr1(x)

        x = self.dense2(x)
        x = self.dropout(x, training=training)
        return x

class TransformerEncoder(keras.Model):

    def __init__(self,inUnits,outUnits):
        super().__init__()
        self.multiHead = Multi_Head_Attention(inUnits=inUnits,numberOfBlocks=8,outUnits=outUnits)
        self.ln0 = keras.layers.LayerNormalization()
        self.feedForward = FeedForward(outputDim=outUnits)
        self.ln1 = keras.layers.LayerNormalization()

    def call(self, inputs, training=None, mask=None):
        multiTensor = self.multiHead([inputs,inputs,inputs],training = training)
        add0Tensor = tf.add(inputs, multiTensor)
        ln0Tensor = self.ln0(add0Tensor)
        feedTensor = self.feedForward(ln0Tensor,training = training)
        add1Tensor = tf.add(feedTensor, ln0Tensor)
        ln1Tensor = self.ln1(add1Tensor)
        return ln1Tensor


class Transformer(keras.Model):

    def __init__(self,embeddingMatrix,labelsNumber,units = 256 ,numberOfTransformer = 6):
        self._transformerLayers = numberOfTransformer
        super().__init__()
        self.iniEmbeddingMatrix  = tf.convert_to_tensor(embeddingMatrix,dtype=tf.float32)
        self.denseTrans = keras.layers.Dense(units,kernel_regularizer=keras.regularizers.l2(l2Lambda))

        self.transformerList = [TransformerEncoder(inUnits=units,outUnits=units) for _ in range(numberOfTransformer)]

        self.flat = keras.layers.Flatten()

        self.dense0 = keras.layers.Dense(units // 2,kernel_regularizer=keras.regularizers.l2(l2Lambda))
        self.bn0 = keras.layers.BatchNormalization()
        self.pRelu = keras.layers.PReLU()
        self.dropout = keras.layers.Dropout(rate=0.2)

        self.dense1 = keras.layers.Dense(labelsNumber,kernel_regularizer=keras.regularizers.l2(l2Lambda))


    ### [batch , times ]
    def call(self, inputs, training=None, mask=None):
        batchTensor = tf.stop_gradient(tf.nn.embedding_lookup(params=self.iniEmbeddingMatrix, ids=inputs))
        # print(batchTensor.shape)
        # print(positionTensor.shape)
        denseTrans = self.denseTrans(batchTensor)

        thisTransformer = K.identity(denseTrans)
        for i in range(self._transformerLayers):
            thisTransformer = self.transformerList[i](thisTransformer,training=training)

        flattenTensor = self.flat(thisTransformer)

        dense0Tensor = self.dense0(flattenTensor)
        bn0Tensor = self.bn0(dense0Tensor,training = training)
        actT = self.pRelu(bn0Tensor)
        dropTensor = self.dropout(actT,training = training)

        dense1Tensor = self.dense1(dropTensor)
        return tf.nn.sigmoid(dense1Tensor)



if __name__ == "__main__":
    testInput = tf.ones(shape=[3,10],dtype=tf.int64)
    model = Transformer(np.ones(shape=[100,150],dtype=np.float32),labelsNumber=1)
    result = model(testInput,training = False)
    print(result)



