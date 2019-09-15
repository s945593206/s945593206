import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.python.keras import backend as K


l1l2Lambda = 5e-7



class Attention(keras.layers.Layer):

    def __init__(self):
        super().__init__()
        self.dense0 = keras.layers.Dense(128)
        self.dense1 = keras.layers.Dense(1)

    ### [b,t,n]
    ### return shape [b,t,n]
    def call(self, inputs, training=None, mask=None):
        identityInput = K.identity(inputs)
        stacks = K.array_ops.unstack(inputs,axis=1)
        ### [b,n]
        tempList = []
        for oneTimeStepTensor in stacks:
            ### [b,1]
            bTensor = self.dense0(oneTimeStepTensor)
            bTensor = K.relu(bTensor)
            bTensor = self.dense1(bTensor)
            tempList.append(bTensor)
        ### [b,t,1]
        stackedTensor = K.array_ops.stack(tempList,axis=1)
        softMaxTensor = K.softmax(stackedTensor,axis=1)
        ###[b,t,1]
        weightEdTensor = K.math_ops.multiply(identityInput,softMaxTensor)
        unstack =K.array_ops.unstack(weightEdTensor,axis=1)
        temp1List = []
        for oneTensor in unstack:
            temp1List.append(oneTensor)
        return K.math_ops.add_n(temp1List)

class Bi_LSTM_Attention(keras.Model):

    def __init__(self,embeddingMatrix,labelsNumber,LSTMOutUnits = 256,LSTM_Cell_Number = 3):
        super().__init__()
        self.iniEmbeddingMatrix  = tf.convert_to_tensor(embeddingMatrix,dtype=tf.float32)
        self.forLstm = keras.layers.RNN([keras.layers.LSTMCell(LSTMOutUnits,
                                                               activation=tf.nn.softsign,
                                                               kernel_regularizer=keras.regularizers.l2(l=l1l2Lambda),
                                                               recurrent_regularizer=keras.regularizers.l2(l=l1l2Lambda),
                                                               dropout=0.2,recurrent_dropout=0.2)
                                         for _ in range(LSTM_Cell_Number)],
                                        return_sequences=True)
        self.backLstm = keras.layers.RNN([keras.layers.LSTMCell(LSTMOutUnits,
                                                            activation=tf.nn.softsign,
                                                            kernel_regularizer=keras.regularizers.l2(l=l1l2Lambda),
                                                            recurrent_regularizer=keras.regularizers.l2(l=l1l2Lambda),
                                                            dropout=0.2, recurrent_dropout=0.2) for _ in range(LSTM_Cell_Number)],
                                         go_backwards=True,return_sequences=True)
        self.attention = Attention()
        self.dense0 = keras.layers.Dense(LSTMOutUnits // 2,kernel_regularizer=keras.regularizers.l2(l=l1l2Lambda))
        self.bn0 = keras.layers.BatchNormalization()
        self.pRelu = keras.layers.PReLU()
        self.dense1 = keras.layers.Dense(labelsNumber,kernel_regularizer=keras.regularizers.l2(l=l1l2Lambda))


    ### [batch , times , embedding]
    def call(self, inputs, training=None, mask=None):
        ### BiLstm part
        batchTensor = tf.stop_gradient(tf.nn.embedding_lookup(params=self.iniEmbeddingMatrix, ids=inputs))
        forTensor = self.forLstm(batchTensor,initial_state=self.forLstm.get_initial_state(inputs=batchTensor),training = training)
        backTensor = self.backLstm(batchTensor,initial_state=self.backLstm.get_initial_state(inputs=batchTensor),training = training)
        encoderTensor = tf.multiply(0.5,tf.add(forTensor,backTensor))
        ### attention part
        attentionEncoder = self.attention(encoderTensor)
        ### decoder part
        #print(attentionEncoder.shape)
        dense0Tensor = self.dense0(attentionEncoder)
        bn0Tensor = self.bn0(dense0Tensor,training = training)
        actT = self.pRelu(bn0Tensor)
        dense1Tensor = self.dense1(actT)
        return tf.nn.sigmoid(dense1Tensor)

if __name__ == "__main__":
    testInput = tf.ones(shape=[3,10],dtype=tf.int64)
    model = Bi_LSTM_Attention(np.ones(shape=[100,150],dtype=np.float32),labelsNumber=1)
    result = model(testInput,training = False)
    print(result)