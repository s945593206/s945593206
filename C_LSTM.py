import tensorflow as tf
from tensorflow import keras
import numpy as np


l1l2Lambda = 5e-7


class LSTM_Model(keras.Model):

    def __init__(self,embeddingMatrix,labelNumbers,LSTMOutUnits = 256,LSTM_Cell_Number = 3):
        super().__init__()
        self.iniEmbeddingMatrix  = tf.convert_to_tensor(embeddingMatrix,dtype=tf.float32)
        self.lstm = keras.layers.RNN([keras.layers.LSTMCell(LSTMOutUnits,
                                                            activation=tf.nn.softsign,
                                                            kernel_regularizer=keras.regularizers.l2(l=l1l2Lambda),
                                                            recurrent_regularizer=keras.regularizers.l2(l=l1l2Lambda),
                                                            dropout=0.2, recurrent_dropout=0.2) for _ in range(LSTM_Cell_Number)])
        self.dense0 = keras.layers.Dense(LSTMOutUnits // 2,kernel_regularizer=keras.regularizers.l2(l=l1l2Lambda))
        self.bn = keras.layers.BatchNormalization()
        self.prelu = keras.layers.PReLU()
        self.dense1 = keras.layers.Dense(labelNumbers,kernel_regularizer=keras.regularizers.l2(l=l1l2Lambda))


    def call(self, inputs, training=None, mask=None):
        batchTensor = tf.stop_gradient(tf.nn.embedding_lookup(params=self.iniEmbeddingMatrix,ids=inputs))
        #print("embedding batch tensor ",batchTensor.shape)
        lstmOut = self.lstm(batchTensor,initial_state=self.lstm.get_initial_state(inputs=batchTensor),training = training)
        dense0Tensor = self.dense0(lstmOut)
        bnTensor = self.bn(dense0Tensor,training = training)
        reluT = self.prelu(bnTensor)
        outTensor = self.dense1(reluT)
        return tf.nn.sigmoid(outTensor)


if __name__ == "__main__":
    testInput = tf.ones(shape=[3,10],dtype=tf.int32)
    model = LSTM_Model(embeddingMatrix=np.ones(shape=[10,150],dtype=np.float32),
                       LSTMOutUnits=25,LSTM_Cell_Number=3,labelNumbers=1)
    result = model(testInput,training = False)
    print(result)

