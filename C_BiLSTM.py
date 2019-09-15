import tensorflow as tf
from tensorflow import keras
import numpy as np


l1l2Lambda = 5e-7


class Bi_LSTM_Model(keras.Model):

    def __init__(self,embeddingMatrix,labelNumbers,LSTMOutUnits = 256,LSTM_Cell_Number = 3):
        super().__init__()
        self.iniEmbeddingMatrix  = tf.convert_to_tensor(embeddingMatrix,dtype=tf.float32)
        self.forLstm = keras.layers.RNN([keras.layers.LSTMCell(LSTMOutUnits,
                                                               activation=tf.nn.softsign,
                                                               kernel_regularizer=keras.regularizers.l2(l=l1l2Lambda),
                                                               recurrent_regularizer=keras.regularizers.l2(l=l1l2Lambda),
                                                               dropout=0.2, recurrent_dropout=0.2) for _ in range(LSTM_Cell_Number)])
        self.backLstm = keras.layers.RNN([keras.layers.LSTMCell(LSTMOutUnits,
                                                            activation=tf.nn.softsign,
                                                            kernel_regularizer=keras.regularizers.l2(l=l1l2Lambda),
                                                            recurrent_regularizer=keras.regularizers.l2(l=l1l2Lambda),
                                                            dropout=0.2, recurrent_dropout=0.2) for _ in range(LSTM_Cell_Number)],
                                         go_backwards=True)
        self.dense0 = keras.layers.Dense(LSTMOutUnits // 2,kernel_regularizer=keras.regularizers.l2(l=l1l2Lambda))
        self.bn = keras.layers.BatchNormalization()
        self.pRelu = keras.layers.PReLU()
        self.dense1 = keras.layers.Dense(labelNumbers,kernel_regularizer=keras.regularizers.l2(l=l1l2Lambda))

    def call(self, inputs, training=None, mask=None):
        batchTensor = tf.stop_gradient(tf.nn.embedding_lookup(params=self.iniEmbeddingMatrix, ids=inputs))
        forTensor = self.forLstm(batchTensor,initial_state=self.forLstm.get_initial_state(inputs=batchTensor),training = training)
        backTensor = self.backLstm(batchTensor,initial_state=self.backLstm.get_initial_state(inputs=batchTensor),training = training)

        addTensor = tf.multiply(0.5,tf.add(forTensor,backTensor))
        #print(addTensor.shape)
        dense0T = self.dense0(addTensor)
        bnT = self.bn(dense0T,training = training)
        reluT = self.pRelu(bnT)
        dense1T = self.dense1(reluT)
        return tf.nn.sigmoid(dense1T)


if __name__ == "__main__":
    testInput = tf.ones(shape=[3,10],dtype=tf.int32)
    model = Bi_LSTM_Model(embeddingMatrix=np.ones(shape=[10,150],dtype=np.float32),labelNumbers=1)
    result = model(testInput,training = False)
    print(result)







