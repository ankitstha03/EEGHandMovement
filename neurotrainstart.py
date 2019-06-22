# LSTM and CNN for sequence classification in the IMDB dataset
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import LeakyReLU
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import SimpleRNN
from keras.layers import Bidirectional
from keras.layers.convolutional import Conv1D
from keras import regularizers
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.models import model_from_json
from attention_net import Attention
import os
import matplotlib.pyplot as plt
import pandas as pd
from keras.utils import np_utils
from ast import literal_eval
import time
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import combine

combine.combinecsv()
ts=time.strftime("%d%b%y%H%M")
# fix random seed for reproducibility
np.random.seed(7)
# load the dataset but only keep the top n words, zero the rest
counta=0
data512 = pd.read_csv('datacombined/dataset512.csv', names=['att','med','poo','rawValue','label'])
X_train512=data512.rawValue.tolist()
y_train512=data512.label.tolist()

data1024 = pd.read_csv('datacombined/dataset1024.csv', names=['att','med','poo','rawValue','label'])
X_train1024=data1024.rawValue.tolist()
y_train1024=data1024.label.tolist()

data2048 = pd.read_csv('datacombined/dataset2048.csv', names=['att','med','poo','rawValue','label'])
X_train2048=data2048.rawValue.tolist()
y_train2048=data2048.label.tolist()


tsts=[]
for i in range(len(X_train512)):
	temp=literal_eval(X_train512[i])
	temp2=np.array(temp)
	tsts.append(temp2)
X_train512=np.array(tsts)
X_train512 = X_train512.reshape(len(X_train512), 512, 1)
y_train512=np.array(y_train512)
y_train512 = y_train512.reshape(len(y_train512), 1)
y_train512 = np_utils.to_categorical(y_train512)
num_classes512 = y_train512.shape[1]

tsts=[]
for i in range(len(X_train1024)):
	temp=literal_eval(X_train1024[i])
	temp2=np.array(temp)
	tsts.append(temp2)

X_train1024=np.array(tsts)
X_train1024 = X_train512.reshape(len(X_train1024), 512, 2)
y_train1024=np.array(y_train1024)
y_train1024 = y_train1024.reshape(len(y_train1024), 1)
y_train1024 = np_utils.to_categorical(y_train1024)
num_classes1024 = y_train1024.shape[1]

tsts=[]
for i in range(len(X_train2048)):
	temp=literal_eval(X_train2048[i])
	temp2=np.array(temp)
	tsts.append(temp2)

X_train2048=np.array(tsts)
X_train2048 = X_train2048.reshape(len(X_train2048), 512, 4)
y_train2048=np.array(y_train2048)
y_train2048 = y_train2048.reshape(len(y_train2048), 1)
y_train2048 = np_utils.to_categorical(y_train2048)
num_classes2048 = y_train2048.shape[1]

model512 = Sequential()
model512.add(LSTM(256, dropout=0.2, recurrent_dropout=0.2, return_sequences=True ,input_shape=(512,1)))
model512.add(Dropout(0.25))
model512.add(LeakyReLU(alpha=0.05))
model512.add(LSTM(512, dropout=0.1, activation='tanh', recurrent_dropout=0.1,return_sequences=True))
model512.add(Attention())
model512.add(Dense(num_classes512, activation='softmax'))
model512.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model512.summary())
history512=model512.fit(X_train512, y_train512, epochs=37, batch_size=50, validation_split=0.2)

model512.save('currentmodel/model512.h5')
model512.save('historymodel/model512org'+ts+'.h5')

print("Saved 512 model to disk")


model1024 = Sequential()
model1024.add(LSTM(256, dropout=0.2, recurrent_dropout=0.2, return_sequences=True ,input_shape=(512,1)))
model1024.add(Dropout(0.25))
model1024.add(LeakyReLU(alpha=0.05))
model1024.add(LSTM(512, dropout=0.1, activation='tanh', recurrent_dropout=0.1,return_sequences=True))
model1024.add(Attention())
model1024.add(Dense(num_classes1024, activation='softmax'))
model1024.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model1024.summary())
history1024=model1024.fit(X_train1024, y_train1024, epochs=37, batch_size=50, validation_split=0.2)

model1024.save('currentmodel/model1024.h5')
model1024.save('historymodelmodel1024org'+ts+'.h5')

print("Saved 512 model to disk")


model2048 = Sequential()
model2048.add(LSTM(256, dropout=0.2, recurrent_dropout=0.2, return_sequences=True ,input_shape=(512,1)))
model2048.add(Dropout(0.25))
model2048.add(LeakyReLU(alpha=0.05))
model2048.add(LSTM(512, dropout=0.1, activation='tanh', recurrent_dropout=0.1,return_sequences=True))
model2048.add(Attention())
model2048.add(Dense(num_classes2048, activation='softmax'))
model2048.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model2048.summary())
history2048=model2048.fit(X_train2048, y_train2048, epochs=37, batch_size=50, validation_split=0.2)

model2048.save('currentmodel/model2048.h5')
model2048.save('historymodel/model2048org'+ts+'.h5')

print("Saved 512 model to disk")

 
print(history512.history.keys())
#  "Accuracy"
plt.plot(history512.history['acc'], 'r--')
plt.plot(history512.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# "Loss"
plt.plot(history512.history['loss'], 'r--')
plt.plot(history512.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

print(history1024.history.keys())
#  "Accuracy"
plt.plot(history1024.history['acc'], 'r--')
plt.plot(history1024.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# "Loss"
plt.plot(history1024.history['loss'], 'r--')
plt.plot(history1024.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

print(history2048.history.keys())
#  "Accuracy"
plt.plot(history2048.history['acc'], 'r--')
plt.plot(history2048.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# "Loss"
plt.plot(history2048.history['loss'], 'r--')
plt.plot(history2048.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

 