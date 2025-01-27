import numpy as np
from attention_net import Attention
from keras.models import model_from_json
from keras.utils import plot_model
import pandas as pd
from ast import literal_eval
from subprocess import call
import webbrowser
import time
import csv
import os
from keras.models import load_model
from keras.utils import np_utils

ts=time.strftime("%d%b%y%H%M")
ts2=time.strftime("%d%b%y")
path=os.getcwd()

model512 = load_model('currentmodel/model512.h5', custom_objects={"Attention":Attention})
model1024 = load_model('currentmodel/model1024.h5', custom_objects={"Attention":Attention})
model2048 = load_model('currentmodel/model2048.h5', custom_objects={"Attention":Attention})

print("Loaded model from disk")
#webbrowser.open('http://169.254.184.158:8000/iot/home', new=2)
next='y'
while(next=='y'):

	# exit_code = call("py -2 nskydatapoor.py", shell=True)
	data512 = pd.read_csv('dataclass/dataset512.csv', names=['att','med','poo','rawValue','label'])
	data1024 = pd.read_csv('dataclass/dataset1024.csv', names=['att','med','poo','rawValue','label'])
	data2048 = pd.read_csv('dataclass/dataset2048.csv', names=['att','med','poo','rawValue','label'])

	counter = 0
	
	X512i = data512.rawValue.tolist()
	y512i = data512.label.tolist()
	
	X1024i = data1024.rawValue.tolist()
	y1024i = data1024.label.tolist()
	
	X2048i = data2048.rawValue.tolist()
	y2048i = data2048.label.tolist()
	
	
	y512 = np.array(y512i)
	test_data = []
	for i in range(len(y512)):
		if (y512[i]==0):
			test_data2 = []
			test_data2.append(1.)
			test_data2.append(0.)
			test_data.append(test_data2)
		else:
			test_data2 = []
			test_data2.append(0.)
			test_data2.append(1.)
			test_data.append(test_data2)
	y512 = np.array(test_data)
	
	y1024 = np.array(y1024i)
	test_data = []
	for i in range(len(y1024)):
		if (y1024[i]==0):
			test_data2 = []
			test_data2.append(1.)
			test_data2.append(0.)
			test_data.append(test_data2)
		else:
			test_data2 = []
			test_data2.append(0.)
			test_data2.append(1.)
			test_data.append(test_data2)
	y1024 = np.array(test_data)

	y2048 = np.array(y2048i)
	test_data = []
	for i in range(len(y2048)):
		if (y2048[i]==0):
			test_data2 = []
			test_data2.append(1.)
			test_data2.append(0.)
			test_data.append(test_data2)
		else:
			test_data2 = []
			test_data2.append(0.)
			test_data2.append(1.)
			test_data.append(test_data2)
	y2048 = np.array(test_data)
	
	
	test_data = []
	for i in range(len(X512i)):
		temp = np.array(literal_eval(X512i[i]))
		test_data.append(temp)
	X512 = np.array(test_data)
	X512 = X512.reshape(len(X512), 512, 1)
	
	test_data = []
	for i in range(len(X1024i)):
		temp = np.array(literal_eval(X1024i[i]))
		test_data.append(temp)
	X1024 = np.array(test_data)
	X1024 = X1024.reshape(len(X1024), 512, 2)
	
	test_data = []
	for i in range(len(X2048i)):
		temp = np.array(literal_eval(X2048i[i]))
		test_data.append(temp)
	X2048 = np.array(test_data)
	X2048 = X2048.reshape(len(X2048), 512,4)

	
	# model512.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# model1024.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# model2048.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	
	
	y_predicted512 = model512.predict_proba(X512)
	y_predicted1024 = model1024.predict_proba(X1024)
	y_predicted2048 = model2048.predict_proba(X2048)
	
	
	y_predicted512_0=(y_predicted512[0][0]+y_predicted512[1][0]+y_predicted512[2][0]+y_predicted512[3][0])/4
	y_predicted512_1=(y_predicted512[0][1]+y_predicted512[1][1]+y_predicted512[2][1]+y_predicted512[3][1])/4
	
	y_predicted1024_0=(y_predicted1024[0][0]+y_predicted1024[1][0])/2
	y_predicted1024_1=(y_predicted1024[0][1]+y_predicted1024[1][1])/2
	
	y_predicted2048_0=y_predicted2048[0][0]
	y_predicted2048_1=y_predicted2048[0][1]
	
	y_predicted0=(y_predicted512_0+y_predicted1024_0+y_predicted2048_0)/3
	y_predicted1=(y_predicted512_1+y_predicted1024_1+y_predicted2048_1)/3
	
	
	print("512 model predicts: left=%s, right=%s",y_predicted512_0 , y_predicted512_1)
	print("1024 model predicts: left=%s, right=%s",y_predicted1024_0 , y_predicted1024_1)
	print("2048 model predicts: left=%s, right=%s",y_predicted2048_0 , y_predicted2048_1)
	
	
	print("Average model predicts: left=%s, right=%s",y_predicted512_0 , y_predicted512_1)
	
	for g in range(len(X512)):
		with open('result/result512-'+ts2+'.csv', 'a') as f:
			writer = csv.writer(f)
			writer.writerow([X512i[g], y512i[g], float(y_predicted512[g][0]), float(y_predicted512[g][1])])
	
	for g in range(len(X1024)):	
		with open('result/result1024-'+ts2+'.csv', 'a') as f:
			writer = csv.writer(f)
			writer.writerow([X1024i[g], y1024i[g], float(y_predicted1024[g][0]), float(y_predicted1024[g][1])])

	with open('result/result2048-'+ts2+'.csv', 'a') as f:
		writer = csv.writer(f)
		writer.writerow([X2048i[0], y2048i[0], float(y_predicted2048[0][0]), float(y_predicted2048[0][1])])
	
	
	with open('result/resultfinal-'+ts2+'.csv', 'a') as f:
		writer = csv.writer(f)
		writer.writerow([X2048i[0], y2048i[0], y_predicted0, y_predicted1])

		
	history512=model512.fit(X512, y512, epochs=1, batch_size=10)
	history1024=model1024.fit(X1024, y1024, epochs=1, batch_size=10)
	history2048=model2048.fit(X2048, y2048, epochs=1, batch_size=10)
	
	model512.save('currentmodel/model512.h5')
	model1024.save('currentmodel/model1024.h5')
	model2048.save('currentmodel/model2048.h5')
	
	model512.save('historymodel/model512-'+ts+'.h5')
	model1024.save('historymodel/model1024-'+ts+'.h5')
	model2048.save('historymodel/model2048-'+ts+'.h5')

	next=input("do ypu want to continue to another data? (y/n):")

	

