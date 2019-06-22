import numpy as np
from attention_net import Attention
from keras.models import model_from_json
from keras.utils import plot_model
import pandas as pd
from ast import literal_eval
from subprocess import call
import webbrowser
import time

ts=time.strftime("%d%b%y%H%M")
ts2=time.strftime("%d%b%y")

model512 = load_model('currentmodel/model512.h5' custom_objects={"Attention":Attention})
model1024 = load_model('currentmodel/model1024.h5' custom_objects={"Attention":Attention})
model2048 = load_model('currentmodel/model2048.h5' custom_objects={"Attention":Attention})

print("Loaded model from disk")
#webbrowser.open('http://169.254.184.158:8000/iot/home', new=2)

while(True):

	exit_code = call("py -2 nskydatapoor.py", shell=True)
	data512 = pd.read_csv('dataclass/dataset512.csv', names=['att','med','poo','rawValue','label'])
	data1024 = pd.read_csv('dataclass/dataset1024.csv', names=['att','med','poo','rawValue','label'])
	data2048 = pd.read_csv('dataclass/dataset2048.csv', names=['att','med','poo','rawValue','label'])

	counter = 0
	
	X512 = data512.rawValue.tolist()
	y512 = data512.label.tolist()
	
	X1024 = data1024.rawValue.tolist()
	y1024 = data1024.label.tolist()
	
	X2048 = data2048.rawValue.tolist()
	y2048 = data2048.label.tolist()
	
	
	y512 = np.array(y512)
	y512 = y.reshape(len(y512), 1)
	y512 = np_utils.to_categorical(y512)
	
	y1024 = np.array(y1024)
	y1024 = y.reshape(len(y1024), 1)
	y1024 = np_utils.to_categorical(y1024)
	
	y2048 = np.array(y2048)
	y2048 = y.reshape(len(y2048), 1)
	y2048 = np_utils.to_categorical(y2048)
	
	
	test_data = []
	for i in range(len(X512)):
		temp = np.array(literal_eval(X512[i]))
		test_data.append(temp)
	X512 = np.array(test_data)
	X512 = X512.reshape(len(X512), 256, 2)
	
	test_data = []
	for i in range(len(X1024)):
		temp = np.array(literal_eval(X1024[i]))
		test_data.append(temp)
	X1024 = np.array(test_data)
	X1024 = X1024.reshape(len(X1024), 256, 4)
	
	test_data = []
	for i in range(len(X2048)):
		temp = np.array(literal_eval(X2048[i]))
		test_data.append(temp)
	X2048 = np.array(test_data)
	X2048 = X2048.reshape(len(X2048), 256,8)

	
	model512.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	model1024.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	model2048.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	
	
	y_predicted512 = model512.predict_proba(X512)
	y_predicted1024 = model512.predict_proba(X1024)
	y_predicted2048 = model512.predict_proba(X2048)
	
	
	y_predicted512_0=(y_predicted512[0][0]+predicted512[1][0]+predicted512[2][0]+predicted512[3][0])/4
	y_predicted512_1=(y_predicted512[0][1]+predicted512[1][1]+predicted512[2][1]+predicted512[3][1])/4
	
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
	
	for f in range(len(X512)):
		with open('result/result512-'+ts2+'.csv', 'ab') as f:
			writer = csv.writer(f)
			writer.writerow(X512[f], y512[f], y_predicted512[f][0], y_predicted512[f][1])
	
	for f in range(len(X1024)):	
		with open('result/result1024-'+ts2+'.csv', 'ab') as f:
			writer = csv.writer(f)
			writer.writerow(X1024[f], y1024[f], y_predicted1024[f][0], y_predicted1024[f][1])

	with open('result/result2048-'+ts2+'.csv', 'ab') as f:
		writer = csv.writer(f)
		writer.writerow(X2048, y2048, y_predicted2048[0][0], y_predicted2048[0][1])
	
	
	with open('result/resultfinal-'+ts2+'.csv', 'ab') as f:
		writer = csv.writer(f)
		writer.writerow(X2048, y2048, y_predicted0, y_predicted1)

		
	history512=model512.fit(X512, y512, epochs=10, batch_size=10)
	history1024=model1024.fit(X1024, y1024, epochs=10, batch_size=10)
	history2048=model2048.fit(X2048, y2048, epochs=10, batch_size=10)
	
	model512.save('currentmodel/model512.h5')
	model1024.save('currentmodel/model1024.h5')
	model2048.save('currentmodel/model2048.h5')
	
	model512.save('historymodel/model512-'+ts2+'.h5')
	model1024.save('historymodel/model1024-'+ts2+'.h5')
	model2048.save('historymodel/model2048-'+ts2+'.h5')

	

