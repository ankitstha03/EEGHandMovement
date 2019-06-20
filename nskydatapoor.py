from NeuroPy import NeuroPy
from time import sleep
import csv
from subprocess import call
import sys
import time
ts=time.strftime("%d%b%y")
val=0
val2=0
val3=0
cnter=0
cnter2=0
cnter3=0
qwe=0
arr=[0.000]*512
arr2=[0.000]*1024
arr3=[0.000]*2048
object1=NeuroPy("COM5")
# with open('dataset5.csv', 'wb') as f:
	# writer = csv.writer(f)

def attention_callback(attention_value):
	if(object1.attention>=10 and object1.meditation>=10 and object1.poorSignal==0 and attention_value<30000):
		global cnter
		global cnter2
		global cnter3
		cnter+=1
		cnter2+=1
		cnter3+=1
		for i in range(len(arr)-1):
			arr[i]=arr[i+1]
		arr[511]=attention_value
		for i in range(len(arr2)-1):
			arr2[i]=arr2[i+1]
		arr2[1023]=attention_value
		for i in range(len(arr3)-1):
			arr3[i]=arr3[i+1]
		arr3[2047]=attention_value
		print attention_value
		if(cnter==512):
			global qwe
			global val
			# if(qwe==0):
				# qwe=1
				# val=input("(0 for red 1 for green 2 for black and 3 for yellow)")
				# cnter=0
				# return None
			with open('dataclass/dataset512.csv', 'wb') as f:
				writer = csv.writer(f)
				writer.writerow([object1.attention, object1.meditation, object1.poorSignal, arr, val])
			with open('datahistory/dataset512-'+ts+'.csv', 'ab') as f:
				writer = csv.writer(f)
				writer.writerow([object1.attention, object1.meditation, object1.poorSignal, arr, val])
				
			cnter=0
		if(cnter2==1024):
			global qwe
			global val2
			# if(qwe==0):
				# qwe=1
				# val=input("(0 for red 1 for green 2 for black and 3 for yellow)")
				# cnter=0
				# return None
			with open('dataclass/dataset1024.csv', 'wb') as f:
				writer = csv.writer(f)
				writer.writerow([object1.attention, object1.meditation, object1.poorSignal, arr, val])
			with open('datahistory/dataset1024-'+ts+'.csv', 'ab') as f:
				writer = csv.writer(f)
				writer.writerow([object1.attention, object1.meditation, object1.poorSignal, arr, val])
				
			cnter2=0
			
			
		if(cnter3==2048):
			global qwe
			global val3
			# if(qwe==0):
				# qwe=1
				# val=input("(0 for red 1 for green 2 for black and 3 for yellow)")
				# cnter=0
				# return None
			with open('dataclass/dataset2048.csv', 'wb') as f:
				writer = csv.writer(f)
				writer.writerow([object1.attention, object1.meditation, object1.poorSignal, arr, val])
			with open('datahistory/dataset2048-'+ts+'.csv', 'ab') as f:
				writer = csv.writer(f)
				writer.writerow([object1.attention, object1.meditation, object1.poorSignal, arr, val])
				
			cnter3=0
			print("data taking completed")
			sys.exit()
	return None 
with open('datasetexp.csv', 'w') as f:
				writer = csv.writer(f)
				writer.writerow([object1.attention, object1.meditation, object1.poorSignal, arr, val])
#set call back: 
object1.setCallBack("rawValue",attention_callback) 
#call start method 
val=input("(0 for red 1 for green 2 for black and 3 for yellow)")
val2=val
val3=val
try:
	object1.start() 
except:
	sys.exit()
sys.exit()
	print(sys.exc_info()[0])
while True:
	if(object1.meditation>100): #another way of accessing data provided by headset (1st being call backs) 
		print "medidation too much"
		#object1.stop() #if meditation level reaches above 70, stop fetching data from the headset