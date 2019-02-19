import numpy as np
import pandas as pd
import pyaudio
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

CHUNK = 2**11
RATE = 44100

train=pd.read_csv('KeyWordData.csv')
train.head()
X_train, X_test, y_train, y_test = train_test_split(train.drop('21',axis=1), train['21'], test_size=0.30, random_state=101)
#print(X_train)
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)

#Audio input prediction

def printare_fisier(lista):
    print(lista)
    if len(lista)<=20:
        w2=open("SpokenKey.csv","w")
        for index in range(1,22):
            if index!=21:
                w2.write(str(index)+",")
            else:
                w2.write(str(index)+"\n")
        
        for index in range(len(lista)+1):
            if index!=len(lista):
                w2.write(str(lista[index])+",")
            else:
                w2.write(str(0)+"\n")
                
        for index in range(1,22):
            if index!=21:
                w2.write("0,")
            else:
                w2.write("0\n")
        w2.close()
        
def procesare_comanda(lista):
    lun=0
    i=0
    while i<len(lista):
        if lista[i]==0:
            lun+=1
        else:
            lun=0
        if lun>2:
            lista.pop(i-3)
            lun-=1
        i+=1
    while lista[0]==0:
        lista.pop(0)
        if len(lista)==1:
            break
    while lista[len(lista)-1]==0 and len(lista)!=1:
        lista.pop(len(lista)-1)
    while len(lista)<20:
        lista.append(0)
        
def preluare_input():
    print("Introduceti comanda vocala: ")
    lista=[]
    for i in range(int(10*8100/1024)):
        data = np.frombuffer(stream.read(CHUNK),dtype=np.int16)
        peak=np.average(np.abs(data))*2
        bars="#"*int(300*peak/2**15)
        if len(bars)>=3:
            lista.append(len(bars))
        else:
            lista.append(0)
        #print(bars)
    print("Analizare date")
    return lista

p=pyaudio.PyAudio()
stream=p.open(format=pyaudio.paInt16,channels=1,rate=RATE,input=True,
              frames_per_buffer=CHUNK)

ok=True
while ok:
    lista=preluare_input()
    procesare_comanda(lista)
    printare_fisier(lista)
    if len(lista)<=20:
        test=pd.read_csv('SpokenKey.csv')
        test.head()
        X_train1, X_test1, y_train1, y_test1 = train_test_split(test.drop('21',axis=1), test['21'], test_size=0.30, random_state=101)
        prediction = logmodel.predict(X_test1)
        if prediction==[1]:
            ok=False
            print("Comanda Confirmata")
        else:
            print("Comanda nu este recunsocuta")
        #y_test2=[1]
        #print(classification_report(y_test2,prediction))

stream.stop_stream()
stream.close()
p.terminate()