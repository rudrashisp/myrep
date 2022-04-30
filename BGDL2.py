#group members
#Balaji Arcot Kulasekaran
#Bhavna Suri
#Harsh Kantilal Sharma
#Rudrashis Poddar
#Rabie Ahmed M Alsaihati
#Importing the libraries
import numpy as np
import pandas as pd
import numpy.matlib
import matplotlib.pyplot as plt
import csv
from statistics import mean
from statistics import stdev

#Data Pre-processing
data = np.genfromtxt("insurance2.csv", delimiter=',')
X = data[1:,0:3]#The three columns are in X
T = range(X.shape[0])
print("Range of Data:", T, "\n")
print("Number of input variables: ", X.shape[1], "\n")
a = pd.read_csv("insurance2.csv")
Attributes=[]
for col in a.columns:
    Attributes.append(col)
Attributes.pop()
Attributes.append("ones")
print("Input Attributes: ", Attributes, "\n")

#Calculating Mean and Standard Deviation while plotting intial data
muX=[]
stdX=[]
for i in range(X.shape[1]):
    plt.plot(T,X[:,i])
    muX.append(mean(X[:,i]))
    stdX.append(stdev(X[:,i]))
plt.title("Data Before Normalization")
plt.xlabel("Range of values")
plt.ylabel("Values of input variablees")
plt.show()

print("Mean: ", muX)
print("Standard Deviation: ", stdX, "\n")

#Data Standarization
repmuX=numpy.matlib.repmat(muX,X.shape[0],1)
repstdX=numpy.matlib.repmat(stdX,X.shape[0],1)
standardizedX=(X-repmuX)/repstdX
X=standardizedX

#Plot of StandarizedX
for i in range(X.shape[1]):
    plt.plot(T,standardizedX[:,i])
plt.title("Standarized X")
plt.xlabel("Range of values")
plt.ylabel("Values of input variablees")
plt.show()

#Training and Test Sets
y=data[1:,3]
X = np.concatenate([X, np.ones(len(X))[:, np.newaxis]], axis=1)
m=int(0.5*len(X))
Xtrain = X[0:m,:]#Training data set
ytrain = y[0:m]#Training data set
Xtest = X[m:,:]#test data set
ytest = y[m:]#test data set
target=ytest

#Model Training
w = np.random.randn(4)
print(w)
eta=0.001
gradMSE=0
alpha=0.01
n_iterations = 10
min=99999;

yhat=np.matmul(Xtrain,np.transpose(w))
#MSEloss=np.matmul((ytrain-yhat),(ytrain-yhat))/m
gradMSE=(2/m)*(np.matmul(np.transpose(Xtrain),(yhat-ytrain)))
w=w-eta*(gradMSE)

print(w)
for i in range(4):
	if(w[i]< min):
		min=w[i]
		pos=i
print(pos)
w=np.delete(w,pos)
print(w)

Xtrain=np.delete(Xtrain,pos,axis=1)
Xtest=np.delete(Xtest,pos,axis=1)
for i in range(n_iterations):
	yhat=np.matmul(Xtrain,np.transpose(w))
	#MSEloss=np.matmul((ytrain-yhat),(ytrain-yhat))/m
	gradMSE=(2/m)*(np.matmul(np.transpose(Xtrain),(yhat-ytrain)))
	w=w-eta*(gradMSE+2*alpha*(w))
	o=np.matmul(Xtest,np.transpose(w))

plt.plot(Xtest,ytest)
plt.show()

plt.plot(Xtest,o)
plt.xlabel("Xtest")
plt.ylabel("output with L2 Regularisation")
plt.show()



