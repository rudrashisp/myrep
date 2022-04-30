#group members
#Balaji Arcot Kulasekaran
#Bhavna Suri
#Harsh Kantilal Sharma
#Rudrashis Poddar
#Rabie Ahmed M Alsaihati
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import csv
from statistics import mean
from statistics import stdev
data = np.genfromtxt("insurance2.csv", delimiter=',')
X = data[1:,0:3]#The three columns are in X
T = range(X.shape[0])
print(T)
print(X.shape[1])
muX=[]
stdX=[]
for i in range(X.shape[1]):
	plt.plot(T,X[:,i])
	muX.append(mean(X[:,i]))
	stdX.append(stdev(X[:,i]))
plt.show()
print(muX)
print(stdX)
repmuX=numpy.matlib.repmat(muX,X.shape[0],1)
repstdX=numpy.matlib.repmat(stdX,X.shape[0],1)
standardizedX=(X-repmuX)/repstdX
X=standardizedX
for i in range(X.shape[1]):
	plt.plot(T,standardizedX[:,i])
plt.show()
y=data[1:,3]
X = np.concatenate([X, np.ones(len(X))[:, np.newaxis]], axis=1)
m=int(0.5*len(X))
Xtrain = X[0:m,:]#Training data set
ytrain = y[0:m]#Training data set
Xtest = X[m:,:]#test data set
ytest = y[m:]#test data set
target=ytest
w = np.random.randn(4)
eta=0.001
gradMSE=0
n_iterations = 20 # can be changed: during coding, use 10, after that use more to get a better model
p=50#Batch of 10
alpha=0.01
for i in range(n_iterations):
	for j in range(m):
		rand_ind=np.random.randint(m)
		xi = Xtrain[rand_ind:rand_ind+1]
		yi = ytrain[rand_ind:rand_ind+1]
		yhat=np.matmul(xi,np.transpose(w))
	#MSEloss=np.matmul((ytrain-yhat),(ytrain-yhat))/m
		gradMSE=2*(np.matmul(np.transpose(xi),(yhat-yi)))/m
		w=w-eta*gradMSE
o=np.matmul(Xtest,np.transpose(w))
print(o)
plt.plot(Xtest,ytest)
plt.show()
plt.plot(Xtest[:,0],o,color='green')#Plotting the age against the linear model with green
plt.plot(Xtest[:,1],o,color='red')#Plotting the bmi against the linear model with red
plt.plot(Xtest[:,2],o,color='blue')#Plotting the no-of-children against the linear model with blue
plt.show()
min=w[0]
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
	for j in range(m):
		rand_ind=np.random.randint(m)
		xi = Xtrain[rand_ind:rand_ind+1]
		yi = ytrain[rand_ind:rand_ind+1]
		yhat=np.matmul(xi,np.transpose(w))
	#MSEloss=np.matmul((ytrain-yhat),(ytrain-yhat))/m
		gradMSE=2*(np.matmul(np.transpose(xi),(yhat-yi)))/m
		w=w-eta*(gradMSE+2*alpha* np.sign(w))
o=np.matmul(Xtest,np.transpose(w))
plt.show()
plt.plot(Xtest[:,0],o,color='green')#Plotting the age against the linear model with green
plt.plot(Xtest[:,1],o,color='red')#Plotting the bmi against the linear model with red
plt.plot(Xtest[:,2],o,color='blue')#Plotting the no-of-children against the linear model with blue
plt.xlabel("Xtest")
plt.ylabel("output with L1 Regularisation")
plt.show()
