import numpy as np;
import pandas as pd;
import random
import math;
import array
dataset=pd.read_csv('data.csv')
X=dataset.iloc[:,:-1].values;
y=dataset.iloc[:,8].values;
epoch=1000;
n = int(len(X))
whales=n/2
features=len(X[0])
featureList=[[0 for pra in range(n)] for pra1 in range(features)]
for feature in range(features):
	featureList[feature]=dataset.iloc[:,feature]
theta=[[0.00 for x in range(features)] for z in range(whales)]
#constant=[1.00 for x in range(whale)]
fitness=[0 for pra in range(whales)]
bestWhale=0
bestMSE=float('inf');
for feature in range(features):
	list1,list2=zip(*sorted(zip(featureList[feature], y)))
	for whale in range(whales):
		if((list1[n-whale-1]-list1[whale])!=0):
			theta[whale][feature]=(list2[n-whale-1]-list2[whale])/(list1[n-whale-1]-list1[whale])
		else:
			theta[whale][feature]=0.00
		#print(theta[whale][feature])


for whale in range(whales):
	#for feature in range(features):
		#theta[whale][feature]=1
		#diffXForSlope=(X[n/2+whale+1][feature]-X[n/2-whale-1][feature])
		#if(diffXForSlope==0.0):
		#	diffXForSlope=0.5
		#theta[whale][feature]=(y[n/2+whale+1]-y[n/2-whale-1])/diffXForSlope
		#thetaAbbrev=theta[whale][feature]
		#if(math.isnan(thetaAbbrev)or math.isinf(thetaAbbrev)):
		#	theta[whale][feature]=1.00
		#theta[whale][feature]=whale+feature
	mse=0.0;
	for row in range(n):
		sum=0
		for feature in range(features):
			sum=sum+theta[whale][feature]*X[row][feature]
		#sum=constant[whale]
		mse=mse+abs(y[row]-sum)
	if(mse<bestMSE):
		bestMSE=mse
		bestWhale=whale
#print(bestWhale)
#print(bestMSE);
a=2.00
b=0.00000000000000000000000001
alpha=0.000000001
for iteration in range(epoch):
	forNextIter=[[0 for pra in range(features)] for pra1 in range(whales)]
	for whale in range(whales):
		if(1==1):
			r=[0.00 for pra in range(features)]
			C=[0.00 for pra in range(features)]
			A=[0.00 for pra in range(features)]
			p=float(random.randint(1,100))/100
			for eachfeature in range(features):
				r[eachfeature]=(float(random.randint(1,100)))/100
				C[eachfeature]=2*r[eachfeature]
				A[eachfeature]=2*a*r[eachfeature]-a
			if(1==0):
				#find magnitude of A vector
				sum_123=0.00
				for feature in range(features):
					#print(A[feature])
					sum_123=sum_123+math.pow(A[feature],2)
				#print(a)\
				#print(sum_123)
				#print(whale)
				#print(iteration)
				magnitudeA=math.sqrt(sum_123)
				if(1==1):
					vecD=0
					for feature in range(features):
						vecD=C[feature]*theta[bestWhale][feature]-theta[whale][feature]
						forNextIter[whale][feature]=theta[bestWhale][feature]-A[feature]*vecD
						#print(forNextIter[whale][feature]-theta[whale][feature]);
						
				else:
					xRand=random.randint(0,whales-1)
					vecD=0
					for feature in range(features):
						vecD=C[feature]*theta[xRand][feature]-theta[whale][feature]
						forNextIter[whale][feature]=theta[xRand][feature]-A[feature]*vecD
						#print(forNextIter[whale][feature]-theta[whale][feature]);
			if(1==1):
				l=float(random.randint(-100,100))/100;
				#l=0.0000000000000000001
				for feature in range(features):
					Ddash=theta[bestWhale][feature]-theta[whale][feature]
					forNextIter[whale][feature]=Ddash*math.exp(2*b*l)*math.cos(2*3.14*l)+theta[whale][feature]
					#print(math.exp(2*b*l)*math.cos(2*3.14*l));
					#print(theta[whale][feature])
					#print(theta[bestWhale][feature])
					#print(forNextIter[whale][feature])
					#print(abs(theta[whale][feature]-forNextIter[whale][feature]))

	if(1==1):
		for whale in range(whales):
			for feature in range(features):
				theta[whale][feature]=forNextIter[whale][feature]


    #apply gradient descent
	for whale in range(whales):
		#print(whale)
		forNextIter=[0 for pra in range(features)]
		j=0.00
		for row in range(n):
			for feature1 in range(features):
				j=j+theta[whale][feature1]*X[row][feature1]
				#print(j)
			j=j-y[row];
		for feature in range(features):
			sum=0.00;
			for row in range(n):
				sum=sum+j*X[row][feature]
			sum=(sum/n)*alpha
			#theta[whale][feature]=theta[whale][feature]-sum
			forNextIter[feature]=theta[whale][feature]-sum
		for feature in range(features):
			theta[whale][feature]=forNextIter[feature]
		#print("hi")
	#check which is bestWhale
	bestWhale=0
	bestMSE=float('inf')
	for whale in range(whales):
		mse=0.0;
		for row in range(n):
			sum=0
			for feature in range(features):
				sum=sum+theta[whale][feature]*X[row][feature]
			#sum=constant[whale]
			mse=mse+abs(y[row]-sum)
		if(mse<bestMSE):
			bestMSE=mse
			bestWhale=whale
	a=a-(2.00/epoch)*(iteration+1)
	#print(bestWhale)
	#print(bestMSE)
	#a=a-0.0000001
	#print(iteration)

#finalResults:
finalResult=[0 for pra in range(n)]
for row in range(n):
	for feature in range(features):
		finalResult[row]=finalResult[row]+theta[bestWhale][feature]*X[row][feature]
	print(finalResult[row])
mse=0.00
for row in range(n):
	j=0.00
	for feature in range(features):
		j=j+theta[bestWhale][feature]*X[row][feature]
	mse=mse+math.pow(j-y[row],2)
mse=math.sqrt(mse)
print(mse)

from sklearn.linear_model import LinearRegression        
lr=LinearRegression()
lr.fit(X,y)
pred=lr.predict(X)
        
        
