import numpy as np;
import pandas as pd;
import random
import math;
import array
dataset=pd.read_csv('data.csv')
X=dataset.iloc[:,:-1].values;
y=dataset.iloc[:,8].values;
n=len(X);
features=len(X[0])
from sklearn.linear_model import LinearRegression        
lr=LinearRegression()
lr.fit(X,y)
pred=lr.predict(X)
n=len(X)
for row in range(n):
	print(pred[row])

mse=0.00
for row in range(n):
	mse=mse+math.pow(pred[row]-y[row],2)
mse=math.sqrt(mse)
print(mse)