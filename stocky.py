import numpy as np
import pandas as pd
import quandl
import math
from sklearn import preprocessing,cross_validation,svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import datetime

df=quandl.get("WIKI/AAPL", authtoken="nwoCxbmg7z8_vJwy4d86")
print (df.head())
df=df[['Adj. Open','Adj. Close','Adj. High','Adj. Low','Adj. Volume']]
print(df.tail())
#percent volatility
df['HL_PCT']=(df['Adj. High']-df['Adj. Close'])/df['Adj. Close']*100.0
#daily percent change
df['PCT_change']=(df['Adj. Close']-df['Adj. Open'])/df['Adj. Open']*100.0

df=df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]

#lets forecast closing amount

forecast='Adj. Close'
df.fillna(-99999,inplace=True)
forecast_it=int(math.ceil(0.01*len(df)))
df['label']=df[forecast].shift(-forecast_it)
x=np.array(df.drop(['label'],1))
x=preprocessing.scale(x)
forcst_it_lte=forecast_it

x_lately=x[-forcst_it_lte:]
x=x[:-forecast_it]
df.dropna(inplace=True)
y=np.array(df['label'])
plt.plot()
x_train,x_test,y_train,y_test=cross_validation.train_test_split(x,y,test_size=0.2)
clf=LinearRegression(n_jobs=-1)
clf.fit(x_train,y_train)
accuracy=clf.score(x_test,y_test)
print(accuracy)
forecast_set=clf.predict(x_lately)
print(forecast_set)
df['Forecast']=np.nan
lst_date=df.iloc[-1].name
print(lst_date)
last_unix=lst_date.timestamp()
one_day=86400
next_unix=last_unix+one_day
for k in forecast_set:
	nxt_date=datetime.datetime.fromtimestamp(next_unix)
	next_unix+=one_day
	df.loc[nxt_date]=[np.nan for _ in range(len(df.columns)-1)]+[k]
	
print(df.tail(10))
df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('date')
plt.ylabel('Price')
plt.show()

