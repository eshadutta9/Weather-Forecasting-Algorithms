import statsmodels.api
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_squared_error
from math import sqrt
import statsmodels as sm
import csv
from sklearn import preprocessing
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse, aic
from statsmodels.tsa.statespace.varmax import VARMAX
from sklearn.preprocessing import StandardScaler
ms=[]
rms=[]
acc=[]

name1=['maxT','minT','humidity','windspeed','rainfall','sunshine','EVP']
df = pd.read_csv('Anand.csv', usecols=[1,2,3,5,6,7,8], names=name1)

# df=preprocessing.normalize(df)
# df=pd.DataFrame(df, columns=name1)

test_copy=df[-365:]

scaler=StandardScaler()
scaler.fit(df)

train=df[8036:-365] 
test=df[-365:]

model=VAR(train)

# x = model.select_order(maxlags=9)
# print(x.summary())
# order 9 has been selected

model_fit=model.fit(9)
print(model_fit.summary())

lag_order = model_fit.k_ar
#print(lag_order)
forecast_input = train.values[-lag_order:]
#print(forecast_input)
fc = model_fit.forecast(y=forecast_input, steps=len(test))
df_forecast = pd.DataFrame(fc, columns=name1)



#-----------------------------CALCULATION OF ACCURACY AND ACCURACY---------------------------

final=scaler.inverse_transform(fc)
test=scaler.inverse_transform(test)
# print(final)

final=pd.DataFrame(final, columns=name1)
test=pd.DataFrame(test, columns=name1)
#print(final)


#-------------PLOTTING MAX TEMP--------------
plt.figure(figsize=(10,7))
plt.plot( train['maxT'], label='Train')
plt.plot(test_copy['maxT'], label='Test')
plt.plot(final['maxT'], label='VAR')
plt.legend(loc='best')

plt.title('maxT')
plt.xlabel('index')
plt.ylabel('maxT')

#-------------PLOTTING MIN TEMP----------------

plt.figure(figsize=(10,7))
plt.plot( train['minT'], label='Train')
plt.plot(test_copy['minT'], label='Test')
plt.plot(final['minT'], label='VAR')
plt.legend(loc='best')

plt.title('minT')
plt.xlabel('index')
plt.ylabel('minT')


 #-------------PLOTTING HUMIDITY--------------
plt.figure(figsize=(10,7))
plt.plot( train['humidity'], label='Train')
plt.plot(test_copy['humidity'], label='Test')
plt.plot(final['humidity'], label='VAR')
plt.legend(loc='best')

plt.title('humidity')
plt.xlabel('index')
plt.ylabel('humidity')

#-------------PLOTTING WINDSPEED--------------
plt.figure(figsize=(10,7))
plt.plot( train['windspeed'], label='Train')
plt.plot(test_copy['windspeed'], label='Test')
plt.plot(final['windspeed'], label='VAR')
plt.legend(loc='best')

plt.title('windspeed')
plt.xlabel('index')
plt.ylabel('windspeed')


#---------------PLOTTING RAINFALL---------------


plt.figure(figsize=(10,7))
plt.plot( train['rainfall'], label='Train')
plt.plot(test_copy['rainfall'], label='Test')
plt.plot(final['rainfall'], label='VAR')
plt.legend(loc='best')

plt.title('rainfall')
plt.xlabel('index')
plt.ylabel('rainfall')
plt.show()

#-------------PLOTTING SUNSHINE-------------
plt.figure(figsize=(10,7))
plt.plot( train['sunshine'], label='Train')
plt.plot(test_copy['sunshine'], label='Test')
plt.plot(final['sunshine'], label='VAR')
plt.legend(loc='best')

plt.title('sunshine')
plt.xlabel('index')
plt.ylabel('sunshine')



ms_max=mean_squared_error(test_copy['maxT'], df_forecast['maxT'])
rms_max = sqrt(mean_squared_error(test_copy['maxT'], df_forecast['maxT']))
ms.append(round(ms_max,2))
rms.append(round(rms_max,2))

ms_min=mean_squared_error(test_copy['minT'], df_forecast['minT'])
rms_min = sqrt(mean_squared_error(test_copy['minT'], df_forecast['minT']))
ms.append(round(ms_min,2))
rms.append(round(rms_min,2))

ms_hum=mean_squared_error(test_copy['humidity'], df_forecast['humidity'])
rms_hum = sqrt(mean_squared_error(test_copy['humidity'], df_forecast['humidity']))
ms.append(round(ms_hum,2))
rms.append(round(rms_hum,2))

ms_ws=mean_squared_error(test_copy['windspeed'], df_forecast['windspeed'])
rms_ws = sqrt(mean_squared_error(test_copy['windspeed'], df_forecast['windspeed']))
ms.append(round(ms_ws,2))
rms.append(round(rms_ws,2))

ms_rain=mean_squared_error(test_copy['rainfall'], df_forecast['rainfall'])
rms_rain = sqrt(mean_squared_error(test_copy['rainfall'], df_forecast['rainfall']))
ms.append(round(ms_rain,2))
rms.append(round(rms_rain,2))

ms_sun=mean_squared_error(test_copy['sunshine'], df_forecast['sunshine'])
rms_sun = sqrt(mean_squared_error(test_copy['sunshine'], df_forecast['sunshine']))
ms.append(round(ms_sun,2))
rms.append(round(rms_sun,2))

ms_evp=mean_squared_error(test_copy['EVP'], df_forecast['EVP'])
rms_evp = sqrt(mean_squared_error(test_copy['EVP'], df_forecast['EVP']))
ms.append(round(ms_evp,2))
rms.append(round(rms_evp,2))


# acc_hum=(100*final['humidity'])/test['humidity']
# acc.append(round((acc_hum.sum()/len(acc_hum)),2))

# acc_maxt=(100*final['maxT'])/test['maxT']
# acc.append(round((acc_maxt.sum()/len(acc_maxt)),2))

# acc_mint=(100*final['minT'])/test['minT']
# acc.append(round((acc_mint.sum()/len(acc_mint)),2))

# acc_ws=(100*final['windspeed'])/test['windspeed']
# acc.append(round((acc_ws.sum()/len(acc_ws)),2))

# acc_sun=(100*final['sunshine'])/test['sunshine']
# acc.append(round((acc_sun.sum()/len(acc_sun)),2))

# acc_rf=(100*final['rainfall'])/test['rainfall']
# acc.append(round((acc_rf.sum()/len(acc_rf)),2))

final_list=df_forecast
acc.append(round(100-abs((sum(final_list['maxT'])-sum(test_copy['maxT']))*100/sum(test_copy['maxT'])),2))
acc.append(round(100-abs((sum(final_list['minT'])-sum(test_copy['minT']))*100/sum(test_copy['minT'])),2))
acc.append(round(100-abs((sum(final_list['humidity'])-sum(test_copy['humidity']))*100/sum(test_copy['humidity'])),2))
acc.append(round(100-abs((sum(final_list['windspeed'])-sum(test_copy['windspeed']))*100/sum(test_copy['windspeed'])),2))
acc.append(round(100-abs((sum(final_list['rainfall'])-sum(test_copy['rainfall']))*100/sum(test_copy['rainfall'])),2))
acc.append(round(100-abs((sum(final_list['sunshine'])-sum(test_copy['sunshine']))*100/sum(test_copy['sunshine'])),2))
acc.append(round(100-abs((sum(final_list['EVP'])-sum(test_copy['EVP']))*100/sum(test_copy['EVP'])),2))

#---------------------------------OUTPUT---------------------------

dash='-'*100
print(dash)
print('          {:<10s}{:>4s}{:>12s}{:>12s}{:>12s}{:>12s}'.format(name1[0],name1[1],name1[2],name1[3],name1[4],name1[5],name1[6]))
print(dash) 
print('MSE:      {:<10.2f}{:>4.2f}{:>12.2f}{:>12.2f}{:>12.2f}{:>12.2f}{:>12.2f}'.format(ms[0],ms[1],ms[2],ms[3],ms[4],ms[5],ms[6]))
print('RMSE:     {:<10.2f}{:>4.2f}{:>12.2f}{:>12.2f}{:>12.2f}{:>12.2f}{:>12.2f}'.format(rms[0],rms[1],rms[2],rms[3],rms[4],rms[5],rms[6]))
print('Accuracy: {:<10.2f}{:>4.2f}{:>12.2f}{:>12.2f}{:>12.2f}   '.format(acc[0],acc[1],acc[2],acc[3],acc[4]),acc[5],acc[6])
print(rms)
print(acc)

final.to_excel("output_var.xlsx")