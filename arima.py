import statsmodels.api
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
import statsmodels as sm
import csv
from sklearn import preprocessing
from statsmodels.tsa.stattools import acf, pacf
from sklearn.preprocessing import StandardScaler

ms=[]
rms=[]
acc=[]
final_list=pd.DataFrame()

name1=['maxT','minT','humidity','windspeed','rainfall','sunshine','EVP']
df = pd.read_csv('Anand.csv', usecols=[1,2,3,5,6,7,8], names=name1)

# df=preprocessing.normalize(df)
# df=pd.DataFrame(df, columns=name1)

copy_test=df[-365:]

scaler=StandardScaler()
scaler.fit(df)

train=df[8036:-365] 
test=df[-365:]

y_hat_avg = test.copy()

#---------------------PREDICTING HUMIDITY---------------------

print("Predicting humidity")

fit1 = sm.tsa.arima_model.ARIMA(np.asarray(train['humidity']), order=(2, 0, 1)).fit()
y_hat_avg['ARIMA'] = fit1.predict(start=1, end=365)

#Plotting

plt.figure(figsize=(10,7))
plt.plot( train['humidity'], label='Train')
plt.plot(test['humidity'], label='Test')
plt.plot(y_hat_avg['ARIMA'], label='ARIMA')
plt.legend(loc='best')

plt.title('Humidity')
plt.xlabel('index')
plt.ylabel('Humidity')
#plt.show()

#Calculating Error

ms_rh1=mean_squared_error(test['humidity'], y_hat_avg.ARIMA)
rms_rh1 = sqrt(mean_squared_error(test['humidity'], y_hat_avg.ARIMA))
#acc_rh1=r2_score(test['humidity'],y_hat_avg.ARIMA)
ms.append(round(ms_rh1,2))
rms.append(round(rms_rh1,2))

final_list['humidity']=y_hat_avg.ARIMA


input("Press enter to continue...\n")


#-----------------PREDICTING MAXTEMP--------------------

print("Predicting Maximum Temperature")

fit1 = sm.tsa.arima_model.ARIMA(np.asarray(train['maxT']), order=(2, 0, 1)).fit()
y_hat_avg['ARIMA'] = fit1.predict(start=1, end=365)

#Plotting

plt.figure(figsize=(10,7))
plt.plot( train['maxT'], label='Train')
plt.plot(test['maxT'], label='Test')
plt.plot(y_hat_avg['ARIMA'], label='ARIMA')
plt.legend(loc='best')

plt.title('Maximum Temperature')
plt.xlabel('index')
plt.ylabel('Maximum Temperature')
#plt.show()

#Calculating Error

ms_rh1=mean_squared_error(test['maxT'], y_hat_avg.ARIMA)
rms_rh1 = sqrt(mean_squared_error(test['maxT'], y_hat_avg.ARIMA))
ms.append(round(ms_rh1,2))
rms.append(round(rms_rh1,2))

final_list['maxT']=y_hat_avg.ARIMA

input("Press enter to continue...\n")

#-----------------PREDICTING MINTEMP----------------------

print("Predicting Minimum Temperature")

fit1 = sm.tsa.arima_model.ARIMA(np.asarray(train['minT']), order=(2, 0, 1)).fit()
y_hat_avg['ARIMA'] = fit1.predict(start=1, end=365)

#Plotting

plt.figure(figsize=(10,7))
plt.plot( train['minT'], label='Train')
plt.plot(test['minT'], label='Test')
plt.plot(y_hat_avg['ARIMA'], label='ARIMA')
plt.legend(loc='best')

plt.title('Minimum Temperature')
plt.xlabel('index')
plt.ylabel('Minimum Temperature')
#plt.show()

#Calculating Error

ms_rh1=mean_squared_error(test['minT'], y_hat_avg.ARIMA)
rms_rh1 = sqrt(mean_squared_error(test['minT'], y_hat_avg.ARIMA))
ms.append(round(ms_rh1,2))
rms.append(round(rms_rh1,2))

final_list['minT']=y_hat_avg.ARIMA


input("Press enter to continue...\n")


#-----------------PREDICTING WINDSPEED--------------------

print("Predicting Windspeed")

fit1 = sm.tsa.arima_model.ARIMA(np.asarray(train['windspeed']), order=(5, 0, 1)).fit()
y_hat_avg['ARIMA'] = fit1.predict(start=1, end=365)

#Plotting

plt.figure(figsize=(10,7))
plt.plot( train['windspeed'], label='Train')
plt.plot(test['windspeed'], label='Test')
plt.plot(y_hat_avg['ARIMA'], label='ARIMA')
plt.legend(loc='best')

plt.title('Windspeed')
plt.xlabel('index')
plt.ylabel('Windspeed')
#plt.show()

#Calculating Error

ms_rh1=mean_squared_error(test['windspeed'], y_hat_avg.ARIMA)
rms_rh1 = sqrt(mean_squared_error(test['windspeed'], y_hat_avg.ARIMA))
ms.append(round(ms_rh1,2))
rms.append(round(rms_rh1,2))

final_list['windspeed']=y_hat_avg.ARIMA


input("Press enter to continue...\n")

#----------------------PREDICTING SUNSHINE-------------------

print("Predicting Sunshine")

fit1 = sm.tsa.arima_model.ARIMA(np.asarray(train['sunshine']), order=(3, 0, 1)).fit()
y_hat_avg['ARIMA'] = fit1.predict(start=1, end=365)

#Plotting

plt.figure(figsize=(10,7))
plt.plot( train['sunshine'], label='Train')
plt.plot(test['sunshine'], label='Test')
plt.plot(y_hat_avg['ARIMA'], label='ARIMA')
plt.legend(loc='best')

plt.title('Sunshine')
plt.xlabel('index')
plt.ylabel('Sunshine')
#plt.show()

#Calculating Error

ms_rh1=mean_squared_error(test['sunshine'], y_hat_avg.ARIMA)
rms_rh1 = sqrt(mean_squared_error(test['sunshine'], y_hat_avg.ARIMA))
ms.append(round(ms_rh1,2))
rms.append(round(rms_rh1,2))

final_list['sunshine']=y_hat_avg.ARIMA


input("Press enter to continue...\n")

#--------------------PREDICTING RAINFALL-------------------

print("Predicting Rainfall")

fit1 = sm.tsa.arima_model.ARIMA(np.asarray(train['rainfall']), order=(2, 0, 1)).fit()
y_hat_avg['ARIMA'] = fit1.predict(start=1, end=365)

#Plotting

plt.figure(figsize=(10,7))
plt.plot( train['rainfall'], label='Train')
plt.plot(test['rainfall'], label='Test')
plt.plot(y_hat_avg['ARIMA'], label='ARIMA')
plt.legend(loc='best')

plt.title('Rainfall')
plt.xlabel('index')
plt.ylabel('Rainfall')
#plt.show()

#Calculating Error

ms_rf=mean_squared_error(test['rainfall'], y_hat_avg.ARIMA)
rms_rf = sqrt(mean_squared_error(test['rainfall'], y_hat_avg.ARIMA))
ms.append(round(ms_rf,2))
rms.append(round(rms_rf,2))

final_list['rainfall']=y_hat_avg.ARIMA


input("Press enter to continue...\n")


#--------------------PREDICTING EVP-------------------

print("Predicting EVP")

fit1 = sm.tsa.arima_model.ARIMA(np.asarray(train['EVP']), order=(2, 0, 1)).fit()
y_hat_avg['ARIMA'] = fit1.predict(start=1, end=365)

#Plotting

plt.figure(figsize=(10,7))
plt.plot( train['rainfall'], label='Train')
plt.plot(test['rainfall'], label='Test')
plt.plot(y_hat_avg['ARIMA'], label='ARIMA')
plt.legend(loc='best')

plt.title('Rainfall')
plt.xlabel('index')
plt.ylabel('Rainfall')
#plt.show()

#Calculating Error

ms_evp=mean_squared_error(test['EVP'], y_hat_avg.ARIMA)
rms_evp = sqrt(mean_squared_error(test['EVP'], y_hat_avg.ARIMA))
ms.append(round(ms_evp,2))
rms.append(round(rms_evp,2))

final_list['EVP']=y_hat_avg.ARIMA


input("Press enter to continue...\n")


#-----------------------CALCULATING ERROR-------------------------

final=scaler.inverse_transform(final_list)
test=scaler.inverse_transform(test)
# print(final)
final=pd.DataFrame(final, columns=name1)
test=pd.DataFrame(test, columns=name1)
#print(final)

# print(100*final['humidity']/test['humidity'])

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

acc.append(round(100-abs((sum(final_list['maxT'])-sum(copy_test['maxT']))*100/sum(copy_test['maxT'])),2))
acc.append(round(100-abs((sum(final_list['minT'])-sum(copy_test['minT']))*100/sum(copy_test['minT'])),2))
acc.append(round(100-abs((sum(final_list['humidity'])-sum(copy_test['humidity']))*100/sum(copy_test['humidity'])),2))
acc.append(round(100-abs((sum(final_list['windspeed'])-sum(copy_test['windspeed']))*100/sum(copy_test['windspeed'])),2))
acc.append(round(100-abs((sum(final_list['rainfall'])-sum(copy_test['rainfall']))*100/sum(copy_test['rainfall'])),2))
acc.append(round(100-abs((sum(final_list['sunshine'])-sum(copy_test['sunshine']))*100/sum(copy_test['sunshine'])),2))
acc.append(round(100-abs((sum(final_list['EVP'])-sum(copy_test['EVP']))*100/sum(copy_test['EVP'])),2))


ms1=[]
diff=copy_test-final_list
ms1.append(round(sum(diff['humidity']**2)/len(test),2))
ms1.append(round(sum(diff['maxT']**2)/len(test),2))
ms1.append(round(sum(diff['minT']**2)/len(test),2))
ms1.append(round(sum(diff['windspeed']**2)/len(test),2))
ms1.append(round(sum(diff['sunshine']**2)/len(test),2))
ms1.append(round(sum(diff['rainfall']**2)/len(test),2))
ms1.append(round(sum(diff['EVP']**2)/len(test),2))

#---------------------------------OUTPUT---------------------------

dash='-'*100
print(dash)
print('          {:<10s}{:>4s}{:>12s}{:>12s}{:>12s}{:>12s}'.format(name1[0],name1[1],name1[2],name1[3],name1[4],name1[5],name1[6]))
print(dash) 
print('MSE:      {:<10.2f}{:>4.2f}{:>12.2f}{:>12.2f}{:>12.2f}{:>12.2f}'.format(ms[0],ms[1],ms[2],ms[3],ms[4],ms[5],ms[6]))
print('RMSE:     {:<10.2f}{:>4.2f}{:>12.2f}{:>12.2f}{:>12.2f}{:>12.2f}'.format(rms[0],rms[1],rms[2],rms[3],rms[4],rms[5],rms[6]))
print('Accuracy: {:<10.2f}{:>4.2f}{:>12.2f}{:>12.2f}{:>12.2f}    '.format(acc[0],acc[1],acc[2],acc[3],acc[4]),acc[5],acc[6])
 
print(ms1)
#writing data into excel file
final_list.to_excel("output_arima.xlsx")
