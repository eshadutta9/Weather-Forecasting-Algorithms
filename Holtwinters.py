import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
import statsmodels.api as sm
from sklearn.model_selection import cross_val_score
import csv
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler


ms=[]
rms=[]
acc=[]
ms1=[]

name1=['maxT','minT','humidity','windspeed','rainfall','sunshine','EVP']
df = pd.read_csv('Anand.csv', usecols=[1,2,3,5,6,7,8], names=name1)

# df=preprocessing.normalize(df)
# df=pd.DataFrame(df, columns=name1)

#Creating train and test set 

copy_test=df[-365:]

scaler=StandardScaler()
scaler.fit(df)

train=df[8036:-365] 
test=df[-365:]



#---------1. PREDICTING RELATIVE HUMIDITY------------

print("Predicting humidity")


y_hat_avg1 = test.copy()
fit1 = ExponentialSmoothing(np.asarray(train['humidity']) , seasonal_periods=2,trend='add', seasonal='add').fit()
#y_hat_avg['Holt_Winter'] = fit1.forecast(len(test))
y_hat_avg1['Holt_Winter'] = fit1.predict(start=0,end=364)

#plotting

plt.figure()
plt.plot( train['humidity'], label='Train')
plt.plot(test['humidity'], label='Test')
plt.plot(y_hat_avg1['Holt_Winter'], label='Holt-Winters')

plt.title('RH1')
plt.xlabel('date')
plt.ylabel('Relative humidity')

plt.legend(loc='best')
#plt.show()

#calculating error

ms_rh1=mean_squared_error(test['humidity'], y_hat_avg1.Holt_Winter)
rms_rh1 = sqrt(mean_squared_error(test['humidity'], y_hat_avg1.Holt_Winter))
ms.append(round(ms_rh1,2))
rms.append(round(rms_rh1,2))

final_list=pd.DataFrame()
final_list['humidity']=y_hat_avg1.Holt_Winter

# acc_hum=(100*final_list['humidity'])/test['humidity']
# acc.append(round((acc_hum.sum()/len(acc_hum)),2))

# X1=scaler.inverse_transform(final_list)
# print(X1)

input("Press enter to continue...\n")


#--------PREDICTING MAXIMUM TEMPERATURE--------------

print("Predicting Maximum Temperature")

y_hat_avg2 = test.copy()

fit2=ExponentialSmoothing(np.asarray(train['maxT']), seasonal_periods=2,trend='add',seasonal='add').fit()
#y_hat_avg['Holt_Winter'] = fit2.forecast(len(test))
y_hat_avg2['Holt_Winter'] = fit2.predict(start=0,end=364)

plt.figure()
plt.plot(train['maxT'],label='Train')
plt.plot(test['maxT'],label='Test')
plt.plot(y_hat_avg2['Holt_Winter'],label='Holt_Winters')

plt.title('MaxT')
plt.xlabel('date')
plt.ylabel('Maximum Temperature')

plt.legend(loc='best')
#plt.show()


ms_maxt=mean_squared_error(test['maxT'],y_hat_avg2.Holt_Winter)
rms_maxt=sqrt(mean_squared_error(test['maxT'],y_hat_avg2.Holt_Winter))
ms.append(round(ms_maxt,2))
rms.append(round(rms_maxt,2))

final_list['maxT']=y_hat_avg2.Holt_Winter

# acc_maxt=(100*final_list['maxT'])/test['maxT']
# acc.append(round((acc_maxt.sum()/len(acc_maxt)),2))


input("Press enter to continue...\n")


#---------PREDICTING MINIMUM TEMPERATURE---------


print("Predicting Minimum Temperature")

y_hat_avg3 = test.copy()
fit3=ExponentialSmoothing(np.asarray(train['minT']), seasonal_periods=2,trend='add',seasonal='add').fit()
#y_hat_avg['Holt_Winter'] = fit3.forecast(len(test))
y_hat_avg3['Holt_Winter'] = fit3.predict(start=0,end=364)

plt.figure()
plt.plot(train['minT'],label='Train')
plt.plot(test['minT'],label='Test')
plt.plot(y_hat_avg3['Holt_Winter'],label='Holt_Winters')

plt.title('MinT')
plt.xlabel('date')
plt.ylabel('Minimum Temperature')

plt.legend(loc='best')
#plt.show()


ms_mint=mean_squared_error(test['minT'],y_hat_avg3.Holt_Winter)
rms_mint=sqrt(mean_squared_error(test['minT'],y_hat_avg3.Holt_Winter))
ms.append(round(ms_mint,2))
rms.append(round(rms_mint,2))

final_list['minT']=y_hat_avg3.Holt_Winter

# acc_mint=(100*final_list['minT'])/test['minT']
# acc.append(round((acc_mint.sum()/len(acc_mint)),2))

input("Press enter to continue...\n")

#---------PREDICTING WINDSPEED--------------


print("Predicting Windspeed")

y_hat_avg4 = test.copy()
fit4=ExponentialSmoothing(np.asarray(train['windspeed']), seasonal_periods=2,trend='add',seasonal='add').fit()
#y_hat_avg['Holt_Winter'] = fit4.predict(start=0,end=364)
y_hat_avg4['Holt_Winter'] = fit4.predict(start=0,end=364)


plt.figure()
plt.plot(train['windspeed'],label='Train')
plt.plot(test['windspeed'],label='Test')
plt.plot(y_hat_avg4['Holt_Winter'],label='Holt_Winters')

plt.title('Windspeed')
plt.xlabel('date')
plt.ylabel('Windspeed')

plt.legend(loc='best')
#plt.show()


ms_ws=mean_squared_error(test['windspeed'],y_hat_avg4.Holt_Winter)
rms_ws=sqrt(mean_squared_error(test['windspeed'],y_hat_avg4.Holt_Winter))
ms.append(round(ms_ws,2))
rms.append(round(rms_ws,2))

final_list['windspeed']=y_hat_avg4.Holt_Winter

# acc_ws=(100*final_list['windspeed'])/test['windspeed']
# acc.append(round((acc_ws.sum()/len(acc_ws)),2))

input("Press enter to continue...\n")

# #----------PREDICTING SUNSHINE---------------

print("Predicting Sunshine")

y_hat_avg5 = test.copy()
fit5=ExponentialSmoothing(np.asarray(train['sunshine']), seasonal_periods=2,trend='add',seasonal='add').fit()
#y_hat_avg['Holt_Winter'] = fit5.predict(start=0,end=364)
y_hat_avg5['Holt_Winter'] = fit5.predict(start=0,end=364)


plt.figure()
plt.plot(train['sunshine'],label='Train')
plt.plot(test['sunshine'],label='Test')
plt.plot(y_hat_avg5['Holt_Winter'],label='Holt_Winters')

plt.title('sunshine')
plt.xlabel('date')
plt.ylabel('sunshine')

plt.legend(loc='best')
#plt.show()

ms_sun=mean_squared_error(test['sunshine'],y_hat_avg5.Holt_Winter)
rms_sun=sqrt(mean_squared_error(test['sunshine'],y_hat_avg5.Holt_Winter))
ms.append(round(ms_sun,2))
rms.append(round(rms_sun,2))

final_list['sunshine']=y_hat_avg5.Holt_Winter

# acc_sun=(100*final_list['sunshine'])/test['sunshine']
# acc.append(round((acc_sun.sum()/len(acc_sun)),2))


input("Press enter to continue...\n")

#----------------PREDICTING RAINFALL----------------


print("Predicting Rainfall")

y_hat_avg5 = test.copy()
fit5=ExponentialSmoothing(np.asarray(train['rainfall']), seasonal_periods=2,trend='add',seasonal='add').fit()
#y_hat_avg['Holt_Winter'] = fit5.predict(start=0,end=364)
y_hat_avg5['Holt_Winter'] = fit5.predict(start=0,end=364)


plt.figure()
plt.plot(train['rainfall'],label='Train')
plt.plot(test['rainfall'],label='Test')
plt.plot(y_hat_avg5['Holt_Winter'],label='Holt_Winters')

plt.title('rainfall')
plt.xlabel('date')
plt.ylabel('rainfall')

plt.legend(loc='best')
#plt.show()

ms_sun=mean_squared_error(test['rainfall'],y_hat_avg5.Holt_Winter)
rms_sun=sqrt(mean_squared_error(test['rainfall'],y_hat_avg5.Holt_Winter))
ms.append(round(ms_sun,2))
rms.append(round(rms_sun,2))

final_list['rainfall']=y_hat_avg5.Holt_Winter

# acc_rf=(100*final_list['rainfall'])/test['rainfall']
# acc.append(round((acc_rf.sum()/len(acc_rf)),2))


input("Press enter to continue...\n")



#----------------PREDICTING EVP----------------


print("Predicting EVP")

y_hat_avg6 = test.copy()
fit6=ExponentialSmoothing(np.asarray(train['EVP']), seasonal_periods=2,trend='add',seasonal='add').fit()
#y_hat_avg['Holt_Winter'] = fit5.predict(start=0,end=364)
y_hat_avg6['Holt_Winter'] = fit6.predict(start=0,end=364)


plt.figure()
plt.plot(train['EVP'],label='Train')
plt.plot(test['EVP'],label='Test')
plt.plot(y_hat_avg6['Holt_Winter'],label='Holt_Winters')

plt.title('EVP')
plt.xlabel('date')
plt.ylabel('EVP')

plt.legend(loc='best')
#plt.show()

ms_evp=mean_squared_error(test['EVP'],y_hat_avg6.Holt_Winter)
rms_evp=sqrt(mean_squared_error(test['EVP'],y_hat_avg6.Holt_Winter))
ms.append(round(ms_evp,2))
rms.append(round(rms_evp,2))

final_list['EVP']=y_hat_avg6.Holt_Winter

# acc_rf=(100*final_list['rainfall'])/test['rainfall']
# acc.append(round((acc_rf.sum()/len(acc_rf)),2))


input("Press enter to continue...\n")


#-----------OUTPUT------------------


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

#acc.append(100-(sum(final['humidity'])-sum(test['humidity'])*100/sum(test['humidity'])))

# Accuracy Calculation
acc.append(round(100-abs((sum(final_list['maxT'])-sum(copy_test['maxT']))*100/sum(copy_test['maxT'])),2))
acc.append(round(100-abs((sum(final_list['minT'])-sum(copy_test['minT']))*100/sum(copy_test['minT'])),2))
acc.append(round(100-abs((sum(final_list['humidity'])-sum(copy_test['humidity']))*100/sum(copy_test['humidity'])),2))
acc.append(round(100-abs((sum(final_list['windspeed'])-sum(copy_test['windspeed']))*100/sum(copy_test['windspeed'])),2))
acc.append(round(100-abs((sum(final_list['rainfall'])-sum(copy_test['rainfall']))*100/sum(copy_test['rainfall'])),2))
acc.append(round(100-abs((sum(final_list['sunshine'])-sum(copy_test['sunshine']))*100/sum(copy_test['sunshine'])),2))
acc.append(round(100-abs((sum(final_list['EVP'])-sum(copy_test['EVP']))*100/sum(copy_test['EVP'])),2))
# Manual Calculation of MSE
diff=copy_test-final_list
ms1.append(round(sum(diff['humidity']**2)/len(test),2))
ms1.append(round(sum(diff['maxT']**2)/len(test),2))
ms1.append(round(sum(diff['minT']**2)/len(test),2))
ms1.append(round(sum(diff['windspeed']**2)/len(test),2))
ms1.append(round(sum(diff['sunshine']**2)/len(test),2))
ms1.append(round(sum(diff['rainfall']**2)/len(test),2))
ms1.append(round(sum(diff['EVP']**2)/len(test),2))

dash='-'*100
print(dash)
print('          {:<10s}{:>4s}{:>12s}{:>12s}{:>12s}{:>12s}'.format(name1[0],name1[1],name1[2],name1[3],name1[4],name1[5],name1[6]))
print(dash) 
print('MSE:      {:<10.2f}{:>4.2f}{:>12.2f}{:>12.2f}{:>12.2f}{:>12.2f}{:>12.2f}'.format(ms[0],ms[1],ms[2],ms[3],ms[4],ms[5],ms[6]))
print('RMSE:     {:<10.2f}{:>4.2f}{:>12.2f}{:>12.2f}{:>12.2f}{:>12.2f}{:>12.2f}'.format(rms[0],rms[1],rms[2],rms[3],rms[4],rms[5],rms[6]))
print('Accuracy: {:<10.2f}{:>4.2f}{:>12.2f}{:>12.2f}{:>12.2f}{:>12.2f}{:>12.2f}'.format(acc[0],acc[1],acc[2],acc[3],acc[4],acc[5],acc[6]))

print(ms1)

 #writing data into CSV file
final_list.to_excel("output.xlsx")