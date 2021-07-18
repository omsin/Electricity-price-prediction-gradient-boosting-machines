import pandas as pd
import numpy as np
from IPython.display import display
from fastai.imports import *
from sklearn import metrics
from pandas.tseries.offsets import DateOffset
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

hour_ahead = 25

path = "C:/Users/omsin/PycharmProjects/pythonProject/Checkpoint in data preparation pipeline/09 Merge all.csv"
Merge_all = pd.read_csv(path)
dti = pd.to_datetime(Merge_all['DateTime'], format='%Y-%m-%d %H', exact=False)
Merge_all.set_index(pd.Index(dti), inplace=True)
Merge_all.drop(['DateTime'], axis=1, inplace=True)

First = Merge_all.iloc[0:10001, :]
N_index = First.shape
Y_temp = First['FI']
Y = Y_temp.iloc[hour_ahead:N_index[0]]
#Y = Y.to_frame()
#Y.reset_index(drop=True, inplace=True)
#Y.columns = [''] * len(Y.columns)


X_temp1 = First['FI']
X_temp2 = First.loc[:, First.columns != 'FI']
X_temp2_index = X_temp2.index + DateOffset(hours=-hour_ahead)
X_temp2.set_index(X_temp2_index, inplace=True)
X_temp1 = X_temp1.to_frame()
X_temp = X_temp1.join(X_temp2)
X = X_temp.iloc[0:N_index[0] - hour_ahead]
#X = X.to_numpy()
#X.reset_index(drop=True, inplace=True)
#X.columns = [''] * len(X.columns)
Pred_index = X.shape


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,mean_squared_error

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, shuffle=False)
RandomForestRegModel = RandomForestRegressor(n_estimators=100)
RandomForestRegModel.fit(x_train, y_train)
y_test_index = y_test.shape

y_pred = RandomForestRegModel.predict(x_test)
print(y_pred)
MSE = mean_squared_error(y_test, y_pred, squared=True)
print(MSE)
RMSE = np.sqrt(MSE)
print(RMSE)
MAE = mean_absolute_error(y_test, y_pred)
print(MAE)


plt.figure(figsize=(10, 6), dpi=80)
plt.suptitle('Random Forests sklearn N=' + str(Pred_index[0]))
plt.plot(np.arange(len(y_test))+1, y_test, color="blue", linewidth=2.5, linestyle="-", label='Real Price')
plt.plot(np.arange(len(y_pred))+1, y_pred, color="red",  linewidth=2.5, linestyle="-", label='Predicted Price')
plt.xlabel('Hours')
plt.ylabel('Price(EUR)')
plt.legend(frameon=False)
plt.show()



plt.figure(figsize=(10, 6), dpi=80)
plt.suptitle('Random Forests sklearn N=' + str(Pred_index[0]))
plt.plot(np.arange(len(y_test[y_test_index[0]-500:y_test_index[0]]))+1, y_test[y_test_index[0]-500:y_test_index[0]], color="blue", linewidth=2.5, linestyle="-", label='Real Price')
plt.plot(np.arange(len(y_pred[y_test_index[0]-500:y_test_index[0]]))+1, y_pred[y_test_index[0]-500:y_test_index[0]], color="red",  linewidth=2.5, linestyle="-", label='Predicted Price')
plt.xlabel('Hours')
plt.ylabel('Price(EUR)')
plt.legend(frameon=False)
plt.show()

