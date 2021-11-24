import os
import glob
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer

pipelinepath = "C:/Users/omsin/PycharmProjects/pythonProject/Checkpoint in data preparation pipeline/"



#Merge Price
path = "C:/Users/omsin/PycharmProjects/pythonProject/Dataset/Elspot Prices Hourly EUR/"
all_files = glob.glob(os.path.join(path, "elspot-prices_*_hourly_eur.csv"))
df_from_each_file = (pd.read_csv(f, sep=',', encoding='latin1') for f in all_files)
temp = list(df_from_each_file)
for y in temp:
  y.drop(y.columns.difference(['Hours', 'FI', 'Unnamed: 0']), axis=1, inplace=True)
temp2 = pd.concat(temp, ignore_index=True)
temp2['Hours'] = temp2['Hours'].str.replace(r'\D', '', regex=True)
temp2['Unnamed: 0'] = temp2['Unnamed: 0'].str.replace('-','/')
temp2['FI'] = temp2['FI'].str.replace(',','.')
temp2['DateTime'] = temp2['Unnamed: 0'] + ' ' + temp2['Hours']
dti = pd.to_datetime(temp2['DateTime'], format='%d/%m/%Y %H', exact=False)
temp2.set_index(pd.Index(dti), inplace=True)
temp2.drop(['Hours', 'Unnamed: 0', 'DateTime'], axis=1, inplace=True)
P_index = temp2.index



#Replace missing price data
P_imp = IterativeImputer(max_iter=10, verbose=0, skip_complete=True)
P_imp.fit(temp2)
P_imputed = P_imp.transform(temp2)
P_imputed = pd.DataFrame(P_imputed, columns=temp2.columns)
P_imputed.set_index(P_index, inplace=True)



#Merge Weather
path2 = "C:/Users/omsin/PycharmProjects/pythonProject/Dataset/Weather/"
W_all_files = glob.glob(os.path.join(path2, "csv-*.csv"))
W_df_from_each_file = (pd.read_csv(f, sep=',') for f in W_all_files)
W_temp = list(W_df_from_each_file)
W_temp2 = pd.date_range("2013-01-01", periods=21000, freq="H")
W_temp3 = pd.DataFrame(range(len(W_temp2)), index=W_temp2)
y = 1
for x in W_temp:
    x['DateTime'] = x['Year'].astype(str) + '-' + x['m'].astype(str) + '-' + x['d'].astype(str) + ' ' + x['Time']
    dti = pd.to_datetime(x['DateTime'], format='%Y-%m-%d %H:%M')
    x.set_index(pd.Index(dti), inplace=True)
    x.drop(['Year', 'm', 'd', 'Time', 'Time zone', 'DateTime'], axis=1, inplace=True)
    x = x.add_prefix(str(y)+"_")
    W_temp3 = W_temp3.join(x)
    y += 1
W_temp3.drop([0], axis=1, inplace=True)
W_temp3.index.name = 'DateTime'



#Weather drop column (missing value > 10%)
percent_missing = W_temp3.isnull().sum() * 100 / len(W_temp3)
missing_value_W_temp3 = pd.DataFrame({
  'column_name': W_temp3.columns,
  'percent_missing': percent_missing
})
temp_missing = missing_value_W_temp3.loc[ missing_value_W_temp3['percent_missing'] > 10, :]
missing = list(temp_missing['column_name'])
missing_value_W_temp3.sort_values('percent_missing', inplace=True)
W_temp3.drop(missing, axis=1, inplace=True)
W_index = W_temp3.index



#Replace missing weather data
W_imp = IterativeImputer(max_iter=10, verbose=0, skip_complete=True, n_nearest_features=10)
W_imp.fit(W_temp3)
W_imputed = W_imp.transform(W_temp3)
W_imputed = pd.DataFrame(W_imputed, columns=W_temp3.columns)
W_imputed.set_index(W_index, inplace=True)



#Add noise to weather data
N_mean = W_imputed.mean(axis=0)
N_index = N_mean.index
for x in N_index:
    noise = np.random.normal(0, 1, W_imputed[x].shape)
    W_imputed[x] = W_imputed[x] + noise



#Merge weather data and immediate electricity price
weather_FI = P_imputed.join(W_imputed)
weather_FI.to_csv(pipelinepath+"01_weather_FI.csv")



#Merge Capacities
path = "C:/Users/omsin/PycharmProjects/pythonProject/Dataset/Elspot capacities/"
all_files = glob.glob(os.path.join(path, "elspot-capacities-fi_*_hourly.csv"))
df_from_each_file = (pd.read_csv(f, sep=',', encoding='latin1') for f in all_files)
temp = list(df_from_each_file)
temp2 = pd.concat(temp, ignore_index=True)
temp2['Hours'] = temp2['Hours'].str.replace(r'\D', '', regex=True)
temp2['Unnamed: 0'] = temp2['Unnamed: 0'].str.replace('-','/')
temp2['DateTime'] = temp2['Unnamed: 0'] + ' ' + temp2['Hours']
dti = pd.to_datetime(temp2['DateTime'], format='%d/%m/%Y %H', exact=False)
temp2.set_index(pd.Index(dti), inplace=True)
temp2.drop(['Hours', 'Unnamed: 0', 'DateTime'], axis=1, inplace=True)
C_index = temp2.index



#Capacities drop column (missing value > 10%)
percent_missing = temp2.isnull().sum() * 100 / len(temp2)
missing_value_temp2 = pd.DataFrame({
  'column_name': temp2.columns,
  'percent_missing': percent_missing
})
temp_missing = missing_value_temp2.loc[ missing_value_temp2['percent_missing'] > 10, :]
missing = list(temp_missing['column_name'])
missing_value_temp2.sort_values('percent_missing', inplace=True)
temp2.drop(missing, axis=1, inplace=True)




#Replace missing Capacities data
C_imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
C_imp.fit(temp2)
C_imputed = C_imp.transform(temp2)
C_imputed = pd.DataFrame(C_imputed, columns=temp2.columns)
C_imputed.set_index(C_index, inplace=True)
C_imputed = C_imputed.add_prefix("C_")




#Merge Flow
path = "C:/Users/omsin/PycharmProjects/pythonProject/Dataset/Elspot flow/"
all_files = glob.glob(os.path.join(path, "elspot-flow-fi_*_hourly.csv"))
df_from_each_file = (pd.read_csv(f, sep=',', encoding='latin1') for f in all_files)
temp = list(df_from_each_file)
temp2 = pd.concat(temp, ignore_index=True)
temp2['Hours'] = temp2['Hours'].str.replace(r'\D', '', regex=True)
temp2['Unnamed: 0'] = temp2['Unnamed: 0'].str.replace('-', '/')
temp2_index = temp2.shape
for i in range(temp2_index[1]):
    temp2[temp2.columns[i]] = temp2[temp2.columns[i]].str.replace(',', '.')
temp2['DateTime'] = temp2['Unnamed: 0'] + ' ' + temp2['Hours']
dti = pd.to_datetime(temp2['DateTime'], format='%d/%m/%Y %H', exact=False)
temp2.set_index(pd.Index(dti), inplace=True)
temp2.drop(['Hours', 'Unnamed: 0', 'DateTime'], axis=1, inplace=True)
F_index = temp2.index



#Flow drop column (missing value > 10%)
percent_missing = temp2.isnull().sum() * 100 / len(temp2)
missing_value_temp2 = pd.DataFrame({
  'column_name': temp2.columns,
  'percent_missing': percent_missing
})
temp_missing = missing_value_temp2.loc[missing_value_temp2['percent_missing'] > 10, :]
missing = list(temp_missing['column_name'])
missing_value_temp2.sort_values('percent_missing', inplace=True)
temp2.drop(missing, axis=1, inplace=True)



#Replace missing Flow data
F_imp = IterativeImputer(max_iter=10, verbose=0, skip_complete=True)
F_imp.fit(temp2)
F_imputed = F_imp.transform(temp2)
F_imputed = pd.DataFrame(F_imputed, columns=temp2.columns)
F_imputed.set_index(F_index, inplace=True)
F_imputed = F_imputed.add_prefix("F_")




#Merge Volumes
path = "C:/Users/omsin/PycharmProjects/pythonProject/Dataset/Elspot volumes/"
all_files = glob.glob(os.path.join(path, "elspot-volumes_*_hourly.csv"))
df_from_each_file = (pd.read_csv(f, sep=',', encoding='latin1') for f in all_files)
temp = list(df_from_each_file)
temp2 = pd.concat(temp, ignore_index=True)
temp2['Hours'] = temp2['Hours'].str.replace(r'\D', '', regex=True)
temp2['Unnamed: 0'] = temp2['Unnamed: 0'].str.replace('-', '/')
temp2_index = temp2.shape
for i in range(temp2_index[1]):
    temp2[temp2.columns[i]] = temp2[temp2.columns[i]].str.replace(',', '.')
temp2['DateTime'] = temp2['Unnamed: 0'] + ' ' + temp2['Hours']
dti = pd.to_datetime(temp2['DateTime'], format='%d/%m/%Y %H', exact=False)
temp2.set_index(pd.Index(dti), inplace=True)
temp2.drop(['Hours', 'Unnamed: 0', 'DateTime'], axis=1, inplace=True)
temp2.drop(temp2.columns.difference(['FI Buy', 'FI Sell']), axis=1, inplace=True)
V_index = temp2.index



#Volumes drop column (missing value > 10%)
percent_missing = temp2.isnull().sum() * 100 / len(temp2)
missing_value_temp2 = pd.DataFrame({
  'column_name': temp2.columns,
  'percent_missing': percent_missing
})
temp_missing = missing_value_temp2.loc[missing_value_temp2['percent_missing'] > 10, :]
missing = list(temp_missing['column_name'])
missing_value_temp2.sort_values('percent_missing', inplace=True)
temp2.drop(missing, axis=1, inplace=True)



#Replace missing Volumes data
V_imp = IterativeImputer(max_iter=10, verbose=0, skip_complete=True)
V_imp.fit(temp2)
V_imputed = V_imp.transform(temp2)
V_imputed = pd.DataFrame(V_imputed, columns=temp2.columns)
V_imputed.set_index(V_index, inplace=True)



#Merge all extend
extend = C_imputed.join(F_imputed)
extend = extend.join(V_imputed)
extend.to_csv(pipelinepath+"02_Extend.csv")



#Merge Time
weather_FI_temp = weather_FI.copy()
weather_FI_temp['DateTime'] = weather_FI_temp.index
time = weather_FI_temp.drop(weather_FI_temp.columns.difference(['DateTime']), axis=1)
time['Day_of_Week'] = time['DateTime'].dt.dayofweek
time['Weekend'] = np.where(time['Day_of_Week'] >= 5, 1, 0)
time['Quarter'] = time['DateTime'].dt.quarter
time['hour'] = time['DateTime'].dt.hour
time['Month'] = time['DateTime'].dt.month
time['Year'] = time['DateTime'].dt.year
Working_Hour_Turth = (time['hour'] >= 8) & (time['hour'] <= 16)
time['Working_Time'] = np.where(Working_Hour_Turth, 1, 0)
time['Working_Time'] = np.where(time['Weekend'] == 1, 0, time['Working_Time'])
time['Parts_of_the_Day'] = 0
Afternoon_Turth = (time['hour'] >= 12) & (time['hour'] <= 16)
time['Parts_of_the_Day'] = np.where(Afternoon_Turth, 1, time['Parts_of_the_Day'])
Evening_Turth = (time['hour'] >= 17) & (time['hour'] <= 20)
time['Parts_of_the_Day'] = np.where(Evening_Turth, 2, time['Parts_of_the_Day'])
Night_Turth = ((time['hour'] >= 21) & (time['hour'] <= 23)) | ((time['hour'] >= 0) & (time['hour'] <= 5))
time['Parts_of_the_Day'] = np.where(Night_Turth, 3, time['Parts_of_the_Day'])
time.drop(['DateTime'], axis=1, inplace=True)
time.to_csv(pipelinepath+"03_Time.csv")



all_feature = weather_FI.join(extend)
all_feature = all_feature.join(time)
all_feature.to_csv(pipelinepath+"04_All_Feature.csv")

weather_FI_extend = weather_FI.join(extend)
weather_FI_extend.to_csv(pipelinepath+"05_Weather_FI_Extend.csv")

weather_FI_time = weather_FI.join(time)
weather_FI_time.to_csv(pipelinepath+"06_Weather_FI_Time.csv")