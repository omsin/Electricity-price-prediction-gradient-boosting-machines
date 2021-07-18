import os
import glob
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

pipelinepath = "C:/Users/omsin/PycharmProjects/pythonProject/Checkpoint in data preparation pipeline/"



#Merge Price
path = "C:/Users/omsin/PycharmProjects/pythonProject/Dataset/Elspot Prices Hourly EUR/"
all_files = glob.glob(os.path.join(path, "elspot-prices_*_hourly_eur.csv"))
df_from_each_file = (pd.read_csv(f, sep=',') for f in all_files)
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
temp2.to_csv(pipelinepath+"01 Price merged.csv")
P_index = temp2.index



#Replace missing price data
P_imp = IterativeImputer(max_iter=10, verbose=0, skip_complete=True)
P_imp.fit(temp2)
P_imputed = P_imp.transform(temp2)
P_imputed = pd.DataFrame(P_imputed, columns=temp2.columns)
P_imputed.set_index(P_index, inplace=True)
P_imputed.to_csv(pipelinepath+"02 Imputed price merged.csv")



#Merge Weather
path2 = "C:/Users/omsin/PycharmProjects/pythonProject/Dataset/Weather/"
W_all_files = glob.glob(os.path.join(path2, "csv-*.csv"))
W_df_from_each_file = (pd.read_csv(f, sep=',') for f in W_all_files)
W_temp = list(W_df_from_each_file)
W_temp2 = pd.date_range("2013-01-01", periods=10000, freq="H")
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
W_temp3.to_csv(pipelinepath+"03 Weather merged.csv")



#Weather drop column (missing value > 15%)
percent_missing = W_temp3.isnull().sum() * 100 / len(W_temp3)
missing_value_W_temp3 = pd.DataFrame({
  'column_name': W_temp3.columns,
  'percent_missing': percent_missing
})
missing_value_W_temp3.to_csv(pipelinepath+"04 Weather missing value.csv")
temp_missing = missing_value_W_temp3.loc[ missing_value_W_temp3['percent_missing'] > 15, :]
missing = list(temp_missing['column_name'])
missing_value_W_temp3.sort_values('percent_missing', inplace=True)
missing_value_W_temp3.to_csv(pipelinepath+"05 Weather missing value (sorting).csv")
W_temp3.drop(missing, axis=1, inplace=True)
W_temp3.to_csv(pipelinepath+"06 Weather merged (drop missing column).csv")
W_index = W_temp3.index



#Replace missing weather data
W_imp = IterativeImputer(max_iter=20, verbose=0, skip_complete=True)
W_imp.fit(W_temp3)
W_imputed = W_imp.transform(W_temp3)
W_imputed = pd.DataFrame(W_imputed, columns=W_temp3.columns)
W_imputed.set_index(W_index, inplace=True)
W_imputed.to_csv(pipelinepath+"07 Imputed Weather merged.csv")



#Add noise to weather data
N_mean = W_imputed.mean(axis=0)
N_index = N_mean.index
for x in N_index:
    noise = np.random.normal(0, 1, W_imputed[x].shape)
    W_imputed[x] = W_imputed[x] + noise
W_imputed.to_csv(pipelinepath+"08 Imputed Weather merged (Add noise).csv")



#Merge all
Merge_all = P_imputed.join(W_imputed)
Merge_all.to_csv(pipelinepath+"09 Merge all.csv")


