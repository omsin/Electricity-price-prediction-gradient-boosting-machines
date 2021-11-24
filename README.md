# Electricity price prediction gradient boosting machines

Nowadays, electric power is one of the important things for everyone. It is connected
to almost everything in our lives and affects the economy, society and technology in
every second the world turns. Therefore, to obtain electricity there is a price to pay
from production and other expenses. Its price depends on the demand and supply
of electricity which is linked to other factors, for example, the effect of wind speed
on electricity generation capacity, the effect of temperature on human behaviour on
electricity demand in households and human behaviour at weekends and weekdays.

This code will predict and analysis the day-ahead wholesale electricity price of Finland by use
machine learning that is a gradient boosting machine and use the random forest,
linear regression, ridge regression, lasso and elastic net as benchmarks. The weather
data of Finland, electricity price of Finland, electricity market data and considerably
more had been used as features. The methodology is divided into 2 parts. The first
part is data preparation, we clean the data and focus on feature selection. The second
part is prediction, we tuning hyperparameter and implement gradient boosting
machine and benchmark. Finally, The result of feature selection and prediction
have been analysed for more insight to carry out what features and machines are
appropriate to predict day-ahead wholesale electricity price.

1. File description
	- data_preparation.py - Cleaning data and aggregate the data into 6 file:
		- 1st file) All weather features from 177 weather stations in Finland andimmediately electricity price (780 feature) file [01_weather_FI.csv]
		- 2nd file) All features from Nord Pool Market Data except immediately electricity price (14 feature) file [02_Extend.csv]
		- 3rd file) All extract feature (8 feature) [03_Time.csv]
		- 4th file) Combine 1st, 2nd and 3rd file [04_All_Feature.csv]
		- 5th file) Combine 1st and 2nd file [05_Weather_FI_Extend.csv]
		- 6th file) Combine 1st and 3rd file [06_Weather_FI_Time.csv]

	- gradient_boosting_regressor.py - main algorithm of gradient boosting machine 
	- random_forests.py - main algorithm of random forest
	- No noise folder - archive of algorithm and result of noise experiment (All file in this archive is ipynb format. You should use jupyter notebook or google colab to open.) 
	- Feature_experiment folder - archive of algorithm and result of feature selection on all approach (All file in this archive is ipynb format. You should use jupyter notebook or google colab to open.)
	- Hour Window folder - archive of algorithm and result of various hour ahead plot (All file in this archive is ipynb format. You should use jupyter notebook or google colab to open.) 
	- Implement space folder - archive of miscellaneous algorithm and result (All file in this archive is ipynb format. You should use jupyter notebook or google colab to open.) 

2. File setting 
	- data_preparation.py
		- adjust the mean and std of gaussian noise that combine with weather feature - go to line 98
		- adjust the per cent of missing data on the weather data and remove - go to line 68
		- adjust the per cent of missing data on the electricity capacities data and remove - go to line 122
		- adjust the per cent of missing data on the electricity flow data and remove - go to line 166
		- adjust the per cent of missing data on the trades volumes data and remove - go to line 210
	- gradient_boosting_regressor.py
		- adjust the file path - go to line 22
		- adjust the hour ahead that want to predict - go to line 14
		- adjust the number of estimators parameter - go to line 15
		- adjust the learning rate parameter - go to line 16
		- adjust the tree depth - go to line 17
		- adjust the minimum sample on the node of trees - go to line 18
	- random_forests.py
		- adjust the file path - go to line 20
		- adjust the hour ahead that want to predict - go to line 14
		- adjust the number of trees - go to line 15
		- adjust the tree depth - go to line 16
		- adjust the minimum sample on the node of trees - go to line 17
