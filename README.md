# Electricity price prediction gradient boosting machines

1.File description
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

2.File setting 
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
