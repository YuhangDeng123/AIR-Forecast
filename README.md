# ID2223 Final project_Air quality prediction for Guangzhou, China
There are 4 python files in google colab notebooks and a file containing the necessary files for the streamlit Huggingface. The function of these files are described as follows.
# 1_backfill_feature_groups
This file is to preprocess the data of weather and air quality in Guangzhou, China for one year, which is downloaded from the Internet. In the data of air quality, the index of pm25 is used to represent the index of air quality. In the weather data, data cleaning is performed by calculating the correlation between each label data and air quality, and the cleaned data is uploaded to hopsworks as two featurestores.

# 2_feature_pipeline
The previous data set was manually downloaded from the Internet, and this feature pipeline can update the daily weather and air quality data set in real time and insert them into the hopsworks' feature store.

# 3_feature_views_and_training_dataset
The previous weather and air quality exist as two independent feature data. For the next model training, this file normalizes and merges the two feature datasets to obtain one feature view for model training.

# 4_model_training(XGB) && 4_model_training
This file is to implement the XGBoost model training for the hopsworks' feature view. XGBoost is an intensive training for GBoost. Through special methods, we can get the model parameters which is more suitable for this model, and construct XGBoost training according to these model parameters. In addition, this file also compares the accuracy of XGBoost and GBoost model. The results show that the accuracy of the XGBoost model is higher than that of the GBoost model.

# Datasets prepared
Guangzhou-air-quality.csv and Guangzhou-weather.csv.


# Huggingface file
The Huggingface file mainly includes three sub-files： the requirement.txt which is installation package configuration file, functions.py which contains the function  used in app.py, and the app.py file is the streamlit interface driver file. This app gives an interface for forecast of air quality for the next week in Guangzhou, China.

# Huggingface space of public URL for real-time weather prediction 
https://huggingface.co/spaces/YuhangDeng123/AIR
