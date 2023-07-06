#!/usr/bin/env python
# coding: utf-8

# In[42]:


#Concern --> Accuracy
  #Change Dataset (scaling, cleaning, filtering, changing the way it is simulated)
#change Model (change percentage of train-test split, cleaning, different model [forest, k-nearest neighbors])


# # Simulating OCEAN Scores and Cleaning

# In[1]:


import pandas as pd
import numpy as np

filename='dataset.csv'
newsong=pd.read_csv(filename)

song = newsong[~newsong.duplicated(subset='track_name')].copy()
song.dropna(inplace=True)

# Assuming you have a DataFrame named 'song' with existing columns 'column1', 'column2', 'column3', and 'column4'

# Step 1: Generate correlated random numbers
n = len(song)  # Number of rows in the DataFrame
correlation_dict_O={'acousticness': -0.281, 'danceability': -0.014, 'duration_ms': 0.149, 'energy':0.263,'instrumentalness':-0.179,'liveness': 0.147,'loudness':0.139,'speechiness': 0.121,'tempo':0.086, 'valence':0.058}
correlation_dict_C={'acousticness': -0.001, 'danceability': -0.06, 'duration_ms': -0.045, 'energy':0.011,'instrumentalness':0.038,'liveness': 0.057,'loudness':0.029,'speechiness': -0.009,'tempo':0.004, 'valence':0.011}
correlation_dict_E={'acousticness': -0.019, 'danceability': -0.021, 'duration_ms': -0.056, 'energy':0.038,'instrumentalness':0.081,'liveness': 0.02,'loudness':0.056,'speechiness': -0.088,'tempo':0.01, 'valence':-0.115}
correlation_dict_A={'acousticness': -0.083, 'danceability': -0.081, 'duration_ms': -0.023, 'energy':0.073,'instrumentalness':0.081,'liveness': 0.079,'loudness':0.063,'speechiness': 0.046,'tempo':-0.052, 'valence':-0.012}
correlation_dict_N={'acousticness': 0.06, 'danceability': 0.069, 'duration_ms': 0.017, 'energy':-0.066,'instrumentalness':-0.013,'liveness': -0.019,'loudness':-0.063,'speechiness': 0.01,'tempo':-0.013, 'valence':0.035}

#OPENNESS

# Calculate the scaling factors based on the standard deviation
scaling_factors = {}
for column, correlation in correlation_dict_O.items():
    scaling_factors[column] = correlation

# Generate random numbers and scale them
random_numbers = np.random.normal(size=(n, len(correlation_dict_O)))
correlated_numbers = np.sum(random_numbers * list(scaling_factors.values()), axis=1)

# Min-Max scaling
min_value = correlated_numbers.min()
max_value = correlated_numbers.max()
scaled_numbers = (correlated_numbers - min_value) / (max_value - min_value)

# Step 2: Add the new column to the DataFrame
song.loc[:, 'Openness'] = scaled_numbers

#CONSCIOUSNESS

scaling_factors = {}
for column, correlation in correlation_dict_C.items():
    scaling_factors[column] = correlation

# Generate random numbers and scale them
random_numbers = np.random.normal(size=(n, len(correlation_dict_C)))
correlated_numbers = np.sum(random_numbers * list(scaling_factors.values()), axis=1)

# Min-Max scaling
min_value = correlated_numbers.min()
max_value = correlated_numbers.max()
scaled_numbers = (correlated_numbers - min_value) / (max_value - min_value)

# Step 2: Add the new column to the DataFrame
song.loc[:, 'Consciousness'] = scaled_numbers

#EXTRAVERSION

scaling_factors = {}
for column, correlation in correlation_dict_E.items():
    scaling_factors[column] = correlation

# Generate random numbers and scale them
random_numbers = np.random.normal(size=(n, len(correlation_dict_E)))
correlated_numbers = np.sum(random_numbers * list(scaling_factors.values()), axis=1)

# Min-Max scaling
min_value = correlated_numbers.min()
max_value = correlated_numbers.max()
scaled_numbers = (correlated_numbers - min_value) / (max_value - min_value)

# Step 2: Add the new column to the DataFrame
song.loc[:, 'Extraversion'] = scaled_numbers

#AGREEABLENESS

scaling_factors = {}
for column, correlation in correlation_dict_A.items():
    scaling_factors[column] = correlation

# Generate random numbers and scale them
random_numbers = np.random.normal(size=(n, len(correlation_dict_A)))
correlated_numbers = np.sum(random_numbers * list(scaling_factors.values()), axis=1)

# Min-Max scaling
min_value = correlated_numbers.min()
max_value = correlated_numbers.max()
scaled_numbers = (correlated_numbers - min_value) / (max_value - min_value)

# Step 2: Add the new column to the DataFrame
song.loc[:, 'Agreeableness'] = scaled_numbers

#NEUROTICISM

scaling_factors = {}
for column, correlation in correlation_dict_N.items():
    scaling_factors[column] = correlation

# Generate random numbers and scale them
random_numbers = np.random.normal(size=(n, len(correlation_dict_N)))
correlated_numbers = np.sum(random_numbers * list(scaling_factors.values()), axis=1)

# Min-Max scaling
min_value = correlated_numbers.min()
max_value = correlated_numbers.max()
scaled_numbers = (correlated_numbers - min_value) / (max_value - min_value)

# Step 2: Add the new column to the DataFrame
song.loc[:, 'Neuroticism'] = scaled_numbers

# Print the updated DataFrame
song


# # Regression Test

# In[3]:


import pandas as pd

#df = pd.read_excel("biased fake dataset again.xlsx")


# In[5]:


#Normalize Data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


labels = ['duration_ms', 'danceability', 'energy',
       'key', 'loudness', 'speechiness', 'acousticness',
       'instrumentalness', 'liveness', 'valence', 'tempo']

normalized_test =scaler.fit_transform(song[labels])


# In[6]:


features = ['Openness', 'Consciousness', 'Extraversion', 'Agreeableness' , 'Neuroticism']


#Assigning x and y values
x = song[features] #Input
y = pd.DataFrame(normalized_test, columns = labels)  #Normalized Labels (Output)


#Trying Train-Test Split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)


from xgboost import XGBRegressor

#XG Boost Regression
xgb_regressor = XGBRegressor(random_state=42) #Creates bot/template
xgb = xgb_regressor.fit(x_train, y_train)

print(xgb_regressor.score(x_test, y_test))


# In[10]:


#Getting Predictions

import matplotlib.pyplot as plt

y_pred = xgb_regressor.predict(x_test)


# In[12]:


#Getting MAE, MSE, RMSE

from sklearn.metrics import mean_absolute_error,mean_squared_error
 
mae = mean_absolute_error(y_true=y_test,y_pred=y_pred)
#squared True returns MSE value, False returns RMSE value.
mse = mean_squared_error(y_true=y_test,y_pred=y_pred) #default=True
rmse = mean_squared_error(y_true=y_test,y_pred=y_pred,squared=False)
 
print("MAE:",mae)
print("MSE:",mse)
print("RMSE:",rmse)


# # Recommending

# In[13]:


import numpy as np

O = float(input("O: "))
C = float(input("C: "))
E = float(input("E: "))
A = float(input("A: "))
N = float(input("N: "))


print("Loading Recommendations... Please Wait.")

user_personality = np.array([[O,C,E,A,N]])
user_personality = pd.DataFrame(user_personality, columns = features)

predictions = xgb.predict(user_personality)


# In[14]:


#Ranking the Similarity of Each Existing Score to the Predicted Score of the User
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


y_duplicate = y.copy()

array_vec_1 = np.array(predictions)
scores = []

for index, array in y.iterrows():
    
    score = cosine_similarity(array_vec_1, np.array([array]))  
    
    scores.append(score[0][0])
    

y_duplicate = y_duplicate.assign(score = scores)


# In[17]:


#Display Top 5 Most Similar Songs
y_duplicate = y_duplicate.sort_values('score', ascending = False)
top_indices = y_duplicate.index.values.tolist()[0:5]


print("Given Your Personality, You Might Like These Songs!")
counter = 1

for i in top_indices:
    print(counter, ":", song.iloc[i, 4], "by", song.iloc[i,2])
    counter+=1

