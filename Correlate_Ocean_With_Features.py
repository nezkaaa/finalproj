import pandas as pd
import numpy as np

filename='ocean_values_generation.csv'

# Load the data
df = pd.read_csv(filename)

# Define columns to keep
columns_to_keep = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness',
                   'instrumentalness', 'liveness', 'valence', 'tempo', 'Openness', 'Consciousness', 
                   'Extraversion', 'Agreeableness', 'Neuroticism']

# Drop unnecessary columns
df = df[columns_to_keep]

# Calculate the correlation
correlation_matrix = df.corr()

correlation_matrix.to_csv("Ocean_Audio_Features_Correlation.csv")
# Print the correlation matrix
print(correlation_matrix)
