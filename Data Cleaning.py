import pandas as pd

df = pd.read_pickle('combined_data.pk1')
print(df.head())

df = df.drop('Unnamed: 0', axis=1)

print(df.head())