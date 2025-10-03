#Bryant Berrio
#501162030
#Project 1

#1 Data Processing (csv to DataFrame)

import pandas as pd

df=pd.read_csv("Project 1 Data.csv")
print(df.shape)
print(df.head())
print(df.dtypes)
print(df.isna().sum())   #verifies missing data

#2 Data Visualization

