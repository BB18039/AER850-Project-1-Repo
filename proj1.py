#Bryant Berrio
#501162030
#Project 1

#1 data processing (csv to dataframe)

import pandas as pd

df=pd.read_csv("Project 1 Data.csv")
print(df.shape)
print(df.head())
print(df.dtypes)
print(df.isna().sum())   #checks for missing data

#2 data visualization

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df['Step']=df['Step'].astype('category')
df['stepcode']=df['Step'].cat.codes

#class count to show how balanced classes are

plt.figure(figsize=(10,4))
sns.countplot(x='Step',data=df, order=df['Step'].cat.categories)
plt.xticks(rotation=45)
plt.title("Counts per Step")
plt.tight_layout()
plt.show()

#histogram

df[['X','Y','Z']].hist(bins=30,figsize=(10,3))
plt.suptitle("Feature Histograms")
plt.tight_layout()
plt.show()

#2d pair scatterplot

#X vs Y
plt.figure(figsize=(6,5))
sns.scatterplot(data=df,x='X',y='Y',hue='stepcode',palette='tab20',legend=False,s=12)
plt.title("X vs Y")
plt.show()

#X vs Z
plt.figure(figsize=(6,5))
sns.scatterplot(data=df,x='X',y='Z',hue='stepcode',palette='tab20',legend=False,s=12)
plt.title("X vs Z")
plt.show()

#Y vs Z
plt.figure(figsize=(6,5))
sns.scatterplot(data=df,x='Y',y='Z',hue='stepcode',palette='tab20',legend=False,s=12)
plt.title("Y vs Z")
plt.show()

#correlation heatmap

plt.figure(figsize=(4,3))
corr=df[['X','Y','Z']].corr()
sns.heatmap(corr,annot=True,fmt='.2f',cmap='coolwarm')
plt.title("Pearson correlation between features")
plt.tight_layout();
plt.show()

#3 Correlation Analysis (Pearson Correlation)
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split

df['Step']=df['Step'].astype('category')
xall=df[['X','Y','Z']]
yall=df['Step']
xtrain, xtest,ytrain, ytest=train_test_split(xall,yall,test_size=0.2, stratify=yall, random_state=42)

ytraincodes= ytrain.cat.codes #integers 0 to n-1

corrfeats=xtrain.corr(method='pearson')
plt.figure(figsize=(4,3))
sns.heatmap(corrfeats, annot=True, fmt='.2f',cmap='coolwarm')
plt.title('Pearson Correlation - features')
plt.tight_layout()
plt.show()

rows=[]
for col in xtrain.columns:
    r,p=pearsonr(xtrain[col],ytraincodes)
    rows.append({'feature':col,'pearson r with step':r,'pvalue':p})
assocdf=pd.DataFrame(rows).sort_values('pearson r with step', key=abs, ascending=False)
print(assocdf)

