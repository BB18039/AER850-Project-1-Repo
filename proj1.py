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

#4 classification model dev/eng (preprocessing, model pipelines, gridsearch and random searchcv)

import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

df['Step']=df['Step'].astype('category')
x=df[['X','Y','Z']].values
yraw=df['Step'].values #categorical labels
le=LabelEncoder()
y=le.fit_transform(yraw)
print("Label mapping:", dict(enumerate(le.classes_)))

#train/test split
xtrain, xtest, ytrain, ytest=train_test_split(x,y,test_size=0.2, stratify=y, random_state=42)
print("Train/test sizes:", xtrain.shape,xtest.shape)

#pipeline
pipe=Pipeline([('scaler',StandardScaler()),('clf',LogisticRegression(max_iter=1000))]) #scaler scales features

#parameter grids for GridSearchCV
param_grid_lr={'clf__penalty':['l2'],'clf__C':[0.01,0.1,1.0,10.0],'clf__solver':['lbfgs']}
param_grid_svc={'clf__C':[0.1,1.0,10.0],'clf__kernel':['rbf','linear'], 'clf__gamma':['scale','auto']}
param_grid_rf={'clf__max_depth':[None,5,10],'clf__min_samples_leaf':[1,2,4]}

#GridSearchCV setup
commoncv=5
scoringchoice='f1_macro' #treats classes equally

#Logic Regression GridSearch
pipe.set_params(clf=LogisticRegression(max_iter=1000))
gs_lr=GridSearchCV(pipe,param_grid=param_grid_lr, cv=commoncv, scoring=scoringchoice, n_jobs=-1,verbose=1)
gs_lr.fit(xtrain,ytrain)
print("Best LR parameters:",gs_lr.best_params_)
print("Best LR CV Score:", gs_lr.best_score_)

#SVC GridSearch
pipe.set_params(clf=SVC(probability=True))
gs_svc=GridSearchCV(pipe, param_grid=param_grid_svc, cv=commoncv, scoring=scoringchoice, n_jobs=-1, verbose=1)
gs_svc.fit(xtrain,ytrain)
print("Best SVC parameters:",gs_svc.best_params_)
print("Best SVC CV score:", gs_svc.best_score_)

#Random Forest GridSearch
pipe.set_params(clf=RandomForestClassifier(random_state=42))
gs_rf=GridSearchCV(pipe, param_grid=param_grid_rf, cv=commoncv, scoring=scoringchoice, n_jobs=-1, verbose=1)
gs_rf.fit(xtrain, ytrain)
print("Best RF parameters:", gs_rf.best_params_)
print("Best RF CV score:", gs_rf.best_score_)

#RandomSearchCV for RandomForest

from scipy.stats import randint
rndparamdist={'clf__n_estimators':randint(50,300),'clf__max_depth':[None,5,10,20],'clf__min_samples_leaf':randint(1,6)}
pipe.set_params(clf=RandomForestClassifier(random_state=42))
rnd_search=RandomizedSearchCV(pipe, param_distributions=rndparamdist, n_iter=20, cv=commoncv, scoring=scoringchoice, random_state=42, n_jobs=-1,verbose=1)
rnd_search.fit(xtrain, ytrain)
print("Best randomized RF parameters:",rnd_search.best_params_)
print("Best randomized RF CV score:",rnd_search.best_score_)

#best estimators evaluated on test set
models={'LogisticRegression':gs_lr.best_estimator_,'SVC':gs_svc.best_estimator_,'RandomForest':gs_rf.best_estimator_,'RandomizedRF':rnd_search.best_estimator_}
for name, model in models.items():
    ypred=model.predict(xtest)
    print("\n=== Evaluation:",name)
    print("Accuracy:",accuracy_score(ytest, ypred))
    print("Macro F1:",f1_score(ytest, ypred, average='macro'))
    print(classification_report(ytest, ypred))
    cm=confusion_matrix(ytest, ypred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm,annot=True, fmt='d')
    plt.title(f'Confusion matrix-{name}')
    plt.xlabel('pred');plt.ylabel('true')
    plt.tight_layout();plt.show()
    
    #chosen model is logistic regression so model and encoder will be saved using joblib
    best_logreg=gs_lr.best_estimator_ #pipeline with scaler+logistic regression
    joblib.dump(best_logreg,'best_pipeline_logreg.joblib')
    print("Saved model --> best_pipeline_logreg.joblib")
    joblib.dump(le,'label_encoder.joblib')
    print("Saved label encoder --> label_encoder.joblib")