#!/usr/bin/env python
# coding: utf-8

# In[290]:


import numpy as np
import pandas as pd
import seaborn as sns
sns.set_context("paper")   

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import statistics
import math
from random import random
from IPython.display import display
from sklearn.preprocessing import LabelEncoder


d_train = pd.read_csv('train.csv')
d_test = pd.read_csv('test.csv')


# In[291]:


def prtSep(length=20, marker="-"):
    print( marker * length)


# In[305]:


##### Baseline Analysis #####
# フィールドの比較 - テストに目的変数以外で存在していない項目はドロップしてもよい
display(d_train.head())
display(d_test.head())

prtSep()

# レコード数の確認
print("Shape of Train:" , d_train.shape)
# (891, 12)
print("Shape of Test:" , d_test.shape)
# (418, 11)
prtSep()

print(d_train.describe())
prtSep()
print(d_train.info())


# In[293]:


##### Preliminary EDA #####
Target = ['Survived']
data1_x = ['Sex','Pclass', 'Embarked', 'SibSp', 'Parch', 'Age', 'Fare'] # pretty name/values for charts


# In[294]:


# Survival
print("Survived Ratio:")
print(d_train['Survived'].value_counts())
print(d_train['Survived'].value_counts(normalize=True))
sns.countplot(d_train['Survived'])
prtSep()


# In[295]:


# Categoral Value Count
i = 1
for col in data1_x:
    if d_train[col].dtype != 'float64' and col != 'Survived':
        print(d_train[col].value_counts())
        print(d_train[col].value_counts(normalize=True))
        prtSep()
    i += 1


# In[296]:


# Categoral count plot
fig = plt.figure(figsize=(30, 60))
i = 1
for col in data1_x:
    if d_train[col].dtype != 'float64' and col != 'Survived':
        fig.add_subplot(10,2,i)
        ax = sns.countplot(x=col, data=d_train);
        i += 1
        ax = fig.add_subplot(10,2,i)
        sns.countplot(x=col, data=d_train,hue='Survived');
        i += 1


# In[297]:


# Survival Correlation for Categoral
for x in data1_x:
    if d_train[x].dtype != 'float64' :
        print('Survival Correlation by:', x)
        print(d_train[[x, Target[0]]].groupby(x, as_index=False).mean())
        print('-'*10, '\n')
        d_train['Died'] = 1 - d_train['Survived']
        d_train.groupby(x).agg('mean')[['Survived', 'Died']].plot(kind='bar', figsize=(25, 7), 
                                                           stacked=True, colors=['g', 'r']);


# In[298]:


i=1
fig = plt.figure(figsize=(20, 30))
for x in data1_x:
        if d_train[x].dtype == 'float64' :
            ax = fig.add_subplot(4,2,i)

            ax.hist(d_train[x].dropna(), rwidth=0.5)
            ax.set_xlabel(x)
   
            ax = fig.add_subplot(4,2,i+1)
            splitAsSurvived = []

             # データセットを生存者で分割
            for i in [0, 1]:
                splitAsSurvived.append(d_train[d_train.Survived == i])
            tmp  = [i[x].dropna() for i in splitAsSurvived]
            ax.set_xlabel(x)
            ax.hist(tmp, histtype="barstacked", bins=10, rwidth=0.5, color=['red', 'blue'], label=['not survived', 'survived'])
            ax.legend()
            i += 2
            


# In[299]:


# Name
print(d_train['Name'].head())
prtSep()
d_train['Name_Title'] = d_train['Name'].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split()[0])
print("Count by Name Title:")
print(d_train['Name_Title'].value_counts())
prtSep()
print("Survied Ratio by Name Title:")
d_train['Survived'].groupby(d_train['Name_Title']).mean()
d_train.drop(['Name_Title'], axis=1, inplace=True)


# In[300]:


##### Correcting Step #####
# No Correcting at this moment


# In[301]:


##### Completing Step #####
data_cleaner = [d_train, d_test]

# First review 
sns.heatmap(d_train.isnull(),yticklabels=False,cbar=False,cmap='viridis')

# print("Count record of Total in Train:")
# print(d_train.count())
print("Count record of Null in Train:")
print(d_train.isnull().sum())
# print("Count record of Total in Test:")
# print(d_test.count())
print("Count record of Null in Test:")
print(d_test.isnull().sum())


# In[302]:


# Complete Embarked : カテゴリカルデータのなので最頻値で補完
pd.value_counts(d_train['Embarked'])
'''
S    644
C    168
Q     77
Name: Embarked, dtype: int64
'''
d_train["Embarked"] = d_train["Embarked"].fillna(d_train["Embarked"].mode()[0])

# Compelte Fare : 件数少ないので中央値を使う
d_test["Fare"] = d_test["Fare"].fillna(d_test['Fare'].median())

# Complete Age : 連続データなので中央値を使う
d_train["Age"] = d_train["Age"].fillna(d_train['Age'].median())
d_test["Age"] = d_test["Age"].fillna(d_test['Age'].median())


# In[303]:


# Verify 欠損値
sns.heatmap(d_train.isnull(),yticklabels=False,cbar=False,cmap='viridis')

print("Count record of Null in Train:")
print(d_train.isnull().sum())
print("Count record of Null in Test:")
print(d_test.isnull().sum())


# In[304]:


##### Creating #####

# Title
def get_title(name):
    if '.' in name:
        return name.split(',')[1].split('.')[0].strip()
    else:
        return 'Unknown'
        
Title_Dictionary = {
    "Capt": "Officer",
    "Col": "Officer",
    "Major": "Officer",
    "Jonkheer": "Royalty",
    "Don": "Royalty",
    "Dona": "Royalty",
    "Sir" : "Royalty",
    "Dr": "Officer",
    "Rev": "Officer",
    "the Countess":"Royalty",
    "Mme": "Mrs",
    "Mlle": "Miss",
    "Ms": "Mrs",
    "Mr" : "Mr",
    "Mrs" : "Mrs",
    "Miss" : "Miss",
    "Master" : "Master",
    "Lady" : "Royalty"
}
d_train['Title'] = d_train['Name'].apply(get_title).map(Title_Dictionary)
d_test['Title'] = d_test['Name'].apply(get_title).map(Title_Dictionary)

#Continuous variable bins; qcut vs cut: （連続値を離散化する、頻度基準 vs 値域基準） https://stackoverflow.com/questions/30211923/what-is-the-difference-between-pandas-qcut-and-pandas-cut
#Fare Bins/Buckets using qcut or frequency bins:（） https://pandas.pydata.org/pandas-docs/stable/generated/pandas.qcut.html
d_train['FareBin'] = pd.qcut(d_train['Fare'], 4)
d_test['FareBin'] = pd.qcut(d_test['Fare'], 4)

#Age Bins/Buckets using cut or value bins: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.cut.html
d_train['AgeBin'] = pd.cut(d_train['Age'].astype(int), 5)
d_test['AgeBin'] = pd.cut(d_test['Age'].astype(int), 5)

for dataset in data_cleaner:
    #Discrete variables
    dataset['FamilySize'] = dataset ['SibSp'] + dataset['Parch'] + 1

    dataset['IsAlone'] = 1 #initialize to yes/1 is alone
    dataset['IsAlone'].loc[dataset['FamilySize'] > 1] = 0 # now update to no/0 if family size is greater than 1
    
# 
def ticket_grouped(train, test):
    for i in [train, test]:
        i['Ticket_Lett'] = i['Ticket'].apply(lambda x: str(x)[0])
        i['Ticket_Lett'] = i['Ticket_Lett'].apply(lambda x: str(x))
        i['Ticket_Lett'] = np.where((i['Ticket_Lett']).isin(['1', '2', '3', 'S', 'P', 'C', 'A']), i['Ticket_Lett'],
                                   np.where((i['Ticket_Lett']).isin(['W', '4', '7', '6', 'L', '5', '8']),
                                            'Low_ticket', 'Other_ticket'))
        i['Ticket_Len'] = i['Ticket'].apply(lambda x: len(x))
        del i['Ticket']
    return train, test
d_train, d_test = ticket_grouped(d_train, d_test)

def cabin(train, test):
    for i in [train, test]:
        i['Cabin_Letter'] = i['Cabin'].apply(lambda x: str(x)[0])
        # del i['Cabin']
    return train, test
d_train, d_test = cabin(d_train, d_test)

def cabin_num(train, test):
    for i in [train, test]:
        i['Cabin_num1'] = i['Cabin'].apply(lambda x: str(x).split(' ')[-1][1:])
        i['Cabin_num1'].replace('an', np.NaN, inplace = True)
        i['Cabin_num1'] = i['Cabin_num1'].apply(lambda x: int(x) if not pd.isnull(x) and x != '' else np.NaN)
        i['Cabin_num'] = pd.qcut(train['Cabin_num1'],3)
        # train = pd.concat((train, pd.get_dummies(train['Cabin_num'], prefix = 'Cabin_num')), axis = 1)
    # test = pd.concat((test, pd.get_dummies(test['Cabin_num'], prefix = 'Cabin_num')), axis = 1)
    #del train['Cabin_num']
    #del test['Cabin_num']
    #del train['Cabin_num1']
    #del test['Cabin_num1']
    return train, test
d_train, d_test = cabin_num(d_train, d_test)


# In[274]:


d_train["Cabin_num"].


# In[217]:


##### Converting #####
#code categorical data （）
label = LabelEncoder()
for dataset in data_cleaner:    
    dataset['Sex_Code'] = label.fit_transform(dataset['Sex'])
    dataset['Embarked_Code'] = label.fit_transform(dataset['Embarked'])
    dataset['Title_Code'] = label.fit_transform(dataset['Title'])
    dataset['AgeBin_Code'] = label.fit_transform(dataset['AgeBin'])
    dataset['FareBin_Code'] = label.fit_transform(dataset['FareBin'])
    dataset['Ticket_Lett_Code'] = label.fit_transform(dataset['Ticket_Lett'])
    dataset['Cabin_Lett_Code'] = label.fit_transform(dataset['Cabin_Letter'])


# In[218]:


#define y variable aka target/outcome
Target = ['Survived']

# define x variables for original w/bin features to remove continuous variables
# 例えばAgeであれば、Ageを使えばよく、Age, Agebin, AgeBin_Codeは除外できる
data1_x_bin = ['Sex_Code', 'Pclass', 'Embarked_Code', 'Title_Code', 'FamilySize', 'AgeBin_Code', 'FareBin_Code', 'Ticket_Lett_Code', 'Ticket_Len', 'Cabin_Lett_Code', 'Cabin_num']
data1_xy_bin = Target + data1_x_bin
print('Bin X Y: ', data1_xy_bin, '\n')


# define x variables for original features aka feature selection
data1_x = ['Sex','Pclass', 'Embarked', 'Title','SibSp', 'Parch', 'Age', 'Fare', 'FamilySize', 'IsAlone', 'Ticket_Lett', 'Ticket_Len', 'Cabin_Letter'] # pretty name/values for charts
data1_x_calc = ['Sex_Code','Pclass', 'Embarked_Code', 'Title_Code','SibSp', 'Parch', 'Age', 'Fare'] #coded for algorithm calculation
data1_xy =  Target + data1_x
print('Original X Y: ', data1_xy, '\n')

#define x and y variables for dummy features original
d_train_dummy = pd.get_dummies(d_train[data1_x]) 
# d_trainをWoBしたデータフレームはd_train_dummy
# なぜLEしたSex_CodeではなくSexをDummyかしているのか
d_train_x_dummy = d_train_dummy.columns.tolist()
d_train_xy_dummy = Target + d_train_x_dummy # data1_xをdummy化している
print('Dummy X Y: ', d_train_xy_dummy, '\n')

d_train_dummy.head()


# In[219]:


# Double check
print('Train columns with null values: \n', d_train.isnull().sum())
prtSep()
print (d_train.info())
prtSep()

print('Test/Validation columns with null values: \n', d_train.isnull().sum())
prtSep()
print (d_test.info())
prtSep()


# In[220]:


##### EDA #####

#Discrete Variable Correlation by Survival using
#group by aka pivot table: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.groupby.html
for x in data1_x:
    if d_train[x].dtype != 'float64' :
        print('Survival Correlation by:', x)
        print(d_train[[x, Target[0]]].groupby(x, as_index=False).mean())
        print('-'*10, '\n')


# In[221]:


#graph distribution of quantitative data
plt.figure(figsize=[16,12])

plt.subplot(231)
plt.boxplot(x=d_train['Fare'], showmeans = True, meanline = True)
plt.title('Fare Boxplot')
plt.ylabel('Fare ($)')

plt.subplot(232)
plt.boxplot(d_train['Age'], showmeans = True, meanline = True)
plt.title('Age Boxplot')
plt.ylabel('Age (Years)')

plt.subplot(233)
plt.boxplot(d_train['FamilySize'], showmeans = True, meanline = True)
plt.title('Family Size Boxplot')
plt.ylabel('Family Size (#)')

plt.subplot(234)
plt.hist(x = [d_train[d_train['Survived']==1]['Fare'], d_train[d_train['Survived']==0]['Fare']], 
         stacked=True, color = ['g','r'],label = ['Survived','Dead'])
plt.title('Fare Histogram by Survival')
plt.xlabel('Fare ($)')
plt.ylabel('# of Passengers')
plt.legend()

plt.subplot(235)
plt.hist(x = [d_train[d_train['Survived']==1]['Age'], d_train[d_train['Survived']==0]['Age']], 
         stacked=True, color = ['g','r'],label = ['Survived','Dead'])
plt.title('Age Histogram by Survival')
plt.xlabel('Age (Years)')
plt.ylabel('# of Passengers')
plt.legend()

plt.subplot(236)
plt.hist(x = [d_train[d_train['Survived']==1]['FamilySize'], d_train[d_train['Survived']==0]['FamilySize']], 
         stacked=True, color = ['g','r'],label = ['Survived','Dead'])
plt.title('Family Size Histogram by Survival')
plt.xlabel('Family Size (#)')
plt.ylabel('# of Passengers')
plt.legend()


# In[222]:


#we will use seaborn graphics for multi-variable comparison:
#graph individual features by survival
fig, saxis = plt.subplots(2, 3,figsize=(16,12))

sns.barplot(x = 'Embarked', y = 'Survived', data=d_train, ax = saxis[0,0])
sns.barplot(x = 'Pclass', y = 'Survived', order=[1,2,3], data=d_train, ax = saxis[0,1])
sns.barplot(x = 'IsAlone', y = 'Survived', order=[1,0], data=d_train, ax = saxis[0,2])
sns.barplot(x = 'FareBin', y = 'Survived',  data=d_train, ax = saxis[1,0])
sns.barplot(x = 'AgeBin', y = 'Survived',  data=d_train, ax = saxis[1,1])
sns.barplot(x = 'FamilySize', y = 'Survived', data=d_train, ax = saxis[1,2])


# In[223]:


#correlation heatmap of dataset
def correlation_heatmap(df):
    _ , ax = plt.subplots(figsize =(14, 12))
    colormap = sns.diverging_palette(220, 10, as_cmap = True)
    
    _ = sns.heatmap(
        df.corr(), 
        cmap = colormap,
        square=True, 
        cbar_kws={'shrink':.9 }, 
        ax=ax,
        annot=True, 
        linewidths=0.1,vmax=1.0, linecolor='white',
        annot_kws={'fontsize':12 }
    )
    
    plt.title('Pearson Correlation of Features', y=1.05, size=15)

correlation_heatmap(d_train)


# In[224]:


from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from xgboost import XGBClassifier


#Common Model Helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics


#Machine Learning Algorithm (MLA) Selection and Initialization
MLA = [
    #Ensemble Methods
    ensemble.AdaBoostClassifier(),
    ensemble.BaggingClassifier(),
    ensemble.ExtraTreesClassifier(),
    ensemble.GradientBoostingClassifier(),
    ensemble.RandomForestClassifier(),

    #Gaussian Processes
    gaussian_process.GaussianProcessClassifier(),
    
    #GLM
    linear_model.LogisticRegressionCV(),
    linear_model.PassiveAggressiveClassifier(),
    linear_model.RidgeClassifierCV(),
    linear_model.SGDClassifier(),
    linear_model.Perceptron(),
    
    #Navies Bayes
    naive_bayes.BernoulliNB(),
    naive_bayes.GaussianNB(),
    
    #Nearest Neighbor
    neighbors.KNeighborsClassifier(),
    
    #SVM
    svm.SVC(probability=True),
    svm.NuSVC(probability=True),
    svm.LinearSVC(),
    
    #Trees    
    tree.DecisionTreeClassifier(),
    tree.ExtraTreeClassifier(),
    
    #Discriminant Analysis
    discriminant_analysis.LinearDiscriminantAnalysis(),
    discriminant_analysis.QuadraticDiscriminantAnalysis(),

    
    #xgboost: http://xgboost.readthedocs.io/en/latest/model.html
    XGBClassifier()    
    ]



#split dataset in cross-validation with this splitter class: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ShuffleSplit.html#sklearn.model_selection.ShuffleSplit
#note: this is an alternative to train_test_split
cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = 0 ) # run model 10x with 60/30 split intentionally leaving out 10%

#create table to compare MLA metrics
MLA_columns = ['MLA Name', 'MLA Parameters','MLA Train Accuracy Mean', 'MLA Test Accuracy Mean', 'MLA Test Accuracy 3*STD' ,'MLA Time']
MLA_compare = pd.DataFrame(columns = MLA_columns)

#create table to compare MLA predictions
MLA_predict = d_train[Target]

#index through MLA and save performance to table
row_index = 0
for alg in MLA:

    #set name and parameters
    MLA_name = alg.__class__.__name__
    MLA_compare.loc[row_index, 'MLA Name'] = MLA_name
    MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())
    
    #score model with cross validation: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate
    cv_results = model_selection.cross_validate(alg, d_train[data1_x_bin], d_train[Target], cv  = cv_split)

    MLA_compare.loc[row_index, 'MLA Time'] = cv_results['fit_time'].mean()
    # MLA_compare.loc[row_index, 'MLA Train Accuracy Mean'] = cv_results['train_score'].mean()
    MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = cv_results['test_score'].mean()   
    #if this is a non-bias random sample, then +/-3 standard deviations (std) from the mean, should statistically capture 99.7% of the subsets
    MLA_compare.loc[row_index, 'MLA Test Accuracy 3*STD'] = cv_results['test_score'].std()*3   #let's know the worst that can happen!
    

    #save MLA predictions - see section 6 for usage
    alg.fit(d_train[data1_x_bin], d_train[Target])
    MLA_predict[MLA_name] = alg.predict(d_train[data1_x_bin])
    
    row_index+=1

    
#print and sort table: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.sort_values.html
MLA_compare.sort_values(by = ['MLA Test Accuracy Mean'], ascending = False, inplace = True)
MLA_compare
#MLA_predict


# In[ ]:


#barplot using https://seaborn.pydata.org/generated/seaborn.barplot.html
sns.barplot(x='MLA Test Accuracy Mean', y = 'MLA Name', data = MLA_compare, color = 'm')

#prettify using pyplot: https://matplotlib.org/api/pyplot_api.html
plt.title('Machine Learning Algorithm Accuracy Score \n')
plt.xlabel('Accuracy Score (%)')
plt.ylabel('Algorithm')


# In[ ]:


#base model
dtree = tree.DecisionTreeClassifier(random_state = 0)
base_results = model_selection.cross_validate(dtree, d_train[data1_x_bin], d_train[Target], cv  = cv_split)
dtree.fit(d_train[data1_x_bin], d_train[Target])

print('BEFORE DT Parameters: ', dtree.get_params())
# print("BEFORE DT Training w/bin score mean: {:.2f}". format(base_results['train_score'].mean()*100)) 
print("BEFORE DT Test w/bin score mean: {:.2f}". format(base_results['test_score'].mean()*100))
print("BEFORE DT Test w/bin score 3*std: +/- {:.2f}". format(base_results['test_score'].std()*100*3))
#print("BEFORE DT Test w/bin set score min: {:.2f}". format(base_results['test_score'].min()*100))
print('-'*10)


#tune hyper-parameters: http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier
param_grid = {'criterion': ['gini', 'entropy'],  #scoring methodology; two supported formulas for calculating information gain - default is gini
              #'splitter': ['best', 'random'], #splitting methodology; two supported strategies - default is best
              'max_depth': [2,4,6,8,10,None], #max depth tree can grow; default is none
              #'min_samples_split': [2,5,10,.03,.05], #minimum subset size BEFORE new split (fraction is % of total); default is 2
              #'min_samples_leaf': [1,5,10,.03,.05], #minimum subset size AFTER new split split (fraction is % of total); default is 1
              #'max_features': [None, 'auto'], #max features to consider when performing split; default none or all
              'random_state': [0] #seed or control random number generator: https://www.quora.com/What-is-seed-in-random-number-generation
             }

#print(list(model_selection.ParameterGrid(param_grid)))

#choose best model with grid_search: #http://scikit-learn.org/stable/modules/grid_search.html#grid-search
#http://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html
tune_model = model_selection.GridSearchCV(tree.DecisionTreeClassifier(), param_grid=param_grid, scoring = 'roc_auc', cv = cv_split)
tune_model.fit(d_train[data1_x_bin], d_train[Target])

#print(tune_model.cv_results_.keys())
#print(tune_model.cv_results_['params'])
print('AFTER DT Parameters: ', tune_model.best_params_)
#print(tune_model.cv_results_['mean_train_score'])
# print("AFTER DT Training w/bin score mean: {:.2f}". format(tune_model.cv_results_['mean_train_score'][tune_model.best_index_]*100)) 
#print(tune_model.cv_results_['mean_test_score'])
print("AFTER DT Test w/bin score mean: {:.2f}". format(tune_model.cv_results_['mean_test_score'][tune_model.best_index_]*100))
print("AFTER DT Test w/bin score 3*std: +/- {:.2f}". format(tune_model.cv_results_['std_test_score'][tune_model.best_index_]*100*3))
print('-'*10)


#duplicates gridsearchcv
#tune_results = model_selection.cross_validate(tune_model, data1[data1_x_bin], data1[Target], cv  = cv_split)

#print('AFTER DT Parameters: ', tune_model.best_params_)
#print("AFTER DT Training w/bin set score mean: {:.2f}". format(tune_results['train_score'].mean()*100)) 
#print("AFTER DT Test w/bin set score mean: {:.2f}". format(tune_results['test_score'].mean()*100))
#print("AFTER DT Test w/bin set score min: {:.2f}". format(tune_results['test_score'].min()*100))
#print('-'*10)


# In[ ]:


X_test = d_test[data1_x_bin]
Y_pred = tune_model.predict(X_test)
        
kaggle_submission = pd.DataFrame({
        "PassengerId": d_test["PassengerId"],
        "Survived": Y_pred
    })
kaggle_submission.to_csv("randomforest01-ticket-cabin.csv", index=False)


# In[ ]:


# 特徴量の確認
feature_dataframe = pd.DataFrame( {'features': X_train.columns,
                                   'RandomForest': clfs['random_forest'].feature_importances_,
                                   'ExtraTrees':  clfs['extra_trees'].feature_importances_,
                                   'AdaBoost': clfs['ada_boost'].feature_importances_,
                                   'GradiendBoostiong': clfs['gradient_boosting'].feature_importances_,
                                   'XGBoost': clfs['xgboost'].feature_importances_,
})
feature_dataframe.plot.bar(x='features', figsize=(12, 3))


# In[256]:


d_train.head()


# In[ ]:




