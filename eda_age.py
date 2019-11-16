import numpy as nm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statistics
import math
from random import random

%matplotlib inline

d_train = pd.read_csv('train.csv')
d_test = pd.read_csv('test.csv')

def exAge(df, min, max):
    data = df[(df['Age'] >= min) & (df['Age'] <= max)]
    return data[['Age', 'Name', 'Survived']]
 
def exAgeSurvived(df, min, max):
    data = df[(df['Age'] >= min) & (df['Age'] <= max) & (df['Survived'] == 1)]
    return data[['Age', 'Name', 'Survived']]
 
def calAgeSurvivedRatio(df, min, max):
    # print(str(min) + "Age ~ " + str(max) + " All�F" + str(len(exAge(df, min, max))))
    # print(str(min) + "Age ~ " + str(max) + " Survived�F" + str(len(exAgeSurvived(df, min, max))))
    df_1=exAge(df, min, max)
    if (len(df_1) == 0):
        return(str(min) + "Age ~ " + str(max) + " Survived Ratio�FNot available")
    else:
        return(str(min) + "Age ~ " + str(max) + " Survived Ratio�F" + str(100 * len(exAgeSurvived(df, min, max)) / len(exAge(df, min, max))) + "%")
    
for i in range(0,100,10):
    print(calAgeSurvivedRatio(d_train, i, i+10))
    
    
# �N��̃q�X�g�O����
plt.hist(d_train["Age"].dropna(), bins=10, rwidth=0.5)
plt.xlabel("Age")
plt.show()

splitAsSurvived = []

# �f�[�^�Z�b�g�𐶑��҂ŕ���
for i in [0, 1]:
    splitAsSurvived.append(d_train[d_train.Survived == i])

tmp  = [i["Age"].dropna() for i in splitAsSurvived]
plt.xlabel("Age")
plt.hist(tmp, histtype="barstacked", bins=10, rwidth=0.5, color=['red', 'blue'], label=['not survived', 'survived'])
plt.legend()
plt.show()

# ���� vs �N��

splitAsSex = [] # ���ʖ��Ƀf�[�^�Z�b�g�𕪊����邽�߂̃��X�g��p��

for i in ["male", "female"]: # 0:male, 1:female
    splitAsSex.append(d_train[d_train.Sex == i])

tmp = [i.Age.dropna() for i in splitAsSex]

plt.hist(tmp, histtype="barstacked", bins=10, rwidth=0.5, color=['pink', 'skyblue'], label=['female', 'male'])
plt.xlabel("Age")
plt.legend() #�}��\��
plt.show()