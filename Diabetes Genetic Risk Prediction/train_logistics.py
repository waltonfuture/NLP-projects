
#!pip install sklearn
#!pip install pandas

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
# preprocess the data
data1=pd.read_csv('比赛训练集.csv',encoding='gbk')
data2=pd.read_csv('比赛测试集.csv',encoding='gbk')

data2['患有糖尿病标识']=-1
data=pd.concat([data1,data2],axis=0,ignore_index=True)

data['舒张压']=data['舒张压'].fillna(0)

#----------------feature engineering----------------

data['出生年份']=2023 - data['出生年份']
# classify the age, BMi, family history of diabetes and diastolic pressure

def resetAge(input):
    if input<=18:
        return 0
    elif 19<=input<=30:
        return 1
    elif 31<=input<=50:
        return 2
    elif input>=51:
        return 3

data['rAge']=data['出生年份'].apply(resetAge)


def BMI(a):
    if a<18.5:
        return 0
    elif 18.5<=a<=24:
        return 1
    elif 24<a<=27:
        return 2
    elif 27<a<=32:
        return 3
    else:
        return 4

data['BMI']=data['体重指数'].apply(BMI)


def FHOD(a):
    if a=='无记录':
        return 0
    elif a=='叔叔或者姑姑有一方患有糖尿病' or a=='叔叔或姑姑有一方患有糖尿病':
        return 1
    else:
        return 2


data['糖尿病家族史']=data['糖尿病家族史'].apply(FHOD)

def DBP(a):
    if a<60:
        return 0
    elif 60<=a<=90:
        return 1
    elif a>90:
        return 2
    else:
        return a
data['DBP']=data['舒张压'].apply(DBP)


# split the train_dataset and test_dataset

train=data[data['患有糖尿病标识'] !=-1]
test=data[data['患有糖尿病标识'] ==-1]
train_label=train['患有糖尿病标识']
train=train.drop(['编号','患有糖尿病标识'],axis=1)
test=test.drop(['编号','患有糖尿病标识'],axis=1)

# create model and train

model = make_pipeline(
    MinMaxScaler(),
    LogisticRegression()
)
model.fit(train,train_label)

# test
pre_y=model.predict(test)

# write results
result=pd.read_csv('提交示例.csv')
result['label']=pre_y
result.to_csv('LR.csv',index=False)