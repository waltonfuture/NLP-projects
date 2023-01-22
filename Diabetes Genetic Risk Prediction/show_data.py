import pandas as pd
train_df=pd.read_csv('比赛训练集.csv',encoding='gbk')
test_df=pd.read_csv('比赛测试集.csv',encoding='gbk')
'''
print('训练集的数据大小：',train_df.shape)
print('测试集的数据大小：',test_df.shape)
print('-'*30)
print('训练集的数据类型：')
print(train_df.dtypes)
print('-'*30)
print(test_df.dtypes)
'''

#----------------查数据的缺失值----------------
print(train_df.isnull().sum())
print('-'*30)
print(test_df.isnull().sum())
#可以看到 训练集和测试集中都是舒张压有缺失值

#----------------查数据相关性----------------
print('-'*30)
print('查看训练集中数据的相关性')
print(train_df.corr())
print(test_df.corr())
#----------------数据的可视化统计----------------
import matplotlib.pyplot as plt
import seaborn as sns

train_df['性别'].value_counts().plot(kind='barh')
sns.set(font='SimHei',font_scale=1.1)  # 解决Seaborn中文显示问题并调整字体大小
sns.countplot(x='患有糖尿病标识', hue='性别', data=train_df)
sns.boxplot(y='出生年份', x='患有糖尿病标识', hue='性别', data=train_df)
sns.violinplot(y='体重指数', x='患有糖尿病标识', hue='性别', data=train_df)

plt.figure(figsize = [20,10],dpi=100)
sns.countplot(x='出生年份',data=train_df)
plt.tight_layout()