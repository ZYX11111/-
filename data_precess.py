import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')


"""
用随机森林缺失值填补，异常值
分割训练集和测试集
训练集train_data
分割完train_X, train_Y, test_X, test_Y
将训练集和测试集分割完之后，再将标签和数据集拼接起来k
处理完之后需要返回的数据 : 
train_X, train_Y, test_X, test_Y
ntrain_data   将train_X和train_Y拼接起来
ntest_data     将test_X和test_Y拼接起来
"""


def ProcessingData():
    train_data = pd.read_csv('./data/cs-training.csv')

    train_data = train_data.iloc[:, 1:]

    # print(train_data.shape)
    # train_data.info()
    # 打印的是header信息
    # print(train_data.info())

    mData = train_data.iloc[:, [5, 0, 1, 2, 3, 4, 6, 7, 8, 9]]

    # MonthlyIncome 非空
    train_known = mData[mData.MonthlyIncome.notnull()].as_matrix()

    # print(train_known.shape)
    # MonthlyIncome 缺失

    train_unknown = mData[mData.MonthlyIncome.isnull()].as_matrix()

    # print(train_unknown.shape)


    # 利用随机森林预测缺失值
    train_X = train_known[:, 1:]
    # print(train_X)
    train_y = train_known[:, 0]  # 0 列为MonthlyIncome缺失列
    rfr = RandomForestRegressor(random_state=0, n_estimators=200, max_depth=3, n_jobs=-1)
    rfr.fit(train_X, train_y)

    predicted_y = rfr.predict(train_unknown[:, 1:]).round(0)  # 利用预测值填充原始数据中的缺失值
    train_data.loc[train_data.MonthlyIncome.isnull(), 'MonthlyIncome'] = predicted_y

    train_data = train_data.dropna()  # 舍弃缺失值
    train_data = train_data.drop_duplicates()  # 舍弃重复值

    # # 异常值处理

    # train_box = train_data.iloc[:, [3, 7, 9]]
    # train_box.boxplot()

    # 去除异常值
    train_data = train_data[train_data['NumberOfTime30-59DaysPastDueNotWorse'] < 90]
    train_data = train_data[train_data['age'] > 0]
    train_data.to_csv('preprocession_data/train_data.csv')


if __name__ == "__main__":
    ProcessingData()
