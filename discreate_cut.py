import pandas as pd
import numpy as np
from interval import Interval
from sklearn.model_selection import train_test_split
import warnings
import pickle

warnings.filterwarnings("ignore")


# https://blog.csdn.net/weixin_37272286/article/details/81276110
# 上边网址是区间操作

def create_interval(x, y):
    # 生成区间，x: 起始值， y: 结束值， lower_closed: 是否包含起始值
    if y is float('inf'):
        zoom_x_y = Interval(x, y, lower_closed=False, upper_closed=False)
    else:
        zoom_x_y = Interval(x, y, lower_closed=False)
    return zoom_x_y
print(create_interval(4,5))

# train_x_feature: 训练集所有样本的某个特征值
# train_y: 对应的结果集：是否逾期超过90天，

def feature_cut(train_x_feature, train_y, zoom_x_y):
    data = pd.DataFrame({"index": train_x_feature.index, "X": train_x_feature, "Y": train_y})
    data['Bucket'] = data['X']
    lst = [i for i, dt in zip(data['index'], data["Bucket"]) if dt in zoom_x_y]
    data = data.loc[lst]
    data.loc[:, 'Bucket'] = zoom_x_y
    data.dropna(inplace=True)
    data.set_index("index", inplace=True)
    return data


# https://blog.csdn.net/qq_34490873/article/details/81205523
# 上边的网址是对DataFrame中的某列数据格式的转化方法
def woe_value(d1, train_y):
    d1 = d1.astype({'X': 'float', 'Y': 'int', "Bucket": "str"})
    d2 = d1.groupby('Bucket', as_index=True)
    good = train_y.sum()
    bad = train_y.count() - good
    d3 = pd.DataFrame(d2.Y.min(), columns=['min'])
    d3['min'] = d2.min().X
    d3['max'] = d2.max().X
    d3['sum'] = d2.sum().Y
    d3['total'] = d2.count().Y
    d3['rate'] = d2.mean().Y
    d3['woe'] = np.log((d3['rate'] / good) / ((1 - d3['rate']) / bad))
    d3['goodattribute'] = d3['sum'] / good
    d3['badattribute'] = (d3['total'] - d3['sum']) / bad
    iv = ((d3['goodattribute'] - d3['badattribute']) * d3['woe']).sum()
    d4 = (d3.sort_index(by='min')).reset_index(drop=True)
    woe = list(d4['woe'].round(3))
    return d4, iv, woe


def interval_cut(x, y, train_X_feature, train_y):
    zoom_x_y = create_interval(x, y)
    d3_x = feature_cut(train_X_feature, train_y, zoom_x_y)
    return d3_x


if __name__ == "__main__":
    train_data = pd.read_csv("./preprocession_data/train_data.csv", index_col='Unnamed: 0')
    X = train_data.iloc[:, 1:]
    y = train_data.iloc[:, 0]

    train_X, _, train_y, _ = train_test_split(X, y, test_size=0.3, random_state=0)

    d3_x1 = interval_cut(float('-inf'), 0, train_X['NumberOfTime30-59DaysPastDueNotWorse'], train_y)
    d3_x2 = interval_cut(0, 1, train_X['NumberOfTime30-59DaysPastDueNotWorse'], train_y)
    d3_x3 = interval_cut(1, 3, train_X['NumberOfTime30-59DaysPastDueNotWorse'], train_y)
    d3_x4 = interval_cut(3, 5, train_X['NumberOfTime30-59DaysPastDueNotWorse'], train_y)
    d3_x5 = interval_cut(5, float('inf'), train_X['NumberOfTime30-59DaysPastDueNotWorse'], train_y)
    d3 = pd.concat([d3_x1, d3_x2, d3_x3, d3_x4, d3_x5])  # print(d3)

    x3_d, x3_iv, x3_woe = woe_value(d3, train_y)
    x3_cut = [float('-inf'), 0, 1, 3, 5, float('+inf')]
    x3 = [x3_iv, x3_woe, x3_cut]

    print(train_X['NumberOfOpenCreditLinesAndLoans'])
    print(train_y)
    d6_x1 = interval_cut(float('-inf'), 0, train_X['NumberOfOpenCreditLinesAndLoans'], train_y)
    d6_x2 = interval_cut(0, 1, train_X['NumberOfOpenCreditLinesAndLoans'], train_y)
    d6_x3 = interval_cut(1, 3, train_X['NumberOfOpenCreditLinesAndLoans'], train_y)
    d6_x4 = interval_cut(3, 5, train_X['NumberOfOpenCreditLinesAndLoans'], train_y)
    d6_x5 = interval_cut(5, float('inf'), train_X['NumberOfOpenCreditLinesAndLoans'], train_y)
    d6 = pd.concat([d6_x1, d6_x2, d6_x3, d6_x4, d6_x5]);  # print(d6)

    x6_d, x6_iv, x6_woe = woe_value(d6, train_y)
    x6_cut = [float('-inf'), 0, 1, 3, 5, float('+inf')]
    x6 = [x6_iv, x6_woe, x6_cut]

    d7_x1 = interval_cut(float('-inf'), 0, train_X['NumberOfTimes90DaysLate'], train_y)
    d7_x2 = interval_cut(0, 1, train_X['NumberOfTimes90DaysLate'], train_y)
    d7_x3 = interval_cut(1, 3, train_X['NumberOfTimes90DaysLate'], train_y)
    d7_x4 = interval_cut(3, 5, train_X['NumberOfTimes90DaysLate'], train_y)
    d7_x5 = interval_cut(5, float('inf'), train_X['NumberOfTimes90DaysLate'], train_y)
    d7 = pd.concat([d7_x1, d7_x2, d7_x3, d7_x4, d7_x5]);  # print(d6)

    x7_d, x7_iv, x7_woe = woe_value(d7, train_y)
    x7_cut = [float('-inf'), 0, 1, 3, 5, float('+inf')]
    x7 = [x7_iv, x7_woe, x7_cut]


    d8_x1 = interval_cut(float('-inf'), 0, train_X['NumberRealEstateLoansOrLines'], train_y)
    d8_x2 = interval_cut(0, 1, train_X['NumberRealEstateLoansOrLines'], train_y)
    d8_x3 = interval_cut(1, 2, train_X['NumberRealEstateLoansOrLines'], train_y)
    d8_x4 = interval_cut(2, 3, train_X['NumberRealEstateLoansOrLines'], train_y)
    d8_x5 = interval_cut(3, float('inf'), train_X['NumberRealEstateLoansOrLines'], train_y)
    d8 = pd.concat([d8_x1, d8_x2, d8_x3, d8_x4, d8_x5]);  # print(d6)

    x8_d, x8_iv, x8_woe = woe_value(d8, train_y)
    x8_cut = [float('-inf'), 0, 1, 2, 3, float('+inf')]
    x8 = [x8_iv, x8_woe, x8_cut]

    d9_x1 = interval_cut(float('-inf'), 0, train_X['NumberOfTime60-89DaysPastDueNotWorse'], train_y)
    d9_x2 = interval_cut(0, 1, train_X['NumberOfTime60-89DaysPastDueNotWorse'], train_y)
    d9_x3 = interval_cut(1, 3, train_X['NumberOfTime60-89DaysPastDueNotWorse'], train_y)
    d9_x4 = interval_cut(3, float('inf'), train_X['NumberOfTime60-89DaysPastDueNotWorse'], train_y)
    d9 = pd.concat([d9_x1, d9_x2, d9_x3, d9_x4])  # print(d6)

    x9_d, x9_iv, x9_woe = woe_value(d9, train_y)
    x9_cut = [float('-inf'), 0, 1, 3, float('+inf')]
    x9 = [x9_iv, x9_woe, x9_cut]
    print(x9)

    d10_x1 = interval_cut(float('-inf'), 0, train_X['NumberOfDependents'], train_y)
    d10_x2 = interval_cut(0, 1, train_X['NumberOfDependents'], train_y)
    d10_x3 = interval_cut(1, 2, train_X['NumberOfDependents'], train_y)
    d10_x4 = interval_cut(2, 3, train_X['NumberOfDependents'], train_y)
    d10_x5 = interval_cut(3, 5, train_X['NumberOfDependents'], train_y)
    d10_x6 = interval_cut(5, float('inf'), train_X['NumberOfDependents'], train_y)
    d10 = pd.concat([d10_x1, d10_x2, d10_x3, d10_x4, d10_x5, d10_x6]);  # print(d6)

    x10_d, x10_iv, x10_woe = woe_value(d10, train_y)
    x10_cut = [float('-inf'), 0, 1, 2, 3, 5, float('+inf')]
    x10 = [x10_iv, x10_woe, x10_cut]

    informationValue = ['NumberOfTime30-59DaysPastDueNotWorse', 'NumberOfOpenCreditLinesAndLoans',
                        'NumberOfTimes90DaysLate', 'NumberRealEstateLoansOrLines',
                        'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents'
                        ]
    infor = [x3, x6, x7, x8, x9, x10]
    IV_table2 = pd.DataFrame(infor, index=informationValue, columns=["IV", "WOE", "cut_Interval"])

    f = open('preprocession_data/IV_table2.p', "wb")
    pickle.dump(IV_table2, f)
