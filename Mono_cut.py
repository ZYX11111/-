import numpy as np
import pandas as pd
from scipy import stats
import warnings
import pickle

warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split


# 单调分箱
# Y 中 1表示好样本
# X为一列为实数变量，第二列Y 为取值为0,1的因变量
# n表示初始化分段数量
def mono_bin(Y, X, n=10):  # Y7267
    r = 0
    good = Y.sum()  # good 6813
    bad = Y.count() - good  # bad 94934
    while np.abs(r) < 1:  # r的绝对值
        # pd.qcut(data['feature'],n):data['feature']:分箱所使用的属性值，n：分箱的个数，
        # 例子：等量分箱，将1000个数据进行分箱，分为10个箱子，每个箱子里要包含100个样本
        # 按照X来分箱  x:是信用额度的余额 / 总额度     n是箱子个数
        # qcut是根据这些值的频率来选择箱子的均匀间隔。
        d1 = pd.DataFrame({"X": X, "Y": Y, "Bucket": pd.qcut(X, n)})
        # 按“Bucket”的方法进行分组

        d2 = d1.groupby('Bucket', as_index=True)
        # print("====",d2.mean().X)
        r, p = stats.spearmanr(d2.mean().X, d2.mean().Y)  # r是相关性
        n = n - 1
    # print("======",d2.min(),"======")

    d3 = pd.DataFrame(d2.X.min(), columns=['min'])

    # print("------",d2.min().X)
    d3['min'] = d2.min().X
    # 每个箱子都包含3个数据X，Y，Butter，先找每个箱子里的三个属性的最小值，再找某个属性的最小值
    d3['max'] = d2.max().X
    # 每个箱子都包含3个数据X，Y，Butter，先找每个箱子里的三个属性的最大值，再找某个属性的最大值
    # print("=====sum",d2.sum(),"======")
    d3['sum'] = d2.sum().Y
    # 每个箱子都包含3个数据X，Y，Butter，先找每个箱子里的三个属性的和，再找某个属性的和
    # print("=====count",d2.count(),"=======")
    d3['total'] = d2.count().Y
    # 每个箱子都包含3个数据X，Y，Butter，先找每个箱子里的三个属性的平均值，再找某个属性的平均值
    d3['rate'] = d2.mean().Y
    # 计算woe值
    d3['woe'] = np.log((d3['rate'] / good) / ((1 - d3['rate']) / bad))
    # 这个箱子中好样本占整体好样本的比例
    d3['goodattribute'] = d3['sum'] / good
    # 这个箱子中坏样本占整体坏样本的比例
    d3['badattribute'] = (d3['total'] - d3['sum']) / bad
    # 计算iv
    iv = ((d3['goodattribute'] - d3['badattribute']) * d3['woe']).sum()
    # 排序
    d4 = (d3.sort_index(by='min')).reset_index(drop=True)
    # 计算woe
    woe = list(d4['woe'].round(3))
    # 计算分位数
    cut = []
    cut.append(float('-inf'))
    for i in range(1, n + 1):
        qua = X.quantile(i / (n + 1))  # 分位数
        cut.append(round(qua, 4))
    cut.append(float('inf'))
    return d4, iv, cut, woe


# cut 是四分位数

# 对几个可分箱的属性变量进行分箱操作
if __name__ == '__main__':
    train_data = pd.read_csv("./preprocession_data/train_data.csv", index_col='Unnamed: 0')
    X = train_data.iloc[:, 1:]
    y = train_data.iloc[:, 0]
    train_X, _, train_y, _ = train_test_split(X, y, test_size=0.3, random_state=0)

    # RevolvingUtilizationOfUnsecuredLines:信用额度的总余额 / 信用总额度
    x1_d, x1_iv, x1_cut, x1_woe = mono_bin(train_y, train_X.RevolvingUtilizationOfUnsecuredLines)
    # age：年龄
    x2_d, x2_iv, x2_cut, x2_woe = mono_bin(train_y, train_X.age)
    # DebtRatio支出 / 收入
    x4_d, x4_iv, x4_cut, x4_woe = mono_bin(train_y, train_X.DebtRatio)
    # MonthlyIncome：月收入
    x5_d, x5_iv, x5_cut, x5_woe = mono_bin(train_y, train_X.MonthlyIncome)

    list_x1 = [x1_iv, x1_woe, x1_cut]
    list_x2 = [x2_iv, x2_woe, x2_cut]
    list_x3 = [x4_iv, x4_woe, x4_cut]
    list_x5 = [x5_iv, x5_woe, x5_cut]

    informationValue = ['RevolvingUtilizationOfUnsecuredLines', 'age',
                        'DebtRatio', 'MonthlyIncome']
    infor = [list_x1, list_x2, list_x3, list_x5]
    IV_table1 = pd.DataFrame(infor, index=informationValue, columns=["IV", "WOE", "cut_Interval"])

    # IV_table1.to_csv("./preprocession_data/Iv_table2.csv")
    f = open('./preprocession_data/IV_table1.p', "wb")
    pickle.dump(IV_table1, f)