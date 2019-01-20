# 验证集测试
import copy
import statsmodels.api as sm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

name = pickle.load(open('./preprocession_data/argument.p', 'rb'))
iv1 = pickle.load(open('./preprocession_data/IV_table1.p', "rb"))
iv2 = pickle.load(open('./preprocession_data/IV_table2.p', "rb"))

IV = pd.concat([iv1, iv2])
val = IV.loc[name, :]
val.reset_index(inplace=True)

fea_name = list(val["index"])
woe = list(val.WOE)
cut = list(val.cut_Interval)

from sklearn.model_selection import train_test_split

train_data = pd.read_csv('./preprocession_data/train_data.csv', index_col='Unnamed: 0')
X = train_data.iloc[:, 1:]
y = train_data.iloc[:, 0]

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=0)
print(train_data.columns)

test_X = pd.read_csv('./preprocession_data/test_X_woe.csv', index_col='Unnamed: 0')
train_X = pd.read_csv('./preprocession_data/train_X_woe.csv', index_col='Unnamed: 0')

# print(train_X.shape)
# print(train_X.columns)
# print('--------------------------')
# print(train_y.columns)

# 使用statsmodels.api进行建模分析需要给模型增加截距
X1 = sm.add_constant(train_X)

logit = sm.Logit(train_y, X1)
result = logit.fit()  # 训练

# 在results上调用summary方法，可能得到一些详细的诊断数据
print(result.summary())

from sklearn.metrics import roc_curve, auc

X3 = sm.add_constant(test_X)
resu = result.predict(X3)
# print(resu)

fpr, tpr, threshold = roc_curve(test_y, resu)

roc_auc = auc(fpr, tpr)
print('auc值: ', roc_auc)
plt.figure(figsize=[10, 8])
plt.plot(fpr, tpr, 'b', label='AUC=%0.2f' % roc_auc)
plt.legend()
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('TPR')
plt.xlabel('FPR')
plt.show()

# 建立评分卡
p = 20 / np.log(2)
q = 600 - 20 * np.log(20) / np.log(2)

# x_coe 是利用逻辑回归拟合后的系数矩阵
x_coe = [2.4908, 0.6603, 0.2043, 0.5229, 0.5984, 0.5049]
base_score = round(q + p * x_coe[0], 0)

print('p: ', p)
print('q: ', q)
print('base_score', base_score)


def get_score(coe, woe, factor):
    print('wwwwwwwwwww', woe)
    scores = []
    for w in woe:
        # 计算每一个属性的分值
        print(coe, w, factor)
        score = round(coe * w * factor, 0)
        scores.append(score)
    return scores


# WOE值
# [[-1.283, -1.193, -0.298, 1.094],
# [0.556, 0.391, 0.265, 0.199, 0.096, -0.202, -0.491, -0.935, -1.123],
# [-0.526, 0.901, 1.734, 2.363, 2.679],
# [-0.379, 1.967, 2.75, 3.207, 3.104],
# [-0.274, 1.835, 2.685, 2.907]]
print('woe: ', woe)

# Score值
# [[-38.0, 64.0, 126.0, 175.0, 186.0],
#  [-7.0, 37.0, 52.0, 61.0, 61.0],
#  [-2.0, 11.0, 16.0, 17.0],
#  [-20.0, -19.0, -4.0, 17.0],
#  [10.0, 6.0, 5.0, 3.0, 2.0, -4.0, -8.0, -16.0, -19.0]]

score = []
for i in range(len(woe)):
    x_score = get_score(x_coe[i+1], woe[i], p)
    score.append(x_score)

# [[-92.0, -86.0, -21.0, 79.0],
#  [11.0, 7.0, 5.0, 4.0, 2.0, -4.0, -9.0, -18.0, -21.0],
#  [-3.0, 5.0, 10.0, 14.0, 16.0],
#  [-6.0, 30.0, 41.0, 48.0, 47.0],
#  [-5.0, 32.0, 46.0, 50.0]]

print('score: ', score)
cut_t = cut

# score 值
# [[-24.0, -23.0, -6.0, 21.0],
#  [3.0, 2.0, 2.0, 1.0, 1.0, -1.0, -3.0, -6.0, -7.0],
#  [-8.0, 14.0, 26.0, 36.0, 40.0],
#  [-7.0, 34.0, 47.0, 55.0, 54.0],
#  [-4.0, 27.0, 39.0, 42.0]]

# 建立一个函数使得当输入x1, x2, x3, x7, x9的值得时候可以返回分数
print('-----------------------********************-------------------------')


def compute_score(x):  # x就是数组, 包含x1, x2, x3, x7, x9
    tot_score = base_score
    cut_d = copy.deepcopy(cut_t)
    print('cut_d: ', cut_d)
    for j in range(len(cut_d)):
        cut_d[j].append(x[j])
        cut_d[j].sort()
        for i in range(len(cut_d[j])):
            if cut_d[j][i] == x[j]:
                # print(score[j][i-1])
                tot_score = score[j][i-1] + tot_score
    return tot_score


final_score = compute_score(x=[0.3, 44, 3, 3, 5])
print(final_score)

# cut
# [[-inf, 0.0312, 0.1579, 0.5581, inf],
# # [-inf, 33.0, 40.0, 45.0, 49.0, 54.0, 59.0, 64.0, 71.0, inf],
# # [-inf, 0, 1, 3, 5, inf],
# # [-inf, 0, 1, 3, 5, inf],
# # [-inf, 0, 1, 3, inf]]
