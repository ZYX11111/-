import matplotlib.pyplot as plt
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.metrics import roc_curve, auc
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

train_data = pd.read_csv('./preprocession_data/train_data.csv', index_col='Unnamed: 0')

X = train_data.iloc[:, 1:]
y = train_data.iloc[:, 0]

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=0)


# woe值的计算
def trans_woe(var, var_name, x_woe, x_cut):
    print(var_name)
    woe_name = var_name + '_woe'
    # print(var[var_name])
    for i in range(len(x_woe)):
        # print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&",len(x_woe))
        if i == 0:
            var.loc[(var[var_name] <= x_cut[i + 1]), woe_name] = x_woe[i]
            print("&&*********************************")
            print(var.loc[(var[var_name] <= x_cut[i + 1]), woe_name])
            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            # print("===================",x_cut[i+1])

        elif (i > 0) and (i <= len(x_woe) - 2):
            var.loc[((var[var_name] > x_cut[i]) & (var[var_name] <= x_cut[i + 1])), woe_name] = x_woe[i]
            print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
            print(var.loc[(var[var_name] > x_cut[i]), woe_name])
            print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            # print(var[var_name]<=x_cut[i+1])
            # print(woe_name)
        else:
            var.loc[(var[var_name] > x_cut[len(x_woe) - 1]), woe_name] = x_woe[len(x_woe) - 1]
    return var


# 建立模型
def logi(train_X, train_y, test_X, test_y):
    # 对训练集结果计算AUC ROC KS值
    # 方法1： 直接用lr预测数值  -- 预测结果0.85
    lr = LogisticRegression()
    lr.fit(train_X, train_y)

    # 注意predict 和predict_proba的区别
    resu = lr.predict_proba(test_X)
    resuLabel = lr.predict(test_X)

    print('predict_proba:', resu)
    print('-------')
    print('predict:', resuLabel)

    # X3=sm.add_constant(test_X)
    # resu=result.predict(X3)
    print("+++++++++++++++++++++++++++++", resu[:, 1])
    fpr, tpr, thershold = roc_curve(test_y, resu[:, 1])
    print("发票融资", fpr)
    print("发的南京", tpr)
    rocauc = auc(fpr, tpr)
    plt.plot(fpr, tpr, 'b', label='AUC=%0.2f' % rocauc)
    plt.legend()
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    plt.title('逻辑回归预测')
    plt.show()

    print('KS:', max(tpr - fpr))


if __name__ == '__main__':
    name = pickle.load(open('preprocession_data/argument.p', 'rb'))

    iv1 = pickle.load(open('preprocession_data/IV_table1.p', "rb"))
    iv2 = pickle.load(open('preprocession_data/IV_table2.p', "rb"))

    IV = pd.concat([iv1, iv2])
    val = IV.loc[name, :]
    val.reset_index(inplace=True)

    print(val.columns)
    fea_name = list(val["index"])
    woe = list(val.WOE)
    cut = list(val.cut_Interval)

    for i in range(len(fea_name)):
        train_X = trans_woe(train_X, fea_name[i], woe[i], cut[i])
    for i in range(len(fea_name)):
        test_X = trans_woe(test_X, fea_name[i], woe[i], cut[i])

    test_X.to_csv('preprocession_data/test_X_woe.csv')
    train_X.to_csv('preprocession_data/train_X_woe.csv')

    logi(train_X, train_y, test_X, test_y)

# IV['IV'].astype(np.float64)
# for index, value in enumerate(IV['WOE']):
#     IV['WOE'][index] = np.array(value[1:-1].split(',')).astype(np.float64)
# for index, value in enumerate(iv2['cut_Interval']):
#     IV['cut_Interval'][index] = np.array(value[1:-1].split(',')).astype(np.float64)
#
# print(IV['IV'])
# print(IV['WOE'])
# print(IV['cut_Interval'])
