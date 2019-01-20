import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle


def tu(train_data, IV, c_name):
    corr = train_data.corr()
    # corr.to_csv('corr.csv')
    xticks = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10']
    yticks = list(corr.index)
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    sns.heatmap(corr, annot=True, cmap='RdBu', ax=ax1, annot_kws={'size': 5, 'color': 'blue'})
    ax1.set_xticklabels(xticks, rotation=0, fontsize=10)
    ax1.set_yticklabels(yticks, rotation=0, fontsize=10)
    plt.show()

    plt.figure(figsize=(10, 8))
    index_num = range(len(c_name))
    ax = plt.bar(index_num, IV, tick_label=c_name)
    plt.xticks(rotation=45)
    plt.show()

    save_col = []
    for i in range(len(IV)):
        if IV[i] < 0.2:
            save_col.append(i)

    save_col_name = []
    for i in range(len(c_name)):
        if i not in save_col:
            save_col_name.append(c_name[i])
    # print(save_col_name)
    return save_col_name


if __name__ == '__main__':
    train_data = pd.read_csv('./preprocession_data/train_data.csv', index_col='Unnamed: 0')

    iv1 = pickle.load(open('./preprocession_data/IV_table1.p', "rb"))
    iv2 = pickle.load(open('./preprocession_data/IV_table2.p', "rb"))

    IV = pd.concat([iv1, iv2])
    IV.reset_index(inplace=True)
    print(IV.columns)
    c_name = IV['index']

    IV = list(IV.IV)
    c_name = list(c_name)

    save_feature_name = tu(train_data, IV, c_name)
    f = open('preprocession_data/argument.p', 'wb')
    pickle.dump(save_feature_name, f)

# 要抛弃的属性
# ['NumberOfOpenCreditLinesAndLoans',
#  'NumberRealEstateLoansOrLines',
#  'NumberOfDependents',
#  'DebtRatio',
#  'MonthlyIncome']
