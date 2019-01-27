"""


"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error,classification_report


def logisticregression():
    """
    逻辑回归进行癌症预测
    :return: None
    """

    # 1.读取数据,处理缺失值以及标准化
    column_name = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
                   'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
                   'Normal Nucleoli', 'Mitoses', 'Class']
    data = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data",
        names=column_name)

    # 处理缺失值
    data = data.replace(to_replace='?', value=np.nan)
    data = data.dropna()

    # 取出特征值
    x = data[column_name[1:10]]
    y = data[column_name[10]]

    # 数据集划分
    x_train, x_test, y_train, y_test = train_test_split(x, y)

    # 数据标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)

    x_test = transfer.transform(x_test)

    # 模型生成,训练数据
    estimator = LogisticRegression()
    estimator.fit(x_train, y_train)
    y_predict = estimator.predict(x_test)
    # 评估模型
    print("逻辑回归分类模型预测结果:\n", y_predict)

    print("逻辑回归分类模型回归系数:\n", estimator.coef_)
    print("逻辑回归分类模型偏置:\n", estimator.intercept_)
    print("逻辑回归分类模型准确率:\n", estimator.score(x_test,y_test))

    # 模型均方误差
    error = mean_squared_error(y_test, y_predict)
    print("逻辑回归分类模型的均方误差:\n", error)

    # 获取分类报告
    report = classification_report(y_test,y_predict,labels=[2,4],target_names=["良性","恶性"])
    print(report)

if __name__ == '__main__':
    logisticregression()
