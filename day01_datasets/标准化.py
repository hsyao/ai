"""


"""
from sklearn.preprocessing import StandardScaler
import pandas as pd


def stand_demo():
    """
    标准化演示
    :return: None
    """

    data = pd.read_csv("dating.txt")
    print(data)

    # 1.实例化一个准换器类
    transfer = StandardScaler()

    # 2.调用fit_transform
    data = transfer.fit_transform(data[['milage', 'Liters', 'Consumtime']])

    print("标准化的结果:\n", data)

    print("每一列特征的平均值:\n", transfer.mean_)

    print("每一列特征的方差:\n",transfer.var_)

    return None

stand_demo()
