"""


"""
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def minmax_demo():
    """
    归一化演示
    :return: None
    """

    data = pd.read_csv("dating.txt")
    print(data)

    # 1. 实例化一个转换器类
    transfer = MinMaxScaler(feature_range=(2, 3))

    # 2. 调用fit_transform

    data = transfer.fit_transform(data[['milage', 'Liters', 'Consumtime']])

    print('最小值最大值归一化处理的结果:\n', data)

    return None


minmax_demo()



