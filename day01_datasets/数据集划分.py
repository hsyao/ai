"""


"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def datasets_demo():
    """
    对莺尾花数据集的演示
    :return:
    """

    # 1. 获取莺尾花数据集
    iris = load_iris()
    print("莺尾花数据集的返回值:\n", iris)
    # 返回值是一个继承自字典的Bunch
    # print("莺尾花的特征值:\n",iris.data)
    print("莺尾花的特征值:\n", iris["data"])

    print("莺尾花的目标值:\n", iris.target)

    print('莺尾花特征的名字:\n', iris.feature_names)

    print("莺尾花目标值的名字:\n", iris.target_names)

    print("莺尾花的描述:\n", iris.DESCR)

    # 2. 对莺尾花数据集进行分割
    # 训练集的特征值x_train 测试集的特征值x_test 训练集的目标值y_train 测试集的目标值y_test
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=22)

    # 随机数种子
    x_train1, x_test1, y_train1, y_test1 = train_test_split(iris.data, iris.target, random_state=6)
    x_train2, x_test2, y_train2, y_test2 = train_test_split(iris.data, iris.target, random_state=6)

    print("如果随机数种子不一致:\n", x_train == x_train1)
    print("如果随机数种子一致:\n", x_train1 == x_train2)

    return None


datasets_demo()
