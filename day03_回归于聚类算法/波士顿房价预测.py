"""


"""
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge, RidgeCV
from sklearn.metrics import mean_squared_error


def linear1():
    """
    用梯度下街优化模型参数的方法进行波士顿房价预测的线性回归案例
    :return: None
    """
    # 1.获取数据集
    boston = load_boston()
    # print("boston:\n", boston.DESCR)

    # 2.划分数据集
    x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target,random_state=22)

    # 特征工程,标准化
    # 实例化一个转换器类
    transfer = StandardScaler()
    # 3.数据标准化
    x_train = transfer.fit_transform(x_train)

    x_test = transfer.transform(x_test)

    # 线性回归的预估器流程
    # 4.训练数据集
    lr = LinearRegression()

    lr.fit(x_train, y_train)
    y_predict = lr.predict(x_test)
    # 5.评估模型
    # print("预测结果:\n",y_predict)

    # print("预测结果与真实值对比:\n",y_test==lr.predict(x_test))
    # print("预测结果的准确性:\n",lr.score(x_test,y_test))

    # 5.得出模型
    print("正规方程求出模型参数的方法预测的房屋价格为:\n", y_predict)
    print("正规方程的回归系数为:\n", lr.coef_)
    print("正规方程的偏置为:\n", lr.intercept_)

    # 6.评估模型-均方误差
    error = mean_squared_error(y_test, y_predict)
    print("正规方程的均方误差为:\n", error)
    return None


def linear2():
    """
    用梯度下降直接求出模型参数的方法进行对波士顿房价预测的线性回归案例
    :return: None
    """
    # 1.获取数据集
    boston = load_boston()
    # print("boston:\n", boston.DESCR)

    # 2.划分数据集
    x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target,random_state=22)

    # 特征工程,标准化
    # 实例化一个转换器类
    transfer = StandardScaler()
    # 3.数据标准化
    x_train = transfer.fit_transform(x_train)

    x_test = transfer.transform(x_test)

    # 线性回归的预估器流程
    # 4.训练数据集
    # sr = SGDRegressor()
    # 加入学习系数
    sr = SGDRegressor(learning_rate="constant",eta0=0.001)

    sr.fit(x_train, y_train)
    y_predict = sr.predict(x_test)
    # 5.评估模型
    # print("预测结果:\n",y_predict)

    # print("预测结果与真实值对比:\n",y_test==lr.predict(x_test))
    print("预测结果的准确性:\n",sr.score(x_test,y_test))

    # 5.得出模型
    print("梯度下降求出模型参数的方法预测的房屋价格为:\n", y_predict)
    print("梯度下降的回归系数为:\n", sr.coef_)
    print("梯度下降的偏置为:\n", sr.intercept_)

    # 6.评估模型-均方误差
    error = mean_squared_error(y_test, y_predict)
    print("梯度下降的均方误差为:\n", error)
    return None


def linear3():
    """
    用岭回归的方法对波士顿房价预测
    :return: None
    """
    # 1.获取数据集
    boston = load_boston()
    # print("boston:\n", boston.DESCR)

    # 2.划分数据集
    x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target,random_state=22)

    # 特征工程,标准化
    # 实例化一个转换器类
    transfer = StandardScaler()
    # 3.数据标准化
    x_train = transfer.fit_transform(x_train)

    x_test = transfer.transform(x_test)

    # 线性回归的预估器流程
    # 4.训练数据集
    estimator = Ridge()

    estimator.fit(x_train, y_train)
    y_predict = estimator.predict(x_test)
    # 5.评估模型
    print("预测结果的准确性:\n",estimator.score(x_test,y_test))

    # 5.得出模型
    print("岭回归求出模型参数的方法预测的房屋价格为:\n", y_predict)
    print("岭回归的回归系数为:\n", estimator.coef_)
    print("岭回归的偏置为:\n", estimator.intercept_)

    # 6.评估模型-均方误差
    error = mean_squared_error(y_test, y_predict)
    print("岭回归的均方误差为:\n", error)
    return None


def linear4():
    """
    用岭回归的方法对波士顿房价预测
    :return: None
    """
    # 1.获取数据集
    boston = load_boston()
    # print("boston:\n", boston.DESCR)

    # 2.划分数据集
    x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target,random_state=22)

    # 特征工程,标准化
    # 实例化一个转换器类
    transfer = StandardScaler()
    # 3.数据标准化
    x_train = transfer.fit_transform(x_train)

    x_test = transfer.transform(x_test)

    # 线性回归的预估器流程
    # 4.训练数据集  加入交叉验证
    estimator = RidgeCV()

    estimator.fit(x_train, y_train)
    y_predict = estimator.predict(x_test)
    # 5.评估模型
    print("预测结果的准确性:\n",estimator.score(x_test,y_test))

    # 5.得出模型
    print("岭回归求出模型参数的方法预测的房屋价格为:\n", y_predict)
    print("岭回归的回归系数为:\n", estimator.coef_)
    print("岭回归的偏置为:\n", estimator.intercept_)

    # 6.评估模型-均方误差
    error = mean_squared_error(y_test, y_predict)
    print("岭回归交叉验证后的均方误差为:\n", error)
    return None


if __name__ == '__main__':
    # 正规方程
    linear1()
    # 梯度下降
    linear2()
    # 岭回归
    linear3()
    # 交叉验证岭回归
    linear4()
