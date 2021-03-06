"""


"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# 加载模块
iris = load_iris()

x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=22)

# 3.特征工程:标准化
transfer = StandardScaler()

x_train = transfer.fit_transform(x_train)
x_test = transfer.fit_transform(x_test)


# 4.实例化API
estimator = KNeighborsClassifier(n_neighbors=9)
estimator.fit(x_train, y_train)

# 5.模型评估
# 方法1:比对真实值和预测值
y_predict=estimator.predict(x_test)
print("预测结果为:\n",y_predict)
print("比对真实值和预测值:\n",y_predict==y_test)

# 方法2:直接计算准确率
score=estimator.score(x_test,y_test)
print("准确率为:\n",score)