"""


"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# 1、获取数据集
# 加载模块
iris = load_iris()
# 2、划分数据集
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=22)

# 3.特征工程:标准化
# 实例化一个转换器类
transfer = StandardScaler()

# 调用fit_transform
x_train = transfer.fit_transform(x_train)
x_test = transfer.fit_transform(x_test)


# 4.实例化API
# 4、KNN预估器流程
#     1）实例化预估器类
estimator = KNeighborsClassifier()


# 5.模型选择与调优--网络搜索和交叉验证
# 准备要调的超参数
param_dict={"n_neighbors":[1,3,5]}

estimator = GridSearchCV(estimator,param_grid=param_dict,cv=3)

#  2)fit数据进行训练
estimator.fit(x_train, y_train)

# 5.模型评估
# 方法1:比对真实值和预测值
y_predict=estimator.predict(x_test)
print("预测结果为:\n",y_predict)
print("比对真实值和预测值:\n",y_predict==y_test)

# 方法2:直接计算准确率
score=estimator.score(x_test,y_test)
print("准确率为:\n",score)

# 评估查看最终选择的结果和交叉验证的结果

print("在交叉验证中的最好结果:\n",estimator.best_score_)
print("最好的参数模型:\n", estimator.best_estimator_)
print("每次交叉验证后的准确率结果:\n",estimator.cv_results_)

