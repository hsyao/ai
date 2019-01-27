"""


"""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz

# 1)获取数据集
iris = load_iris()

# 2)划分数据集
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target)

# 3)决策树预估器
dt = DecisionTreeClassifier(criterion="entropy")

# 训练决策树分类器
dt.fit(x_train, y_train)

y_predict = dt.predict(x_test)
# 4)模型评估
print("预测的特征名称为:\n", iris.feature_names)
print("预测的目标名称:\n", iris.target_names)
print("预测结果:\n", y_predict)
print("真实值对比预测值:\n", y_test == y_predict)

score = dt.score(x_test, y_test)
print("预测准确度:", score)
export_graphviz(dt, out_file='iris_tree.dot',feature_names=iris.feature_names)

