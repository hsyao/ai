"""


"""
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split  # 数据集划分
from sklearn.feature_extraction.text import TfidfVectorizer  # 特征提取:tfidf重要性
from sklearn.naive_bayes import MultinomialNB  # 贝叶斯分类训练器

# 1.获取新闻的数据,20个类别

news = fetch_20newsgroups(data_home='J:\datasets', subset='all')

# 2.数据集划分

x_train, x_test, y_train, y_test = train_test_split(news.data, news.target)

# 3.特征提取
# TFIDF文本特征抽取
tf = TfidfVectorizer()

x_train = tf.fit_transform(x_train)
# 这里打印出来的列表是：训练集当中的所有不同词的组成的一个列表
print(tf.get_feature_names())

# 不能调用fit_transform
x_test= tf.transform(x_test)

# 4.生成贝叶斯评估器
estimator = MultinomialNB()

# 5.训练数据
estimator.fit(x_train, y_train)

# 获取预测结果
y_predict = estimator.predict(x_test)

# 6.评估模型
print("预测值与真实值比对:\n", y_predict == y_test)

print("预测每篇文章的类别:\n", y_predict[:100])
print("真实类别为:\n", y_test[:100])
score = estimator.score(x_test, y_test)
print("预测结果准确度:\n", score)
