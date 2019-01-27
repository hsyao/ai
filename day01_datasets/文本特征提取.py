"""


"""

from sklearn.feature_extraction.text import CountVectorizer

def text_count_demo():
    """
    对文本进行特征提取,countvetorizer
    :return: None
    """

    data=["life is short,i like like python", "life is too long,i dislike python"]

    data=["人生苦短，我喜欢Python" "生活太长久，我不喜欢Python"]
    # 1、实例化一个转换器类
    # transfer = CountVectorizer(sparse=False)
    transfer = CountVectorizer()

    # 2. 调用fit_transform
    data = transfer.fit_transform(data)
    print("文本特征提取的结果:\n", data.toarray())

    print("返回文本特征的名字:\n", transfer.get_feature_names())

    return None

text_count_demo()