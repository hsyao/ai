"""
注意问题：警告指出你的CPU支持AVX运算加速了线性代数计算，即点积，矩阵乘法，卷积等。可以从源代码安装TensorFlow来编译，当然也可以选择关闭

"""
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def tensorflow_demo():
    """
    通过剪刀案例来了解tensorflow的基础结构
    :return: None
    """

    # 一. 原生Python实现加法运算
    a = 10
    b = 20
    c = a + b
    print("原生Python实现加法运算方法1:\n", c)

    def add(a, b):
        return a + b

    sum = add(a, b)
    print("原生Python实现加法运算方法2:\n", sum)

    # 二. tensorflow实现加法运算
    a_t = tf.constant(10)
    b_t = tf.constant(20)
    # 不提倡直接运用这种符号运算符进行计算
    # 更常用tensorflow提供的函数进行计算
    # c_t= a_t+b_t
    c_t = tf.add(a_t, b_t)
    print("tensorflow实现加法运算:\n", c_t)

    # 如何让计算结果出现?
    # 开启绘画

    with tf.Session() as sess:
        sum_t = sess.run(c_t)
        print("在sess当中的sum_t:\n", sum_t)

    return None


def graph_demo():
    # 图的演示
    a_t = tf.constant(10)
    b_t = tf.constant(20)
    # 不提倡直接运用这种符号运算符进行计算
    # 更常用tensorflow提供的函数进行计算
    # c_t=a_t+b_t

    c_t = tf.add(a_t, b_t)
    print("tensorflow实现加法运算:\n", c_t)

    # 获取默认图
    default_g = tf.get_default_graph()
    print("获取默认图:\n", default_g)

    # 数据的图属性
    print("a_t的graph:\n", a_t.graph)
    print("b_t的graph:\n", b_t.graph)

    # 操作的图属性
    print("c_t的graph:\n", c_t.graph)

    # 自定义图
    new_g = tf.Graph()
    print("自定义图:\n", new_g)
    # 在自定义图中去定义数据和操作
    with new_g.as_default():
        new_a = tf.constant(30)
        new_b = tf.constant(40)
        new_c = tf.add(new_a, new_b)

    # 数据的图属性
    print("new_a的graph:\n", new_a.graph)
    print("new_b的graph:\n", new_b.graph)

    # 操作的图属性
    print("new_c的graph:\n", new_c.graph)

    # 开启会话
    with tf.Session() as sess:
        sum_t = sess.run(c_t)
        print("在sess当中的sum_t:\n", sum_t)

        # 会话的图属性
        print("会话的图属性:\n", sess.graph)

    with tf.Session(graph=new_g) as sess2:
        sum_new = sess2.run(new_c)
        print("在sess2当中的sum_new:\n", sum_new)
        print("会话图的属性:\n", sess2.graph)

    tf.summary.FileWriter("",sess.graph)

    return None


def tesorboard_demo():
    # 图的演示
    a_t = tf.constant(10,name='a')
    b_t = tf.constant(20, name='b')
    # 不提倡直接运用这种符号运算符进行计算
    # 更常用tensorflow提供的函数进行计算
    # c_t=a_t+b_t

    c_t = tf.add(a_t, b_t)
    print("tensorflow实现加法运算:\n", c_t)

    # 获取默认图
    default_g = tf.get_default_graph()
    print("获取默认图:\n", default_g)

    # 数据的图属性
    print("a_t的graph:\n", a_t.graph)
    print("b_t的graph:\n", b_t.graph)

    # 操作的图属性
    print("c_t的graph:\n", c_t.graph)

    # 开启会话
    with tf.Session() as sess:
        sum_t = sess.run(c_t)
        print("在sess当中的sum_t:\n", sum_t)

        # 会话的图属性
        print("会话的图属性:\n", sess.graph)

        tf.summary.FileWriter("F:\\pycode\\logs",graph=sess.graph)

    return None


def tesor_print():
    # 实现一个加法运算
    con_a = tf.constant(3.0)
    con_b = tf.constant(4.0)

    sum_c = tf.add(con_a,con_b)

    print("打印con_a:",con_a)
    print("打印con_b:", con_b)
    print("打印sum_c:", sum_c)


if __name__ == '__main__':
    # tensorflow_demo()
    # graph_demo()

    tesorboard_demo()
    # tesor_print()