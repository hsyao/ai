"""


"""

import tensorflow as tf

# a=tf.constant(10)
# b=tf.constant([1,5,6,10])
# c=tf.constant([[3],[9],[34],[2]],dtype=tf.int32)
#
# print(a.shape,b.shape,c.shape)


def tensor_demo():
    """
    张量的介绍
    :return:
    """

    a=tf.constant(value=30.0,dtype=tf.float32,name='a')

    b=tf.constant([[1,2],[3,4]],dtype=tf.int32,name="b")

    a2 = tf.constant(value=30.0,dtype=tf.float32,name="a2")

    c = tf.placeholder(dtype=tf.float32,shape=[2,3,4],name='c')

    sum = tf.add(a,a2,name="my_add")
    print(a,a2,b,c)
    print(sum)

    # 获取张量属性
    print("a的图属性:\n",a.graph)
    print('b的名字:\n',b.name)
    print("a2的形状:\n",a2.shape)
    print("c的数据类型:\n",c.dtype)
    print("sum的op:\n",sum.op)

    # 获取静态形状
    print("b的静态形状:\n",b.get_shape())

    # 定义占位符
    a_p = tf.placeholder(dtype=tf.float32,shape=[None,None])
    b_p = tf.placeholder(dtype=tf.float32,shape=[None,10])
    c_p = tf.placeholder(dtype=tf.float32,shape=[3,2])

    # 获取静态形状
    print("a_p的静态形状为:\n",a_p.get_shape())
    print("b_p的静态形状为:\n",b_p.get_shape())
    print("c_p的静态形状为:\n",c_p.get_shape())

    # 形状更新
    a_p.set_shape([2,3])
    # 静态形状已经固定部分就不能修改了
    # a_p.set_shape([3,2])
    # b_p.set_shape([10,10])

    # c_p.set_shape([2,3])

    # 静态形状已经固定的部分包括它的阶数,如果阶数固定了,就不能跨阶更新形状
    # 如果想要跨阶改变形状,就要用动态形状
    # a_p.set_shape([1,2,3])

    # 获取静态形状
    print("a_p的静态形状为:\n",a_p.get_shape())
    print("b_p的静态形状为:\n",b_p.get_shape())
    print("c_p的静态形状为:\n",c_p.get_shape())


    # 动态形状
    c_p_r=tf.reshape(c_p,[1,2,3])

    # c_p_r = tf.reshape(c_p,[2,3])

    # 动态形状,改变的时候,不能改变元素的总个数
    # c_p_r2 = tf.reshape(c_p,[3,1])

    print("动态形状的结果:\n",c_p_r)


def variable_demo():
    """
    变量op
    :return:
    """

    # 定义变量
    with tf.variable_scope("name"):
        a = tf.Variable(initial_value=30)
    b = tf.Variable(initial_value=40)
    c= tf.add(a,b)

    print(a)
    print(b)
    # 初始化变量
    init = tf.global_variables_initializer()

    # 开启会话
    with tf.Session() as sess:
        # 变量初始化
        sess.run(init)
        print("累加和:",sess.run(c))


if __name__ == '__main__':
    # tensor_demo()

    variable_demo()