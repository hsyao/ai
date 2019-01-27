"""


"""
import tensorflow as tf


def session_demo():
    """
    会话演示
    :return:
    """

    a_t = tf.constant(10)
    b_t = tf.constant(20)

    c_t = tf.add(a_t, b_t)

    e = tf.placeholder(tf.float32)
    f = tf.placeholder(tf.float32)
    sum_g = tf.add(e, f)
    print("sum_g", sum_g)
    print("tensorflow实现加法运算:\n", c_t)

    # 开启会话
    # 传统会话定义
    sess = tf.Session()

    print("c_t:", sess.run(c_t))
    print("采用feed_dict参数:\n", sess.run(sum_g, feed_dict={e: 90, f: 80}))
    sess.close()

    # config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)
    with tf.Session() as sess:
        # 同时执行多个tensor
        print(sess.run([a_t, b_t, c_t]))

        # 方便获取张量值的方法
        print("在sess当中的sum_t:\n", c_t.eval())

        # 会话的图属性
        print("会话的图属性:\n", sess.graph)

        # print(c_t.eval())


if __name__ == '__main__':
    session_demo()
