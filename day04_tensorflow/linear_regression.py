"""


"""
import tensorflow as tf
import os
def linear_regression():
    """
    自实现线性回归
    :return: None
    """

    # 1.准备数据集: y=0.8x+0.7  100个样本
    # y_true [100,1]
    with tf.variable_scope("original_data"):
        x = tf.random_normal(shape=(100, 1), mean=2, stddev=2)
        y_true = tf.matmul(x, [[0.8]]) + 0.7

    # 2.构建线性模型
    with tf.variable_scope("linear_model"):
        weights = tf.Variable(initial_value=tf.random_normal(shape=(1, 1)), name='weights')
        bias = tf.Variable(initial_value=tf.random_normal(shape=(1, 1)), name='bias')

    y_predict = tf.matmul(x, weights) + bias

    # 4.构建损失函数,求均方误差,预测值与真实值之间的误差
    with tf.variable_scope("loss"):
        error = tf.reduce_mean(tf.square(y_predict - y_true, name='error_op'))

    # 5.梯度下降优化损失:需要制定学习率(超参数)
    with tf.variable_scope("gd_optimizer"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01, name='optimizer').minimize(error)

    # 收集变量
    tf.summary.scalar('Error', error)
    tf.summary.histogram('Weights', weights)
    tf.summary.histogram('Bias', bias)
    # 合并变量
    merge = tf.summary.merge_all()

    # 初始化变量
    init = tf.global_variables_initializer()

    # 初始化保存对象
    saver = tf.train.Saver()

    # 定义训练标志
    train_flag = 0

    # 5.开启会话
    with tf.Session() as sess:
        # 开始变量初始化
        sess.run(init)
        print("随机初始化的权重变为%f,偏置为%f" % (weights.eval(), bias.eval()))

        file_writer = tf.summary.FileWriter('F:\\pycode\\logs', graph=sess.graph)
        if train_flag == 1:
            for i in range(500):
                sess.run(optimizer)
                print("第%d次的误差为%f,权重为%f,偏置为%f" % (i + 1, error.eval(), weights.eval(), bias.eval()))

                # 运行合并变量op
                summary = sess.run(merge)
                file_writer.add_summary(summary, i)

                # 保存模型
                if i % 10 == 0:
                    saver.save(sess, 'F:\\pycode\\save_model\\myregression.ckpt')

        # 模型加载
        if os.path.exists("F:/pycode/save_model/checkpoint"):
            saver.restore(sess,'F:\\pycode\\save_model\\myregression.ckpt')
            print("模型加载后的权重变为%f,偏置为%f,此时的均方误差为%f" % (weights.eval(), bias.eval(), error.eval()))
    return None


if __name__ == '__main__':
    linear_regression()
