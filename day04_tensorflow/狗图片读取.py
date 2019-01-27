import os
import tensorflow as tf

# 1.构造文件名队列

# 1.1.获取文件名列表
filename_list = os.listdir(r"F:\内网通文件\7期-深度学习day02\02-代码\dog")
# print(file_list)

# 构造路径+文件名列表作为string_tensor
file_list = [os.path.join("F:\\内网通文件\\7期-深度学习day02\\02-代码\\dog\\", file) for file in filename_list]

# print(file_list)

file_queue = tf.train.string_input_producer(file_list)

# 2.读取与解码
# 实例化读取器,读取文件名队列
reader = tf.WholeFileReader()
key, value = reader.read(file_queue)

print("key:\n", key)
print("value:\n", value)

# 解码图片数据
image = tf.image.decode_jpeg(value)

print("image:\n", image)

# 处理图片的代销
image_resize = tf.image.resize_images(image, [200, 200])
print("images_resize:\n", image_resize)

# 设置固定形状,这里可以使用静态形状api去修改
image_resize.set_shape([200,200,3])
print("image_resize:\n", image_resize)

# 4. 批处理图片数据
# 每个样本的形状必须全部定义
image_batch = tf.train.batch([image_resize], batch_size=100, num_threads=1, capacity=100)

print("image_batch:\n",image_batch)
# 开启会话
with tf.Session() as sess:
    # 开启线程,处理队列

    # 创建线程协调器
    coord = tf.train.Coordinator()

    # 开启子线程去读取数据
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)

    sess_key, sess_value,sess_image,sess_image_resize = sess.run([key, value, image, image_resize])
    # 文件名队列张量
    print("sess_key:\n", sess_key)
    # 图片张量队列
    print("sess_value:\n", sess_value)

    # 解码后的图片张量队列
    print("sess_image:\n", sess_image)

    print("sess_image_batch\n",sess_image_resize)

    # 关闭子线程,回收
    coord.request_stop()

    coord.join(threads)
if __name__ == '__main__':
    pass
