import numpy as np
import tflearn
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression

# 训练与测试时数据实时预处理
img_pre = ImagePreprocessing()
# 均值归零
img_pre.add_featurewise_zero_center()
# 标准差归一 
img_pre.add_featurewise_stdnorm()

# 训练时数据实时增强
img_aug = ImageAugmentation()
# 左右翻转
img_aug.add_random_flip_leftright()
# 随机旋转
img_aug.add_random_rotation(max_angle=25.)

# CNN模型
# 3个是卷积层、2个最大池化层、2个全连接层
# 输入 batch*32*32*3
network = input_data(shape=[None, 32, 32, 3], data_preprocessing=img_pre, data_augmentation=img_aug)
# 卷积层1:32个卷积核，尺寸3*3*3*32，激活函数ReLU
network = conv_2d(network, 32, 3, activation='relu')
# 最大池化层1:核同conv，步幅strides=2
network = max_pool_2d(network, 2)
# 卷积层2:64个卷积核
network = conv_2d(network, 64, 3, activation='relu')
# 卷积层3
network = conv_2d(network, 64, 3, activation='relu')
# 最大池化层2
network = max_pool_2d(network, 2)
# 全连接层1：512个神经元
network = fully_connected(network, 512, activation='relu')
 # dropout：50%保留
network = dropout(network, 0.5)
# 全连接层2：10个神经元和Softmax激活函数
network = fully_connected(network, 10, activation='softmax')
 # 参数优化：Adam，损失函数：交叉熵，学习率：0.001
network = regression(network, optimizer='adam', loss='categorical_crossentropy', learning_rate=0.001)
# Tensorboard可视化
# 0: Loss, Accuracy (Best Speed).
# 1: Loss, Accuracy, Gradients.
# 2: Loss, Accuracy, Gradients, Weights.
# 3: Loss, Accuracy, Gradients, Weights, Activations, Sparsity.
model = tflearn.DNN(network, tensorboard_verbose=1,tensorboard_dir='tflearn_logs/')
