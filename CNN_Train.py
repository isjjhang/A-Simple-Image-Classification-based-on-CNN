from tflearn.datasets import cifar10
from tflearn.data_utils import shuffle, to_categorical
import CNN_Model as CNN
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

model = CNN.model
 # 加载cifar-10 50000张训练 10000张测试
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data("./cifar-10")

X_train,Y_train = shuffle(X_train,Y_train)

# 生成one-hot矩阵，做分类label
Y_train = to_categorical(Y_train, 10)
Y_test = to_categorical(Y_test, 10)

# 迭代5轮，batch=5
model.fit(X_train, Y_train, n_epoch=1, shuffle=True, validation_set=(X_test, Y_test),show_metric=True, batch_size=100, run_id='cifar10_cnn')
model.save("./CNN_on_Cifar10_Model.tfl")