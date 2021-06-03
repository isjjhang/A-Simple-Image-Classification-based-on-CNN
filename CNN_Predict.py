from tflearn.datasets import cifar10
import CNN_Model as CNN
import numpy as np
from PIL import Image
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
# Cifar-10共10个类别
CLASS_NUM = 10

model = CNN.model
path_to_model = "CNN_on_Cifar10_Model.tfl"
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data("./cifar-10")
model.load(path_to_model)

# 混淆矩阵 [标签][预测值]
Confus_Mat = np.zeros((CLASS_NUM+1,CLASS_NUM+1),int)
for index in range(0,len(Y_test)):
    x = X_test[index].reshape((32, 32, 3))
    result = model.predict([x])[0]
    prediction = result.tolist().index(max(result))
    Confus_Mat[ Y_test[index] ][ prediction ]+=1

# 测试样例数、预测正确样例数
preAll = 0.
preCor = 0.

# 统计样本每类实际总样例数、预测得到的总样例数
# 统计总样例数，预测正确数
for i in range(0,CLASS_NUM):
    for j in range(0, CLASS_NUM):
        Confus_Mat[i][CLASS_NUM]+=Confus_Mat[i][j]
        Confus_Mat[CLASS_NUM][j]+=Confus_Mat[i][j]
        preAll += Confus_Mat[i][j]
    preCor += Confus_Mat[i][i]

Confus_Mat[CLASS_NUM][CLASS_NUM] = preAll

accuracy = preCor/preAll
print("confusion matrix:")
print(Confus_Mat)
print("accuracy: %f"%(accuracy))




