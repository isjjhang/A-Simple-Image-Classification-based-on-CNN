from django.shortcuts import render
from django.conf import settings
import os
from .import models
import numpy as np
from PIL import Image
from mySystem import CNN_Model as CNN
os.environ['KMP_DUPLICATE_LIB_OK']='True'
# 导入模型
model = CNN.model
path_to_model = "./mySystem/CNN_on_Cifar10_Model.tfl"
model.load(path_to_model)

def classify(request):
    return render(request, 'classify.html')

def predict(request):
    kind=request.POST.get('type')

    if kind=="1":
        global model
        # Cifar-10的10个类别
        classes = ['飞机', '汽车', '鸟', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车']

        pic=request.FILES.get('pic', None)
        p=models.Pic.objects.create(pic = pic)
        p.save()
        
        # 后台图像跟目录
        root = "./static/images/"
        # 更改后尺寸x*y=32*32
        length=32
        try: 
            im = Image.open(os.path.join(root, str(pic)))
            out = im.resize((length,length),Image.ANTIALIAS)
            out= np.array(out)
            # 标准化
            out = (out-np.mean(out))/np.std(out,ddof=1)

            result = model.predict([out])[0]
            accuracy = max(result)
            prediction = result.tolist().index(accuracy)
            category = classes[prediction]

            # # 从后台删除对象
            # os.remove(os.path.join(root, str(pic)))
            p.delete()
        except:
            category=""
            accuracy=""
            pic=""
    else:
        category=""
        accuracy=""
        pic=""
    return render(request, 'classify.html',{'category': category, 'accuracy': accuracy, 'pic': pic})


