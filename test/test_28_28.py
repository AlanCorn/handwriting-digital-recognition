import numpy as np
from keras.models import load_model
from PIL import Image
# 读取图片转成灰度格式
imgPath = ['../pic/zero.png','../pic/one.png','../pic/two.png','../pic/three.png',
           '../pic/four.png','../pic/five.png','../pic/six.png','../pic/seven.png',
           '../pic/eight.png','../pic/nine.png']

result = []
for each in imgPath:
    img = Image.open(each).convert('L')
    model = load_model('./googlenet.model')
    if img.size[0] != 28 or img.size[1] != 28:
        img = img.resize((28, 28))
    arr = []
    for i in range(28):
        for j in range(28):
            pixel = 1.0 - float(img.getpixel((j, i)))/255.0
            arr.append(pixel)
    # arr1 = np.array(arr).reshape((1, 784))
    arr1 = np.array(arr).reshape(-1,28,28,1)
    predict = model.predict(arr1)[0]
    result.append(np.argmax(predict))

print('识别结果：')

for each in imgPath:
    print(each,':',result[imgPath.index(each)])
