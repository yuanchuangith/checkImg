import random

import numpy as np
import os

from PIL import Image
from captcha.image import ImageCaptcha
from matplotlib import pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import tensorflow.keras as keras
# from tensorflow.keras.callbacks import ModelCheckpoint
# from tensorflow.keras.optimizers import Adadelta
# from tensorflow.keras.utils import plot_model
import tensorflow as tf
import glob

from tensorflow.python.keras.callbacks import ReduceLROnPlateau, EarlyStopping


def random_captcha(char_set,capych_size):
    if char_set is None:
        char_set = number + lowercase + capital
    captch_text=[]
    for i in range(capych_size):
        captch_text.append(random.choice(char_set))
    return captch_text
def getCaptcha_image(set,num,width=160, height=60):
    image=ImageCaptcha(width=width,height=height)
    captcha_text=random_captcha(set,num)
    captcha_texts=''.join(captcha_text)
    captcha=image.generate(captcha_texts)
    captcha_image=Image.open(captcha)
    captcha_image=np.array(captcha_image)
    return captcha_texts,captcha_image
# def text2vec(text):
#     global maxlen
#     global maxsetlen
#     global capital
#     global lowercase
#     text_len=len(text)
#
#     if text_len>maxlen:
#         raise ValueError('longer than 4')
#     vector=np.zeros(maxlen*maxsetlen)
#
#     for i,c in enumerate(text):
#         index=None
#         if c.isdigit():
#             index=maxsetlen*i+int(c)
#
#         elif c.isupper():
#             temp=capital.index(c)
#             index=36+maxsetlen*i+temp
#
#         else:
#             temp = lowercase.index(c)
#             index =10+ maxsetlen * i + temp
#         vector[index]=1
#
#     return vector
def text2vec(text):
    text_len = len(text)

    if text_len > maxlen:
        raise ValueError('longer than 4')
    vector = np.zeros([maxlen,maxsetlen])

    for i, c in enumerate(text):
        index = None
        if c.isdigit():
            index =  int(c)

        elif c.isupper():
            temp = capital.index(c)
            index = 36 +temp

        else:
            temp = lowercase.index(c)
            index = 10   + temp
        vector[i][index] = 1

    return vector
def conver2gray(img):
    if len(img.shape)>2:
        gray = np.mean(img,-1)
        return gray
    else:
        return img
def get_batch(num,batch_size):
    batch_x = np.zeros([batch_size, height, width, 1])
    batch_y = np.zeros([batch_size, maxlen, maxsetlen])
    # generrator=ImageCaptcha(height=height,width=width)
    def get_captcha_info():
        while True:
            text, image = getCaptcha_image(set_sumarrys, num)
            if image.shape==(60,160,3):
                return text,image
    for i in range(batch_size):
        text, image = get_captcha_info()
        image = tf.reshape(conver2gray(image).flatten()/255, (height, width, 1))
        batch_x[i, :] = image
        # print((image.flatten() / 255).shape)
        batch_y[i, :] = text2vec(text)
    return batch_x,batch_y
def predict_get_batch(num=4,batch_size=1):
    batch_x = np.zeros([batch_size, height, width, 1])
    batch_y = np.zeros([batch_size, maxlen, maxsetlen])
    # generrator=ImageCaptcha(height=height,width=width)
    def get_captcha_info():
        while True:
            text, image = getCaptcha_image(set_sumarrys, num)
            if image.shape==(60,160,3):
                return text,image
    texts=''
    for i in range(batch_size):
        text, image = get_captcha_info()
        image = tf.reshape(conver2gray(image).flatten()/255, (height, width, 1))
        batch_x[i, :] = image
        # print((image.flatten() / 255).shape)
        batch_y[i, :] = text2vec(text)
        texts=text
    return batch_x,batch_y,texts
def trainModel():
    chAndim=-1
    model=tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation=keras.activations.elu,
                            input_shape=(height,width,1),padding='SAME',kernel_initializer=keras.initializers.RandomNormal(mean=0.01, stddev=0.05, seed=None),bias_initializer=keras.initializers.RandomNormal(mean=0.1, stddev=0.05, seed=None)))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation=keras.activations.relu,
                                     padding='SAME'))
    model.add(tf.keras.layers.BatchNormalization(axis=chAndim))
    model.add(tf.keras.layers.MaxPool2D((2, 2)))

    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation=keras.activations.elu,
                                     padding='SAME',kernel_initializer=keras.initializers.RandomNormal(mean=0.01, stddev=0.05, seed=None),bias_initializer=keras.initializers.RandomNormal(mean=0.1, stddev=0.05, seed=None)))
    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation=keras.activations.relu,
                                     padding='SAME'))
    model.add(tf.keras.layers.BatchNormalization(axis=chAndim))
    model.add(tf.keras.layers.MaxPool2D((2, 2)))

    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation=keras.activations.elu,
                                     padding='SAME'))
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation=keras.activations.relu,
                                     padding='SAME'))
    model.add(tf.keras.layers.BatchNormalization(axis=chAndim))
    model.add(tf.keras.layers.MaxPool2D((2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(maxlen*maxsetlen, activation=keras.activations.sigmoid))
    model.add(tf.keras.layers.Reshape([maxlen, maxsetlen]))
    model.compile(loss='categorical_crossentropy',optimizer=Adam(learning_rate=1e-4),
                  metrics=['acc'],
                 )
    # model.add(tf.keras.layers.BatchNormalization())
    return model
def start():
    model = trainModel()
    step=0
    while True:
        batch_x, batch_y = get_batch(num, 128)
        model.fit(x=batch_x, y=batch_y, epochs=2, verbose=0)
        if step%10==0:
            text_batch_x, test_batch_y=get_batch(num,100)
            # text_batch_x = tf.reshape(text_batch_x, shape=[-1, height, width, 1])
            test_loss, test_acc = model.evaluate(x=text_batch_x,y=test_batch_y)
            print('step+acc', step, test_acc)
            if test_acc>0.95:
                model.save(SAVE_PATH+'captcha95.h5')
                break
        step += 1
    pass
def vec2text(vec):
    text = []
    for i, c in enumerate(vec):
        text.append(set_sumarrys[c])
    return "".join(text)
def predicate():
    model = tf.keras.models.load_model('model/captcha95.h5')
    true=10
    for i in range(100):
        data_x, data_y, text = predict_get_batch()
        prediction_value = model.predict(data_x)
        print('正确的验证码：{}'.format(text))
        prediction_values = np.argmax(prediction_value, axis=2)[0]
        prediction_values = vec2text(prediction_values)
        print(np.argmax(prediction_value, axis=2)[0])
        print('预测的验证码：{}'.format(prediction_values))
        if text.upper() == prediction_values.upper():
            true+=1
    print('预测准确率：{}'.format(true/100))




if __name__ == '__main__':
    trian=1
    SAVE_PATH = "../pythonProject3/model/"
    number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    lowercase = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
                 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    capital = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
               'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    # temp = []
    set_sumarrys = number + lowercase + capital
    num = 4
    text, image = getCaptcha_image(set_sumarrys, num)
    height = image.shape[0]
    width = image.shape[1]
    maxlen = len(text)
    maxsetlen = len(set_sumarrys)
    if trian==0:


        figeur = plt.figure()
        ax = figeur.add_subplot(111)

        ax.text(0.1, 0.9, text, ha='center', va='center', transform=ax.transAxes)

        plt.imshow(image)


        plt.show()

        start()
    elif trian==1:
        predicate()
