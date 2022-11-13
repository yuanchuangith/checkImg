import numpy as np
import tensorflow
import tensorflow.compat.v1 as tf
from captcha.image import ImageCaptcha
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
#只做4位数字的识别

tf.disable_v2_behavior()




keep_prob = tf.placeholder(tf.float32)  # dropout

def random_captcha_text(char_set, captcha_size):
    captcha_text = []
    for i in range(captcha_size):
        c = random.choice(char_set)
        captcha_text.append(c)
    return captcha_text

def gen_captcha_text_and_image(char_set,size):
    image=ImageCaptcha()  #生成随机图片
    captcha_text=random_captcha_text(char_set,size)
    captcha_text=''.join(captcha_text)
    captcha=image.generate(captcha_text)

    captcha_image=Image.open(captcha)
    captcha_image=np.array(captcha_image)
    return  captcha_text,captcha_image

def conver2gray(img):
    if len(img.shape)>2:
        gray=np.mean(img,-1)
        return gray
    else:return img

def text2vec(text):
    text_len=len(text)
    if text_len>MAX_CAPTCHA:
        raise ValueError('longer than 4')
    vector=np.zeros(MAX_CAPTCHA*CHAR_SET_LEN)
    # alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
    #             'v', 'w', 'x', 'y', 'z']
    # ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
    #             'V', 'W', 'X', 'Y', 'Z']
    for i,c in enumerate(text):
        index = None
        if c.isdigit():

            index = i * CHAR_SET_LEN + int(c)  # [1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0....]一共四十个元素 里面只有四个元素为1
            vector[index] = 1

        elif c.isupper():
            temp = ALPHABET.index(c)
            index = 36 + CHAR_SET_LEN * i + temp
            vector[index] = 1

        else:
            temp = alphabet.index(c)
            index = 10 + CHAR_SET_LEN * i + temp
            vector[index] = 1

    return vector

def vec2text(vec):
    text=[]
    char_pos=vec.nonzero()[0]
    for i,c in enumerate(char_pos):
        number=i%10+1
        text.append(str(number))
    return ''.join(text)


# 测试 nonzero用法
# char_pos=np.array([1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1])
# char_pos =vec2text(char_pos)
# print(char_pos)

#测试编码与解码
# text='1234'
# text=text2vec(text)
# print(text)
# text=vec2text(text)
# print(text)

#生成训练batch

def get_next_batch(batch_size=128):
    batch_x=np.zeros([batch_size,IMAGE_HEIGHT*IMAGE_WIDTH])
    batch_y=np.zeros([batch_size,MAX_CAPTCHA*CHAR_SET_LEN])
    def wrap_gen_captcha_text_and_image():
        while True:
            text,image=gen_captcha_text_and_image(char_set,4)
            if image.shape==(60,160,3):
                return text,image
    templist=[]
    for i in range(batch_size):
        text,image=wrap_gen_captcha_text_and_image()
        image=conver2gray(image)
        # f = plt.figure()
        # ax = f.add_subplot(111)
        # ax.text(0.1, 0.9, text, ha='center', va='center', transform=ax.transAxes)
        # plt.imshow(image)
        # plt.show()
        batch_x[i,:]=image.flatten()/255
        print((image.flatten()/255).shape)
        batch_y[i,:]=text2vec(text)
        templist.append(text)

    return batch_x,batch_y,templist

# batch_x,batch_y=get_next_batch(batch_size=2)
# print(batch_x,batch_y)

def cnn(w_alpha=0.01, b_alpha=0.1):
    x=tf.reshape(X,shape=[-1,IMAGE_HEIGHT, IMAGE_WIDTH, 1])

    w_c1=tf.Variable(w_alpha*tf.random_normal([3,3,1,32]))
    b_c1=tf.Variable(b_alpha*tf.random_normal([32]))
    conv1=tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x,w_c1,strides=[1,1,1,1],padding='SAME'),b_c1))
    conv1=tf.nn.max_pool(conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    conv1 = tf.nn.dropout(conv1, keep_prob)

    w_c2=tf.Variable(w_alpha*tf.random_normal([3,3,32,64]))
    b_c2=tf.Variable(b_alpha*tf.random_normal([64]))
    conv2=tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1,w_c2,strides=[1,1,1,1],padding='SAME'),b_c2))
    conv2=tf.nn.max_pool(conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    conv2 = tf.nn.dropout(conv2, keep_prob)

    w_c3=tf.Variable(w_alpha*tf.random_normal([3,3,64,64]))
    b_c3=tf.Variable(b_alpha*tf.random_normal([64]))
    conv3=tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2,w_c3,strides=[1,1,1,1],padding='SAME'),b_c3))
    conv3=tf.nn.max_pool(conv3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    conv3 = tf.nn.dropout(conv3, keep_prob)

    # 全连接层

    w_d=tf.Variable(w_alpha*tf.random_normal([8*20*64,1024]))
    b_d=tf.Variable(w_alpha*tf.random_normal([1024]))
    temp = w_d.get_shape().as_list()[0]
    dense=tf.reshape(conv3,shape=[-1, w_d.get_shape().as_list()[0]])
    dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
    dense = tf.nn.dropout(dense, keep_prob)

    w_out = tf.Variable(w_alpha * tf.random_normal([1024, MAX_CAPTCHA * CHAR_SET_LEN]))
    b_out = tf.Variable(b_alpha * tf.random_normal([MAX_CAPTCHA * CHAR_SET_LEN]))
    out = tf.add(tf.matmul(dense, w_out), b_out)
    return out

def train_cnn():
    output = cnn()
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output,labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
    predict = tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN])

    max_idx_p = tf.argmax(predict, 2)
    max_idx_l = tf.argmax(tf.reshape(Y, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        step = 0
        while True:
            batch_x, batch_y,templist = get_next_batch(128)
            _, loss_ = sess.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.75})

            if step % 10 == 0:
                batch_x_test, batch_y_test,templistw = get_next_batch(100)
                acc = sess.run(accuracy, feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1.})
                print('step+acc',step, acc)
                # predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
                # text_list = sess.run(predict, feed_dict={X:batch_x, keep_prob: 1})
                # print(text_list)
                # print(templistw)
                # 如果准确率大于50%,保存模型,完成训练
                if acc > 0.8:
                    saver.save(sess, "./model/crack_capcha.model", global_step=step)
                    break

            step += 1


def crack_captcha(captcha_image):
    output = cnn()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, "./model/crack_capcha.model-6970")

        predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
        text_list = sess.run(predict, feed_dict={X: [captcha_image], keep_prob: 1})
        text = text_list[0].tolist()

        return text


if __name__ == '__main__':
    train =0
    if train == 0:
        number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
        ALPHABET = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
        char_set = number + alphabet + ALPHABET  # 如果验证码长度小于4, '_'用来补齐
        CHAR_SET_LEN = len(char_set)
        text, image = gen_captcha_text_and_image(char_set,4)
        print("验证码图像channel:", image.shape)  # (60, 160, 3)
        # 图像大小
        IMAGE_HEIGHT = 60
        IMAGE_WIDTH = 160
        MAX_CAPTCHA = len(text)
        print("验证码文本最长字符数", MAX_CAPTCHA)
        # 文本转向量

        # char_set = number


        X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH])
        Y = tf.placeholder(tf.float32, [None, MAX_CAPTCHA * CHAR_SET_LEN])

        # print(Y.shape)
        keep_prob = tf.placeholder(tf.float32)  # dropout
        train_cnn()
    if train == 1:
        number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                    'u', 'v', 'w', 'x', 'y', 'z']
        ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                    'U', 'V', 'W', 'X', 'Y', 'Z']
        char_set = number + alphabet + ALPHABET  # 如果验证码长度小于4, '_'用来补齐
        CHAR_SET_LEN = len(char_set)
        IMAGE_HEIGHT = 60
        IMAGE_WIDTH = 160

        CHAR_SET_LEN = len(char_set)

        text, image = gen_captcha_text_and_image(char_set,4)

        f = plt.figure()
        ax = f.add_subplot(111)
        ax.text(0.1, 0.9, text, ha='center', va='center', transform=ax.transAxes)
        plt.imshow(image)

        plt.show()

        MAX_CAPTCHA = len(text)
        image = conver2gray(image)
        image = image.flatten() / 255

        X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH])
        Y = tf.placeholder(tf.float32, [None, MAX_CAPTCHA * CHAR_SET_LEN])
        keep_prob = tf.placeholder(tf.float32)  # dropout

        predict_text = crack_captcha(image)
        capt=''
        for i in predict_text:
            capt+=char_set[i]
        print(capt)
        print("正确: {}  预测: {}".format(text, capt))

