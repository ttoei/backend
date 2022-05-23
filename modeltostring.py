import math
from pyexpat import model
import cv2
from tensorflow import keras
from tensorflow.keras.layers.experimental.preprocessing import StringLookup
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import os
import random
from pythainlp.util import *

# from keras.backend import manual_variable_initialization
# manual_variable_initialization(True)
SEED = 0
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'  # new flag present in tf 2.0+
# tf.reset_default_graph()
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
# !/usr/bin/python -tt
# -*- coding: utf-8

# import pytesseract

# from keras.optimizers import TFOptimizer
image_width = 150
image_height = 32


def num_char():
    max_len = 22
    characters = ['้', '์', '็', '๊', '่', ',', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', 'A',
                  'B', 'C', 'D', 'E', 'F',
                  'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
                  'a', 'b', 'e', 'f', 'i',
                  'l', 'm', 'o', 'p', 'r', 's', 't', 'u', 'v', 'ก', 'ข', 'ค', 'ง', 'จ', 'ช', 'ซ', 'ญ', 'ด', 'ต', 'ท',
                  'น', 'บ', 'ป', 'พ', 'ฟ',
                  'ม', 'ย', 'ร', 'ล', 'ว', 'ส', 'ห', 'อ', 'ฮ', 'ะ', 'ั', 'า', 'ำ', 'ิ', 'ี', 'ื', 'ุ', 'ู', 'เ', 'แ',
                  'โ', 'ใ']
    xx = collate(characters)
    AUTOTUNE = tf.data.AUTOTUNE
    # สับคำว่ามีตัวพญัญชนะ
    # Mapping characters to integers.
    char_to_num = StringLookup(vocabulary=xx, mask_token=None)

    # Mapping integers back to original characters.
    num_to_char = StringLookup(
        vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
    )
    print(xx)
    return num_to_char, max_len, char_to_num


num_to_char, max_len, char_to_num = num_char()


def pre_process(img):
    width = img.shape[1]
    hight = img.shape[0]
    if hight > 900:
        cut_Hight_img = math.ceil(hight / 2)
        cropX = img[cut_Hight_img:cut_Hight_img + (hight - (cut_Hight_img + 100)), 0:0 + width]
    elif hight <= 900:
        cut_Hight_img = hight
        cropX = img[cut_Hight_img:cut_Hight_img + (hight - (cut_Hight_img + 100)), 0:0 + width]

    rgb = cv2.pyrDown(cropX)
    small = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grad = cv2.morphologyEx(small, cv2.MORPH_GRADIENT, kernel)
    _, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    # using RETR_EXTERNAL instead of RETR_CCOMP
    contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    mask = np.zeros(bw.shape, dtype=np.uint8)
    crop_img = []
    for idx in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[idx])
        mask[y:y + h, x:x + w] = 0
        cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
        r = float(cv2.countNonZero(mask[y:y + h, x:x + w])) / (w * h)
        if r > 0.45 and w > 8 and h > 8:
            cv2.rectangle(cropX, (x, y), (x + w - 1, y + h - 1), (0, 255, 0), 2)
            crop = rgb[y:y + h + 1, x:x + w + 1]
            crop_img.append(crop)
    # plt.title('crop_img')
    # plt.imshow(crop_img[23])
    # plt.show() 
    print('pre_processes success')
    return crop_img


def numpy_to_tensor(array):
    array = tf.convert_to_tensor(array, dtype=tf.float32)
    array = tf.expand_dims(array, 2)
    array = distortion_free_resize(array, img_size=(image_width, image_height))
    array = tf.cast(array, tf.float32) / 255.0

    return array


def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search.
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
              :, :max_len
              ]
    # Iterate over the results and get back the text.
    output_text = []
    for res in results:
        res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text


def distortion_free_resize(image, img_size):
    w, h = img_size
    image = tf.image.resize(image, size=(h, w), preserve_aspect_ratio=True)

    # Check tha amount of padding needed to be done.
    pad_height = h - tf.shape(image)[0]
    pad_width = w - tf.shape(image)[1]

    # Only necessary if you want to do same amount of padding on both sides.
    if pad_height % 2 != 0:
        height = pad_height // 2
        pad_height_top = height + 1
        pad_height_bottom = height
    else:
        pad_height_top = pad_height_bottom = pad_height // 2

    if pad_width % 2 != 0:
        width = pad_width // 2
        pad_width_left = width + 1
        pad_width_right = width
    else:
        pad_width_left = pad_width_right = pad_width // 2

    image = tf.pad(
        image,
        paddings=[
            [pad_height_top, pad_height_bottom],
            [pad_width_left, pad_width_right],
            [0, 0],
        ],
    )

    image = tf.transpose(image, perm=[1, 0, 2])
    image = tf.image.flip_left_right(image)
    return image


def img_to_string(img):
    # img = cv2.imread(filename)
    # plt.title('Word Detection')
    # plt.imshow(img)
    # plt.show()
    # print(img.shape)

    img_pre = pre_process(img)

    # convert to grayscal
    for i in range(len(img_pre)):
        img_pre[i] = cv2.cvtColor(img_pre[i], cv2.COLOR_BGR2GRAY)

    print(img_pre[1].shape)

    data_text = []
    imgto_tensor = []

    model_load = load_model('./Model/handwriting_recognizer_model.h5')

    model_load.load_weights('./Model/handwriting_recognizer_model_weights.h5')

    # convert arry to string

    for i in range(len(img_pre)):
        imgto_tensor.append(numpy_to_tensor(img_pre[i]))
        # เพิ่มขนาดของอาเรย์ภาพจาก 3 ไป 4 มิติ
        imgto_tensor[i] = tf.expand_dims(imgto_tensor[i], 0)

    for i in range(len(imgto_tensor)):
        preds = model_load.predict(imgto_tensor[i])
        # print(preds.shape)
        pred_texts = decode_batch_predictions(preds)
        data_text.append(pred_texts)

    for i in range(len(data_text)):
        data_text = data_text[::-1]

    listToStr = ' '.join(map(str, data_text))
    return listToStr

# filename = 'G:\\My Drive\\titis\\reject_Data\\CROP100\\CROP18.JPG'
# a = img_to_string(filename)  
# print(a)
