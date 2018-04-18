#!/usr/bin/python
# -*- coding: utf-8 -*-
# https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html : 数据集太小的方法

# Using the bottleneck features of a pre-trained network Xception
# Method:   1. ensemble features of xception and inception + lr (Kaggle rank: 494 score:0.3039)
#           2. ensemble predicted probs of xception and inception + lr(Kaggle rank: 465 score:0.286)
#        => 3. ensemble features of xception and inception + nn (score:0.3037)
# score: 0.4 have to adjust the parameters: https://www.zhihu.com/question/41631631

import pandas as pd
import numpy as np
import pickle
from os import listdir
from keras import Sequential
from keras.layers import Dense, Flatten, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
from keras.applications import xception,inception_v3
from keras.preprocessing import image

data_dir = './data/'
SEED = 1987
POOLING = 'avg'
INPUT_SIZE = 299


def read_img(img_id, train_or_test, size):
    img = image.load_img(data_dir+train_or_test+"/"+str(img_id)+".jpg", target_size=size)
    img_pixels = image.img_to_array(img)
    return img_pixels


def main():
    # labels = pd.read_csv(data_dir+'labels.csv')
    # num_classes = len(labels.groupby('breed'))
    # selected_labels = labels.groupby('breed').count().sort_values(by='id',ascending=False).head(num_classes).index.values
    # labels = labels[labels['breed'].isin(selected_labels)]
    # labels['target'] = 1
    # labels['rank'] = labels.groupby('breed').rank()['id']
    # labels_pivot = labels.pivot('id', 'breed', 'target').reset_index().fillna(0) # values必须是breed和target对应的值
    # np.random.seed(SEED)
    # # rnd = np.random.random(len(labels))
    # # is_train = rnd < 0.8
    # # is_val = rnd >= 0.8
    # y_train = labels_pivot[selected_labels].values
    # # ytr = y_train[is_train]
    # # yv = y_train[is_val]
    # ytr = y_train
    #
    # x_train = np.zeros((len(labels), INPUT_SIZE, INPUT_SIZE, 3), dtype='float32')
    # for i, img_id in tqdm(enumerate(labels['id'])):
    #     # print i, img_id
    #     img = read_img(img_id, 'train', (INPUT_SIZE,INPUT_SIZE))
    #     x = xception.preprocess_input(np.expand_dims(img.copy(), axis=0))
    #     x_train[i] = x
    # print('Train Images shape: {} size: {:,}'.format(x_train.shape, x_train.size))
    #
    # num_tests = len(listdir(data_dir + '/test/'))
    # x_test = np.zeros((num_tests,INPUT_SIZE, INPUT_SIZE, 3), dtype='float32')
    # test_id = []
    # for i in range(num_tests):
    #     img_file_name = listdir(data_dir + '/test/')[i]
    #     img_id = img_file_name[0:len(img_file_name)-4]
    #     img = read_img(img_id, 'test',(INPUT_SIZE,INPUT_SIZE))
    #     x = xception.preprocess_input(np.expand_dims(img.copy(), axis=0))
    #     x_test[i] = x
    #     test_id.append(img_id)
    #
    # # xtr = x_train[is_train]
    # xtr = x_train
    # # xv = x_train[is_val]
    #
    # xception_bottleneck = xception.Xception(weights='imagenet', include_top=False, pooling=POOLING)
    # train_x_bf = xception_bottleneck.predict(xtr, batch_size=32, verbose=1)
    # valid_x_bf = xception_bottleneck.predict(x_test, batch_size=32, verbose=1)
    #
    # inception_bottleneck = inception_v3.InceptionV3(weights='imagenet', include_top=False, pooling=POOLING)
    # train_i_bf = inception_bottleneck.predict(xtr, batch_size=32, verbose=1)
    # valid_i_bf = inception_bottleneck.predict(x_test, batch_size=32, verbose=1)
    #
    # train_x = np.hstack([train_x_bf, train_i_bf])
    # test_x = np.hstack([valid_x_bf, valid_i_bf])
    # data = {'train_x':train_x, 'train_y':ytr, 'test_x':test_x}
    # with open('x_inception_fc_data.pickle', 'wb') as handle:
    #     pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(data_dir+'xicpt_data.pickle', 'rb') as handle:
        load_data = pickle.load(handle)

    fc = Sequential()
    fc.add(Dense(500, activation='relu', input_shape=(4096,)))
    fc.add(Dropout(0.2))
    fc.add(BatchNormalization())
    fc.add(Dense(load_data['num_class'],activation='softmax'))
    fc.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
    fc.fit(load_data['train_x'], load_data['train_y'],
              batch_size=32,
              epochs=10,
              verbose=1)
    valid_probs = fc.predict(load_data['test_x'])

    # without formatting
    df1 = {'id': load_data['test_id']}
    res1 = pd.DataFrame(data=df1)
    res2 = pd.DataFrame(columns=load_data['selected_labels'], data=valid_probs)
    res = pd.concat([res1, res2], axis=1)
    # res.to_csv("./x_icpt_fc"+str(dropout)+"_"+str(output)+"_"+str(opt)+"_"+str(batch)+".csv", index=False)

    # format as the sample submission
    sample_submission = pd.read_csv(data_dir + 'sample_submission.csv')
    sample_ids = list(sample_submission['id'])
    sample_breeds = list(sample_submission.columns.values)[1:]
    reorder_df = res.set_index('id')
    reorder_df = reorder_df.loc[sample_ids].reset_index().reindex(columns=['id'] + sample_breeds)
    reorder_df.to_csv("./xicpt_fcbnorm.csv", index=False)


if __name__ == '__main__':
    main()

"""
总结防止overfitting的方法（数据集小的时候）：dropout，data augmentation, l1 and l2 regularization抗干扰能力强：https://blog.csdn.net/u012162613/article/details/44261657

"""