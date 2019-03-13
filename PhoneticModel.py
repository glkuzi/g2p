# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 11:20:10 2018

@author: User
"""

from PhonGenerator import *
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import History
from keras.models import load_model
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt


train = 'train1.csv'
valid = 'valid.csv'
test = 'test.txt'
metr = 'metric.csv'
#alp_name = 'alphabet1.json'
#phon_name = 'phonems1.json'


def to_transcr(arr):
    transcr = []
    phon = json.load(open('phonems2.json'))
    phon = {y:x for x,y in phon.items()}
    arr = arr[0]
    for y in arr:
        transcr.append(phon[y[2]])
    return transcr


def to_word_only(train, alphabet, count, nums=5):
    """
    Возвращает тоько закодированное слово, для тестовой выборки.
    """
    n = (nums - 1) // 2
    blanks = ['0'] * n
    k_word = ['0'] * int(40 - 2 * n - len(list(train[count])))
    word = blanks + list(train[count]) + blanks + k_word
    coded_word = []
    wlen = len(word) - nums + 1
    for i in range(wlen):
        buf = []
        for x in word[i:i+nums]:
            buf.append(alphabet[x])
        coded_word.append(buf)
    return coded_word


def mfunc(file, target):
    lfile = file.split()
    ltarget = target.split()
    leng = min(len(lfile), len(ltarget))
    count = 0
    for i in range(leng):
        if lfile[i] == ltarget[i]:
            count += 1
    return count
            


def main():
    flag = True
    tr = DataGenerator(train, alp_name, phon_name)
    val = DataGenerator(valid, alp_name, phon_name)
    if flag:
        model = load_model('mod.hdf5')
    else:
        model = Sequential()
        model.add(Dense(35, activation = 'tanh', input_shape = (5,35)))
        model.add(Dense(35, activation = 'tanh'))
        model.add(Dense(53, activation = 'softmax'))
        model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', 
                      metrics = ['accuracy'])
    history = History()
    model.fit_generator(tr, epochs = 10, validation_data = val, callbacks = [history])
    model.save('mod.hdf5')
    #plt.plot(history.history['val_loss'])
    """
    Тест для проверки точности, считается процент совпадений как число 
    совпадающих фонем на их общее число
    """
    i = 0
    met = pd.read_csv(metr, engine = 'python')
    words = list(met['word'])
    targets = list(met['transript'])
    files = []
    count = 0
    L = 0
    for i in range(len(words)):
        a = []
        coded_word = to_word_only(words, json.load(open(alp_name)), i)
        a.append(model.predict_classes(np.array(coded_word)))
        files.append((" ".join(str(x) for x in to_transcr(a))).replace(' 0', ''))
        L += max(len(files[i]), len(targets[i]))
        count += mfunc(files[i], targets[i])
    print(count / L)
    '''
    # для заполнения тестового файла по заданию хакатона
    fl = open('fin.txt', 'w')
    for i in range(len(words)):
        a = []
        coded_word = to_word_only(words, json.load(open(alp_name)), i)
        a.append(model.predict_classes(np.array(coded_word)))
        #fl = open('fin.txt', 'wa')
        fl.write((" ".join(str(x) for x in to_transcr(a))).replace(' 0', ''))
        fl.write('\n')
        #print((" ".join(str(x) for x in to_transcr(a))).replace(' 0', ''), fl)
        #fl.close()
    #print(coded_phon)
    '''
    return 0


if __name__ == '__main__':
    main()