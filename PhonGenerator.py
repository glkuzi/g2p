# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 09:06:28 2018

@author: User
"""

import numpy as np
import json
import pandas as pd
from sklearn.utils import shuffle
from keras.utils import Sequence, to_categorical


src = 'train.csv'
train = 'train1.csv'
valid = 'valid.csv'
alp_name = 'alphabet1.json'
phon_name = 'phonems1.json'


def gen_sets(src, train, valid, val=0.8):
    """
    Функция для разбиения исходного файла на тестовую и валидационную выборку
    """
    df = pd.read_csv(src)
    df = shuffle(df)
    df[:][: int(val * len(df))].to_csv(train, index = False, 
      index_label = False, encoding = 'utf-8')
    df[:][int(val * len(df)) + 1:].to_csv(valid, index = False, 
      index_label = False, encoding = 'utf-8')
    return 0


def to_word2(train, alphabet, phonems, count):
    n = 0
    blanks = ['0'] * n
    k_word = ['0'] * int(40 - 2 * n - len(list(train['word'][count])))
    k_phon = ['0'] * int(40 - 2 * n - len(train['transript'][count].split()))
    word = blanks + list(train['word'][count]) + blanks + k_word
    #print(word)
    phon = blanks + train['transript'][count].split() + blanks + k_phon
    #print(phon)
    coded_word = []
    coded_phon = []
    wlen = len(word)
    plen = len(phon)
    #print(plen)
    for i in range(wlen):
        coded_word.append(alphabet[word[i]])
    for i in range(plen):
        coded_phon.append(phonems[phon[i]])
        
    return coded_word, coded_phon


def to_word(train, alphabet, phonems, count, nums=5):
    """
    Функция, переводящая слова и фонемы в векторы для нейросети.
    Одной букве(фонеме) соответсвует nums элементов.
    Все слова выравниваются до размера 40 для одинакового размера сэмпла
    """
    n = (nums - 1) // 2
    blanks = ['0'] * n
    k_word = ['0'] * int(40 - 2 * n - len(list(train['word'][count])))
    k_phon = ['0'] * int(40 - 2 * n - len(train['transript'][count].split()))
    word = blanks + list(train['word'][count]) + blanks + k_word
    phon = blanks + train['transript'][count].split() + blanks + k_phon
    coded_word = []
    coded_phon = []
    wlen = len(word) - nums + 1
    plen = len(phon) - nums + 1
    for i in range(wlen):
        buf = []
        for x in word[i:i+nums]:
            buf.append(alphabet[x])
        coded_word.append(buf)
    for i in range(plen):
        #coded_phon.append(phonems[phon[i + n]])
        buf = []
        for x in phon[i:i+nums]:
            buf.append(phonems[x])
        coded_phon.append(buf)
    return coded_word, coded_phon
    

class DataGenerator(Sequence):
    """
    Генератор данных для нейросети, наследует обьект класса Sequence
    """
    def __init__(self, train, alphabet, phonems, batch_size=1):
        self.train = pd.read_csv(train)
        self.batch_size = batch_size
        self.count = 0
        self.alphabet = json.load(open(alphabet))
        self.phonems = json.load(open(phonems))
        self.coded_word, self.coded_phon = to_word(
                self.train, self.alphabet, self.phonems, self.count)
        self.on_epoch_end()

    def __getitem__(self, index):
        X = self.coded_word[index * self.batch_size:self.batch_size * (index + 1)]
        Y = self.coded_phon[index * self.batch_size:self.batch_size * (index + 1)]
        return np.array(X), np.array(Y)
    
    def __len__(self):
        return int(np.floor((len(self.coded_word)/ self.batch_size)))
    
    def on_epoch_end(self):
        self.train = (self.train).drop(self.count, axis = 0)
        self.count += 1
        self.coded_word, self.coded_phon = to_word(
                self.train, self.alphabet, self.phonems, self.count)
        return 0


def main():
    dg = DataGenerator(train, alp_name, phon_name)
    x, y = dg.__getitem__(0)
    print(len(x), len(y))
    print(x.shape, y.shape)
    print(x, y)
    return 0


if __name__ == '__main__':
    main()