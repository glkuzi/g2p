# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 08:20:02 2018

@author: User
"""

import numpy as np
import pandas as pd
import json


"""
Файл для создания словарей разметки, которые используются как 
ортогональные векторы букв и фонем.
"""


def main():
    alp_file = 'alphabet1.json'
    phon_file = 'phonems1.json'
    alp = {}
    '''
    # для создания словаря с элементами в виде чисел, также используется 
    при расшифровке вывода нейросети
    k = 34
    for i in range(1072, 1104):
        alp[chr(i)] = k
        k -= 1
    alp['ё'] = k
    k -= 1
    alp['-'] = k
    k -= 1
    alp['0'] = k
    json.dump(alp, open(alp_file, 'w'))
    print(alp)
    filename = 'train.csv'
    df = pd.read_csv(filename)
    #print(df.head())
    phon = {}
    k = 52
    for x in df['transript']:
        buf = x.split()
        for p in buf:
            if p not in phon.keys():
                phon[p] = k
                k -= 1
    phon['0'] = k
    print(phon)
    json.dump(phon, open(phon_file, 'w'))
    print(k)
    '''
    k = 35
    alp_ar = np.zeros(k)
    k = 34
    for i in range(1072, 1104):
        alp_ar[k] = 1
        alp[chr(i)] = list(alp_ar.copy())
        alp_ar[k] = 0
        k -= 1
    alp_ar[k] = 1
    alp['ё'] = list(alp_ar.copy())
    alp_ar[k] = 0
    k -= 1
    alp_ar[k] = 1
    alp['-'] = list(alp_ar.copy())
    alp_ar[k] = 0
    k -= 1
    alp_ar[k] = 1
    alp['0'] = list(alp_ar.copy())
    json.dump(alp, open(alp_file, 'w'))
    print(alp)
    filename = 'train.csv'
    df = pd.read_csv(filename)
    phon = {}
    k = 1
    k = 53
    phon_ar = np.zeros(k)
    k = 52
    for x in df['transript']:
        buf = x.split()
        for p in buf:
            if p not in phon.keys():
                phon_ar[k] = 1
                phon[p] = list(phon_ar.copy())
                phon_ar[k] = 0
                k -= 1
    phon_ar[k] = 1
    phon['0'] = list(phon_ar.copy())
    print(phon)
    json.dump(phon, open(phon_file, 'w'))    
    return 0


main()