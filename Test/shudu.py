import numpy as np
import random

global left
left = np.array([set(range(1, 10)) for i in range(81)]).reshape((9, 9))


def setdata(x, y):
    value = get_random(x, y)
    left[x, y] = value

    [i.remove(value) for i in left[x, :].flatten() if type(i) == set and i.__contains__(value) and i.__len__() > 1]

    [i.remove(value) for i in left[:, y].flatten() if type(i) == set and i.__contains__(value) and i.__len__() > 1]

    xc, yc = 0 if x < 3 else (3 if x < 6 else 6), 0 if y < 3 else (3 if y < 6 else 6)

    [i.remove(value) for i in left[xc: xc + 3, yc:yc + 3].flatten() if type(i) == set and i.__contains__(value) and i.__len__() > 1]

    check()


def check():
    [deal_block(i, j) for i in [0, 3, 6] for j in [0, 3, 6]]


def deal_block(i, j):
    arrkey, arrval = deal_count(left[i:i + 3, j:j + 3].flatten())
    if arrval.__len__() > 1:
        for index, key in enumerate(arrkey):
            length = key.__len__()
            if type(key) == set and 1 < length < 9 and length == arrval[index]:
                for idx, item in enumerate(left[i:i + 3, j:j + 3].flatten()):
                    if type(item) == set and item != key:
                        dd = item - key
                        if dd.__len__() >= 1:
                            left[i + (int(idx / 3)), j + (idx % 3)] = item - key


def deal_count(data):
    temp = np.array([i for i in data if type(i) == set])
    return np.unique(temp, return_counts=True)


def get_random(x, y):
    try:
        temp = list(left[x, y])
        return random.choice(temp)
    except Exception as e:
        print(e)
        exit(0)


def randonset():
    length = 10
    index_key = -1
    temp = []
    for key, val in enumerate(left.flatten()):
        if type(val) == set and 0 < val.__len__() <= length:
            curlength = val.__len__()
            if curlength == 1:
                setdata(int(key / 9), int(key % 9))
                return
            if curlength == length:
                temp.append(key)
            else:
                temp = []
                temp.append(key)
            length = curlength
            index_key = random.choice(temp)

    if index_key > -1 and length < 10:
        setdata(int(index_key / 9), int(index_key % 9))
        return

for i in range(81):
    randonset()

print(left)
exit(0)
