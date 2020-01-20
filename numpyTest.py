import numpy as np

# 一维度
# d = np.arange(5)                                                  # [0 1 2 3 4]
# d = np.arange(5, dtype=np.int)                                    # [0 1 2 3 4]
# d = np.arange(5, dtype=np.float)                                  # [0. 1. 2. 3. 4.]
# d = np.arange(1, 10, 2)                                           # [1 3 5 7 9]
# d = np.arange(1, 10, 2, dtype=np.float)                           # [1. 3. 5. 7. 9.]
# d = np.random.random(5)                                           # [0.30347475 0.35609686 0.9404516  0.05356108 0.3143494 ]
# d = np.zeros(10)                                                  # [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
# d = np.array([1, 3, 4, 5, 6, 9, 2])                               # [1 3 4 5 6 9 2]
# d = np.ones(10)                                                   # [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
# d = np.linspace(0, 100, 11)                                       # [  0.  10.  20.  30.  40.  50.  60.  70.  80.  90. 100.]
# d = np.empty(5)                                                   # [2.10077583e-312 6.79038654e-313 2.22809558e-312 2.14321575e-312 2.35541533e-312] 内容随机生成 取决于内存的状态

# likearr = np.arange(3)
# d = np.ones_like(likearr)                                         # [1 1 1] 返回一个结构与likearr一样的 值为1的np数组
# d = np.zeros_like(likearr)                                        # [0 0 0] 返回一个结构与likearr一样的 值为0的np数组
# d = np.empty_like(likearr)                                        # [0 0 none] 返回一个结构与likearr一样的 内容随机的np数组

# 二维度
# d = np.zeros((2, 3))                                              # [[0. 0. 0.][0. 0. 0.]]
# d = np.ones((2, 3))                                               # [[1. 1. 1.][1. 1. 1.]]
# d = np.empty((2, 3))                                              #  内容随机生成 取决于内存的状态
# d = np.random.random((2, 3))                                      # [[0.47131612 0.97750122 0.08358294] [0.20567422 0.04448278 0.28582201]]
# d = np.arange(0, 6).reshape((2, 3))                               # [[0 1 2][3 4 5]] reshape 一维转二维


# 数学计算
# a = np.arange(1, 5)                                                 # [1 2 3 4]
# b = np.arange(1, 5)                                                 # [1 2 3 4]
# print(a + b)                                                        # [2 4 6 8]
# print(a - b)                                                        # [0 0 0 0]
# print(a * b)                                                        # [1 4 9 16]
# print(a / b)                                                        # [1. 1. 1. 1.]
# print(a + 5)                                                        # [6 7 8 9]
# print(a ** 2)                                                       # [ 1  4  9 16]
# print(a ** 3)                                                       # [ 1  8 27 64]
# print(a < 3)                                                        # [ True  True False False]
# print(a & 1)                                                        # [1 0 1 0]
# print(a | 1)                                                        # [1 3 3 5]
#  对二维的数组操作， 可以裂解成先转为一维， 然后转为二维


# 聚合计算
# 聚合计算

exit(0)
# for i in d:
#     print(i)
#     exit(0)
# exit(0)
# d = np.ndarray(range(10))
# print(d)
# exit(0)


# Boolean masking
import matplotlib.pyplot as plt

a = np.linspace(0, 2 * np.pi, 50)
b = np.sin(a)
plt.plot(a,b)
mask = b >= 0
plt.plot(a[mask], b[mask], 'bo')
mask = (b >= 0) & (a <= np.pi / 2)
plt.plot(a[mask], b[mask], 'go')
plt.show()