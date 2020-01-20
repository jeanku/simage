import numpy as np
#
# d = np.zeros((3, 3), dtype=np.int)
# d = np.ones((3, 3), dtype=np.int)
#
#
# d= np.arange(9).reshape(3, 3)
#
# for k, kv in enumerate(d):
#     for j, jv in enumerate(kv):
#         d[k, j] = k + j * 10


dd = np.arange(9).reshape((3, 3))
# print(dd)
# exit(0)

print(dd)
print(np.where(dd>3, dd, 0))
print(dd)
exit(0)