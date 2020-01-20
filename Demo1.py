# name = '123'
#
# def app():
#     print('app say')
#
# from Demo2 import Demo2
#
# class Demo1():
#     def say(self):
#         # Demo2().say()
#         print('demo1.say')
#
#     # def
#
# # eval('from Demo2 import Demo2')
#
# class Demo3():
#     def say(self):
#         # Demo2().say()
#         print('demo1.say')
#
# # Demo2.Demo2().say()
#
#
#
# class C(Demo1, Demo3):
#     pass
#
# d = Demo1()
# c = C()
#
# # print(isinstance('阿', str))
# print(isinstance(d, Demo1))
# print(isinstance(c, Demo1))
# print(isinstance(c, Demo3))
# print(issubclass(C, Demo1))
# exit(0)
# eval('Demo2().say()')

# def getnext():
#     i = 0
#     while i < 10:
#         yield i
#         i += 1

# import inspect
# # d1 = [i for i in range(10)]
# d = inspect.isgeneratorfunction(range(10))
# d1 = inspect.isgenerator(range(10))
# # d = inspect.isgeneratorfunction(d1)
#
# # print(isfunction(getnext))
# print(d)
# print(d1)
# exit(0)
# exit(0)
# a = 1
# def a():
#     a = 1
#     return a
# import dis
# dis.dis(a)
# from memory_profiler import profile
#
# @profile
# def read():
#     a = list(range(10000000))
#     b = a[10:]
#     print('readed')


# if __name__ == '__main__':
#     read()

# import random
# import threading
#
# results = []
#
#
# def compute():
#     results.append(sum([random.randint(1, 100) for i in range(1000000)]))
#
#
# workers = [threading.Thread(target=compute) for x in range(4)]
# for worker in workers:
#     worker.start()
#
# for worker in workers:
#     worker.join()
#
# print("results: %s" % results)

import multiprocessing
import random

results = []

def compute(n):
    return sum([random.randint(1, 100) for i in range(1000000)])


pool = multiprocessing.Pool(8)
print("results: %s" % pool.map(compute, range(8)))


