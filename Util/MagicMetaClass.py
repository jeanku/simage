#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""metaclass静态方法调用处理类"""

__author__ = ''
import cv2


class BaseMagicMetaClass(type):
    def __getattr__(self, key):
        if key == "__new__":
            return object.__new__(self)
        try:
            print(123)
            exit(0)
            return object.__getattribute__(self, key)
        except:
            print(4123)
            exit(0)
            return getattr(cv2, key)


class MagicMetaClass(metaclass=BaseMagicMetaClass):
    pass
