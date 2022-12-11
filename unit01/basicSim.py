# -*- coding: utf-8 -*-
"""
@Time ： 2022/12/11 15:16
@Auth ： xiedong
@File ：basicSim.py
@IDE ：PyCharm
"""


# CN 相似度（Common Neighbors）
def CN(set1, set2):
    return len(set1 & set2)


if __name__ == '__main__':
    a = {1, 2, 3}
    b = {2, 3, 4}
