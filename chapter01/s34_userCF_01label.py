# -*- coding: utf-8 -*-
"""
@Time ： 2022/12/11 21:19
@Auth ： xiedong
@File ：s34_userCF_01label.py
@IDE ：PyCharm
"""
import collections
from tqdm import tqdm
from chapter01 import dataloader
from data_set import filepaths as fp
from chapter01 import s2_basicSim as s_sim


def getSet(triples):
    user_items = collections.defaultdict(set)
    for u, i, r in triples:
        if r == 1:
            user_items[u].add(i)
    return user_items


# 得到基于UserCF的推荐列表
def get_recommendation_by_userCF(user_sims, user_o_set):
    """
    :param user_sims: 用户的近邻集 {样本1:{近邻1，近邻2，近邻3}}
    :param user_o_set: 用户原本喜欢的物品集合{用户1:{物品1，物品2，物品3}}
    :return: 每个用户的推荐列表 {用户1：[物品1，物品2，物品3]}
    """
    recommendations = collections.defaultdict(set)
    for u in user_sims:
        for sim_u in user_sims[u]:
            # 将近邻用户喜爱的电影 与自己观看的电影去重后推荐给自己
            recommendations[u] |= (user_o_set[sim_u] - user_o_set[u])
    return recommendations


# knn算法
def knn4set(trains_set, k, sim_method):
    """
    :param trains_set: 训练集
    :param k: 近邻数量
    :param sim_method: 相似度方法
    :return: {样本1：{近邻1，近邻2，近邻3。。。}}
    """
    sims = {}
    for e1 in tqdm(trains_set):
        ulist = []  # 初始化列表来记录样本e1的近邻
        for e2 in trains_set:
            if e1 == e2 or len(trains_set[e1] & trains_set[e2]) == 0:
                continue
            sim = sim_method(trains_set[e1], trains_set[e2])
            ulist.append((e2, sim))
        # 排序后截取K的样本
        sims[e1] = [i[0] for i in sorted(ulist, key=lambda x: x[1], reverse=True)[:k]]
    return sims


def trainUserCF(user_items, sim_method, k=5):
    # 寻找用户的相似用户
    user_sim = knn4set(user_items, k, sim_method)
    # 获取推荐列表
    recommedation = get_recommendation_by_userCF(user_sim, user_items)
    return recommedation


if __name__ == '__main__':
    # 1.加载数据
    _, _, train_set, test_set = dataloader.readRecData(fp.Ml_100K.RATING, test_ratio=0.1)
    user_items = getSet(train_set)
    # 2.训练数据
    recommendations_by_userCF = trainUserCF(user_items, s_sim.cos4set, k=5)
    # 3.推荐结果打印
    print(recommendations_by_userCF)
