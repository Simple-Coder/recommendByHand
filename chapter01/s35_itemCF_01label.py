# -*- coding: utf-8 -*-
"""
@Time ： 2022/12/11 22:16
@Auth ： xiedong
@File ：s35_itemCF_01label.py
@IDE ：PyCharm
"""
import collections
from tqdm import tqdm
from chapter01 import dataloader
from data_set import filepaths as fp
from chapter01 import s2_basicSim as s_sim
from chapter01 import s34_userCF_01label as userCF


def getSet(triples):
    # 以物品为索引，喜欢物品的用户集
    items_users = collections.defaultdict(set)
    user_items = collections.defaultdict(set)
    for u, i, r in triples:
        if r == 1:
            user_items[u].add(i)
            items_users[i].add(u)
    return items_users, user_items


# 得到基于ItemCF的推荐列表
def get_recommendation_by_itemCF(item_sims, user_o_set):
    """
    :param item_sims: 用户的近邻集 {样本1:{近邻1，近邻2，近邻3}}
    :param user_o_set: 用户原本喜欢的物品集合{用户1:{物品1，物品2，物品3}}
    :return: 每个用户的推荐列表 {用户1：[物品1，物品2，物品3]}
    """
    recommendations = collections.defaultdict(set)
    # 遍历每个用户
    for u in user_o_set:
        # 遍历每个用户的近邻用户
        for item in user_o_set[u]:
            # 将近邻用户喜爱的电影 与自己观看的电影去重后推荐给自己
            recommendations[u] |= (set(item_sims[item]) - user_o_set[u])
    return recommendations


def trainItemCF(items_user, sim_method, user_items, k=5):
    # 寻找物品的相似物品
    item_sim = userCF.knn4set(items_user, k, sim_method)
    # 获取推荐列表
    recommedation = get_recommendation_by_itemCF(item_sim, user_items)
    return recommedation


if __name__ == '__main__':
    # 1.加载数据
    _, _, train_set, test_set = dataloader.readRecData(fp.Ml_100K.RATING, test_ratio=0.1)
    items_user, user_items = getSet(train_set)
    # 2.训练数据
    recommendations_by_itemCF = trainItemCF(items_user, s_sim.cos4set, user_items, k=5)
    # 3.推荐结果打印
    print(recommendations_by_itemCF)
