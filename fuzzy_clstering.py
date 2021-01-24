import numpy as np
from clustering_util.fairness_util import (cal_sizes, cal_ratios, cal_loss)


def distance(data, c):

    d = np.sqrt(np.sum((data - c) ** 2))
    return d


def initialize_u(n, cluster_num):  # 输入样本点数量和聚类数目

    u = np.random.rand(n, cluster_num)  # 随机生成一个隶属度矩阵
    denominator = np.sum(u, axis=1)  # 求每行的和
    return u / denominator[:, np.newaxis]  # 返回按行标准化的矩阵（每行相加是1）


def update_c(u, m, data):

    um = u ** m
    c = um.T.dot(data) / np.atleast_2d(um.T.sum(axis=1)).T
    return c


def update_u(data, c, m, label, attributes, representation, df, eta):

    n = data.shape[0]
    cluster_num = len(c)
    u = np.zeros((n, cluster_num))
    # 更新隶属度矩阵
    loss = np.zeros((n, cluster_num))
    for j in range(n):
        label_s = label.copy()
        # 改变标签，计算每个簇中的loss
        for i in range(cluster_num):
            label_s[j] = i
            sizes = cal_sizes(label_s, cluster_num)
            ratios = cal_ratios(attributes, df, label_s, cluster_num, sizes)
            loss[j][i] = cal_loss(attributes, cluster_num, ratios, representation, eta)
    for j in range(n):
        for i in range(cluster_num):
            numerator = (1 / (distance(data[j], c[i]) + loss[j][i])) ** 2
            denominator = 0
            for k in range(cluster_num):
                denominator += (1 / (distance(data[j], c[k]) + loss[j][k])) ** 2
            u[j, i] = numerator / denominator
    return u



