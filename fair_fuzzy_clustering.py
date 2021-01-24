import configparser
from clustering_util import data_processing as dp
from clustering_util.config_util import read_list
import numpy as np
from collections import defaultdict
from fuzzy_clstering import (update_c, update_u)
from skfuzzy import cmeans
from clustering_util.fairness_util import (cal_sizes, cal_ratios, cal_loss)


def fair_clustering(dataset, config_file, max_points, cluster_num, m, epsilon, maxiter, eta):
    # 读取数据
    config = configparser.ConfigParser(converters={'list': read_list})
    config.read(config_file)
    df = dp.read_data(config, dataset)

    # 设置数据集大小
    if max_points and len(df) > max_points:
       df = dp.subsample_data(df, max_points)

    df, _ = dp.clean_data(df, config, dataset)


    # 获取平衡属性
    fairness_variable = config[dataset].getlist("fairness_variable")

    # 对敏感属性建模
    # attributes 保存每个颜色类别的点的索引
    # color_flag 从点到它所属的颜色类别的映射（与“attributes”相反）
    attributes, color_flag = {}, {}
    for variable in fairness_variable:
        colors = defaultdict(list)
        this_color_flag = [0] * len(df)
        condition_str = variable + "_conditions"
        bucket_conditions = config[dataset].getlist(condition_str)
        for i, row in df.iterrows():
            for bucket_idx, bucket in enumerate(bucket_conditions):
                if eval(bucket)(row[variable]):
                    colors[bucket_idx].append(i)
                    this_color_flag[i] = bucket_idx

        attributes[variable] = colors
        color_flag[variable] = this_color_flag
    # 敏感属性在整个数据集所占比例
    representation = {}
    for var, bucket_dict in attributes.items():
        representation[var] = {k: (len(bucket_dict[k]) / len(df)) for k in bucket_dict.keys()}
    # 选取用作定义距离的属性
    selected_columns = config[dataset].getlist("columns")
    df1 = df[[col for col in selected_columns]]
    df = df1.iloc[:, :].values
    centers, u_fz, u0, d_fz, jm, p, fpc = cmeans(df.T, c=cluster_num, m=m, error=epsilon, maxiter=1000)

    # 加入loss调整u

    u = u_fz.T
    label = np.argmax(u, axis=1)
    p = 0
    loss_old = 999999999
    while p < maxiter - 1:
        c = update_c(u, m, df)
        u = update_u(df, c, m, label, attributes, representation, df1, eta)
        label = np.argmax(u, axis=1)
        sizes = cal_sizes(label, cluster_num)
        ratios = cal_ratios(attributes, df1, label, cluster_num, sizes)
        loss = cal_loss(attributes, cluster_num, ratios, representation, eta)
        p += 1
        if np.max(np.abs(loss - loss_old)) < epsilon:
            break
        loss_old = loss


