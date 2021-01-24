from functools import partial
from collections import defaultdict


def cal_sizes(label, num_clusters):
    sizes = [0 for _ in range(num_clusters)]
    for p in label:
        sizes[p] += 1
    return sizes


def cal_ratios(attributes, df, label, num_clusters, sizes):
    fairness = {}
    # For each point in the dataset, assign it to the cluster and color it belongs too
    for attr, colors in attributes.items():
        fairness[attr] = defaultdict(partial(defaultdict, int))
        for i, row in enumerate(df.iterrows()):
            cluster = label[i]
            for color in colors:
                if i in colors[color]:
                    fairness[attr][cluster][color] += 1
                    continue
    ratios = {}
    for attr, colors in attributes.items():
        attr_ratio = {}
        for cluster in range(num_clusters):
            if sizes[cluster] != 0:
                attr_ratio[cluster] = [fairness[attr][cluster][color] / sizes[cluster] for color in sorted(colors.keys())]
            else:
                attr_ratio[cluster] = [0 for color in sorted(colors.keys())]
        ratios[attr] = attr_ratio
    return ratios


def cal_loss(attributes, num_clusters, ratios, representation, eta):
    loss = []
    for attr, colors in attributes.items():
        for k in range(num_clusters):
            loss_color = 0
            for color in sorted(colors.keys()) :
                loss_color += ((ratios[attr][k][color] - representation[attr][color]) ** 2)   # 一个簇内颜色的偏差和
            avg_loss_color = loss_color / len(colors)       # 一个簇内所有颜色的偏差的均值
            loss.append(avg_loss_color)           # 所有簇内所有颜色的偏差

    sum_loss = sum(loss)
    return eta * sum_loss

