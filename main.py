'''
using python 3.7
# -*- coding: utf-8 -*-
Author HsuHsia
'''


import configparser
from clustering_util.config_util import read_list
from fair_fuzzy_clustering import fair_clustering


def main():

    config_file = "config/experiment_config.ini"
    config = configparser.ConfigParser(converters={'list': read_list})
    config.read(config_file)

    config_str = "bank"
    dataset = config[config_str].get("dataset")
    print("Clustering dataset: {}".format(dataset))
    num_clusters = config[config_str].getint("num_clusters")
    max_points = config[config_str].getint("max_points")
    dataset_config_file = config[config_str].get("config_file")
    m = config[config_str].getint("m")
    epsilon = config[config_str].getfloat("epsilon")
    eta = config[config_str].getfloat("eta")
    maxiter = 10

    fair_clustering(dataset, dataset_config_file, max_points, num_clusters, m, epsilon, maxiter, eta)


if __name__ == "__main__":
    main()
