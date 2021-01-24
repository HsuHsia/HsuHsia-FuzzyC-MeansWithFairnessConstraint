def read_list(config_string, delimiter=','):
    config_list = config_string.replace("\n", "").split(delimiter)
    return [s.strip() for s in config_list]