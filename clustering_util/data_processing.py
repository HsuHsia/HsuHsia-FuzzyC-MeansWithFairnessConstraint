import pandas as pd


def read_data(config, dataset):
    csv_file = config[dataset]["csv_file"]
    df = pd.read_csv(csv_file, sep=config[dataset]["separator"])

    if config["DEFAULT"].getboolean("describe"):
        print(df.describe())

    return df


def clean_data(df, config, dataset):

    # selected_columns = ['age', 'balance', 'duration']
    selected_columns = config[dataset].getlist("columns")

    #
    # (variables_of_interest) = ['marital', 'default']
    variables_of_interest = config[dataset].getlist("fairness_variable")

    # Bucketize text data
    text_columns = config[dataset].getlist("fairness_variable", [])
    for col in text_columns:
        # Cat codes is the 'category code'. Aka it creates integer buckets automatically.
        df[col] = df[col].astype('category').cat.codes

    # Remove the unnecessary columns. Save the variable of interest column, in case
    # it is not used for clustering.
    variable_columns = [df[var] for var in variables_of_interest]
    # df = df[[col for col in selected_columns]]

    # Convert to float, otherwise JSON cannot serialize int64
    for col in df:
        if col in text_columns or col not in selected_columns: continue
        df[col] = df[col].astype(float)

    if config["DEFAULT"].getboolean("describe_selected"):
        print(df.describe())

    return df, variable_columns


def subsample_data(df, N):
    return df.sample(n=N).reset_index(drop=True)

