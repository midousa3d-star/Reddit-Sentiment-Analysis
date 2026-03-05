import pandas as pd


def merge_data_and_results(data_df, results_df):
    df = pd.concat([data_df, results_df], axis=1)
    return df

def use_csv_for_dataframe(filepath):
    df = pd.read_csv(filepath)
    return df

# save dataframe to a csv given dataframe and filepath
def save_df_to_csv(dataframe, filepath):
    dataframe.to_csv(filepath, index=False)