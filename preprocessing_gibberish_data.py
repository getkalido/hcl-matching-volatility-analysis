from typing import List
import math
import pandas as pd
from tqdm import tqdm
import numpy as np
from mlflow_matching_model import MlFlowMatchingModel

from preprocessing_skills_master_data import create_matching_dataset, add_noise_to_text, get_predictions
from config import GIBBERISH, MLFLOW_TRACKING_PASSWORD, MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME

np.random.seed(7)
pd.set_option('display.max_columns', None)


def read_gibberish_data(filepath:str) -> List[str]:
    df = pd.read_csv(filepath, header=0, names=['need', 'offer'], encoding= 'unicode_escape')
    need = list(df['need'].unique())
    offer = list(df['offer'].unique())
    l = need + offer
    return list(set(l))


def save_results(results_df: pd.DataFrame, model_name: str, model_v: int, with_noise: bool=False, dropout: bool=False, max_rows: int=10000):
    if dropout:
        results_df.to_csv(f'./predictions_gibberish/with_single_dropout_{model_name}_{str(model_v)}_sample_{str(max_rows)}.csv')
    elif with_noise:
        results_df.to_csv(f'./predictions_gibberish/with_noise_{model_name}_{str(model_v)}_sample_{str(max_rows)}.csv')
    else:
        results_df.to_csv(f'./predictions_gibberish/{model_name}_{str(model_v)}_sample_{str(max_rows)}.csv')


if __name__ == "__main__":
    model_name = 'bert_matching_pytorch'
    model_v = 1
    max_rows = 100000
    drop = False
    add_noise = False

    df = create_matching_dataset(read_gibberish_data(GIBBERISH))
    sample_df = get_predictions(df, model_name, model_v, max_rows, dropout=drop, with_noise=add_noise)
    save_results(sample_df, model_name, model_v, with_noise=add_noise, dropout=drop, max_rows=max_rows)
