from typing import List
import math
import pandas as pd
from tqdm import tqdm
import numpy as np
import random
from mlflow_matching_model import MlFlowMatchingModel

from preprocessing_skills_master_data import create_matching_dataset, add_noise_to_text, get_predictions
from config import RAND_VOCAB, MLFLOW_TRACKING_PASSWORD, MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME

np.random.seed(7)
pd.set_option('display.max_columns', None)


def read_rand_vocab_data(filepath:str) -> List[str]:
    f = open(filepath, "r")
    tokens = [x.strip() for x in f.read().split("\n") if not "unused" in x and not "##" in x and x.strip().isalnum()]
    left_n = np.random.randint(1, 10)
    right_n = np.random.randint(1, 10)
    c = 0
    l = []
    while c <= 10000:
        need = ""
        offer = ""
        for x in range(left_n):
            need += " "
            need += random.choice(tokens)
        for y in range(right_n):
            offer += " "
            offer += random.choice(tokens)
        l.append((need.replace("\n", " "), offer.replace("\n", " ")))
        c += 1
    return l


def save_results(results_df: pd.DataFrame, model_name: str, model_v: int, with_noise: bool=False, dropout: bool=False, max_rows: int=10000):
    if dropout:
        results_df.to_csv(f'./predictions_rand_vocab/with_single_dropout_{model_name}_{str(model_v)}_sample_{str(max_rows)}.csv')
    elif with_noise:
        results_df.to_csv(f'./predictions_rand_vocab/with_noise_{model_name}_{str(model_v)}_sample_{str(max_rows)}.csv')
    else:
        results_df.to_csv(f'./predictions_rand_vocab/{model_name}_{str(model_v)}_sample_{str(max_rows)}.csv')


if __name__ == "__main__":
    model_name = 'distilroberta_matching_pytorch'
    model_v = 4
    max_rows = 100000
    drop = False
    add_noise = False
    df = create_matching_dataset(read_rand_vocab_data(RAND_VOCAB))
    sample_df = get_predictions(df, model_name, model_v, max_rows, dropout=drop, with_noise=add_noise)
    save_results(sample_df, model_name, model_v, with_noise=add_noise, dropout=drop, max_rows=max_rows)
