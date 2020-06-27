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


def isEnglish(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True


def read_rand_vocab_data(filepath:str, max_rows: int=10000) -> List[str]:
    f = open(filepath, "r")
    tokens = [x.strip() for x in f.read().split("\n") if not "unused" in x and not "##" in x and x.strip().isalnum() and isEnglish(x)]
    n = np.random.randint(1, 3)
    c = 0
    l = []
    while c <= max_rows:
        s = ""
        for x in range(n):
            s += " "
            s += random.choice(tokens)
        l.append(s.replace("\n", " ").strip())
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
    model_name = 'bert_matching_pytorch'
    model_v = 1
    max_rows = 5000
    drop = False
    add_noise = False
    df = create_matching_dataset(read_rand_vocab_data(RAND_VOCAB, max_rows=max_rows))
    sample_df = get_predictions(df, model_name, model_v, max_rows, dropout=drop, with_noise=add_noise)
    save_results(sample_df, model_name, model_v, with_noise=add_noise, dropout=drop, max_rows=max_rows)
