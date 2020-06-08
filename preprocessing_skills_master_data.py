from typing import List
import pandas as pd
from tqdm import tqdm
import numpy as np
from mlflow_matching_model import MlFlowMatchingModel

from config import GS_SKILL_MASTER, MLFLOW_TRACKING_PASSWORD, MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME

np.random.seed(7)
pd.set_option('display.max_columns', None)


def download_from_gcp(gs_path: str) -> List[str]:
    df = pd.read_csv(gs_path)
    skill = list(df['Skill'].unique())
    skillarea = list(df['SkillArea'].unique())
    subskill = list(df['FE/SubSkill'].unique())
    subcategory = list(df['SubCategory'].unique())
    l = skill + subskill + skillarea + subcategory
    return [x for x in l if not x.strip().lower().startswith('obsolete')]


def create_matching_dataset(l: List[str]) -> pd.DataFrame:
    tuples_l = []
    for x in l:
        for y in l:
            tuples_l.append(
                ("[SKILL_REQUIREMNENT] " + x.strip(), "[SKILL] " + y.strip())
            )
    return pd.DataFrame(tuples_l, columns =['need', 'offer'])


def add_noise_to_text(s: str) -> str:
    l = s.split(" ")
    prefix = l[0]
    new_s = " ".join(l[1:])
    n = np.random.randint(1, 9)
    if n <= 3:
        return prefix.strip() + " " + new_s.strip()
    elif n >= 7:
        return prefix.strip() + " " + new_s.strip() + " expert"
    else:
        return prefix.strip() + " professional " + new_s.strip()


def get_predictions(df: pd.DataFrame, model_name: str, model_v: int, max_rows: int=100000, dropout: bool=False, with_noise: bool=False) -> pd.DataFrame:
    model = MlFlowMatchingModel(model_name, model_v) # model name, version
    print("Model methods available:", dir(model))
    if dropout:
        model.load()
        model.model.model.train()
        model.model.model.config.attention_probs_dropout_prob = 0.0
        model.model.model.config.hidden_dropout_prob = 1.0
    pbas = []
    should_match = []
    sample_df = df.sample(frac=1).head(max_rows)
    if with_noise:
        sample_df['offer'] = sample_df['offer'].apply(add_noise_to_text)
        sample_df['need'] = sample_df['need'].apply(add_noise_to_text)
    sample_df['should_match'] = model.predict(sample_df["need"], sample_df["offer"])
    sample_df['pbas'] = model.predict_match_pba(sample_df["need"], sample_df["offer"])
    return sample_df


def save_results(results_df: pd.DataFrame, model_name: str, model_v: int, with_noise: bool=False, dropout: bool=False, max_rows: int=10000):
    if dropout:
        results_df.to_csv(f'./predictions/with_dropout_{model_name}_{str(model_v)}_sample_{str(max_rows)}.csv')
    elif with_noise:
        results_df.to_csv(f'./predictions/with_noise_{model_name}_{str(model_v)}_sample_{str(max_rows)}.csv')
    else:
        results_df.to_csv(f'./predictions/{model_name}_{str(model_v)}_sample_{str(max_rows)}.csv')


if __name__ == "__main__":
    model_name = 'distilroberta_matching_pytorch'
    model_v = 4
    max_rows = 100000
    drop = False
    add_noise = True

    df = create_matching_dataset(download_from_gcp(GS_SKILL_MASTER))
    sample_df = get_predictions(df, model_name, model_v, max_rows, dropout=drop, with_noise=add_noise)
    save_results(sample_df, model_name, model_v, with_noise=add_noise, dropout=drop, max_rows=max_rows)
