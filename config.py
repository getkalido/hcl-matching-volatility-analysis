import os

GS_SKILL_MASTER = "gs://hcl-data/csv-data/SkillMaster/ERS_SkillsMaster_21Nov19.csv"
MLFLOW_TRACKING_PASSWORD = os.environ.get("MLFLOW_TRACKING_PASSWORD", "")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "")
MLFLOW_TRACKING_USERNAME = os.environ.get("MLFLOW_TRACKING_USERNAME", "")
GIBBERISH = "datasets-Gibberish.csv"
RAND_VOCAB = "vocab.txt"
