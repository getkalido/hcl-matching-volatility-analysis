# hcl-matching-volatility-analysis
**PG data matching predictions &amp; analysis of volatility of matching models**


Creates the following matching datasets from HCL PG-data:
- gs://hcl-data/model_volatility_analyis_on_pg_data/bert_matching_old_2_sample_100000.csv
- gs://hcl-data/model_volatility_analyis_on_pg_data/bert_matching_pytorch_1_sample_100000.csv
- gs://hcl-data/model_volatility_analyis_on_pg_data/distilroberta_matching_pytorch_4_sample_100000.csv

*First 3 datasets are predictions of the same examples with slighlty different model versions as denoted by the filenames*

- gs://hcl-data/model_volatility_analyis_on_pg_data/with_dropout_distilroberta_matching_pytorch_4_sample_100000.csv

*Runs inference with dropout of hidden layer always on*

- gs://hcl-data/model_volatility_analyis_on_pg_data/with_noise_distilroberta_matching_pytorch_4_sample_100000.csv

*Contains slight random modfications to strings that are matched (adding suffix "expert" or "professional" or none to one of or both SKILL REQUIREMENT/SKILL entries)*
