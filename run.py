from recbole.quick_start import run_recbole

run_recbole(
    model='SASRec',
    dataset='ml-1m',                               # must match folder & file prefix
    config_file_list=['config_files/ml.yaml'],     # your YAML
    config_dict={'data_path': './datasets'}        # parent of the ml-1m folder
)
