from recbole.quick_start import run_recbole

run_recbole(model='SASRec', 
            dataset='yelp2022',
            config_file_list=['config_files/yelp_simple.yaml'],
            config_dict={'data_path': './datasets'})
