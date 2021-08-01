from mapping_estimation import *

input_file_name = 'output_common2_trained.csv'

df = pd.read_csv(input_file_name, delimiter='#', dtype=dtypes2, usecols=list(dtypes2))

evaluate_subject_matching('wikifier', df, 'trained') # trained
