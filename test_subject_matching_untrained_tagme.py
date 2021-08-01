from mapping_estimation import *

input_file_name = 'data/output_common2.csv'
df = pd.read_csv(input_file_name, delimiter='#', dtype=dtypes, usecols=list(dtypes))

evaluate_subject_matching('tagme', df) # untrained
