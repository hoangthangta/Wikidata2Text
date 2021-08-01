from mapping_estimation import *

input_file_name = 'data/output_common2.csv'
df = pd.read_csv(input_file_name, delimiter='#', dtype=dtypes, usecols=list(dtypes))

evaluate_subject_matching_type(df, 'dbpedia', 0) # untrained
evaluate_subject_matching_type(df, 'dbpedia', 1) # untrained
evaluate_subject_matching_type(df, 'dbpedia', 2) # untrained
