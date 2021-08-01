from mapping_estimation import *

input_file_name = 'output_common2.csv'
df = pd.read_csv(input_file_name, delimiter='#', dtype=dtypes, usecols=list(dtypes))

evaluate_object_matching_type(df, 'dbpedia', 0)
evaluate_object_matching_type(df, 'dbpedia', 1)
evaluate_object_matching_type(df, 'dbpedia', 2)