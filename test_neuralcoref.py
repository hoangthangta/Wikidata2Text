from neuralcoref_training import *

# output will be stored at "data/output_common2_trained.csv"
input_file_name = 'data/output_common2.csv'
df = pd.read_csv(input_file_name, delimiter='#', dtype=dtypes, usecols=list(dtypes))
get_trained_sentence(df)