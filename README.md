# Wikidata2Text
We do the task of translating Wikidata claims or statements, organized as a set of triples or quadruples to Wikipedia sentences. This paper will be submitted to http://www.semantic-web-journal.net/ soon. Please check the guideline to follow when we publish our paper. The code is mainly about the data mapping process.

# Data
We have two folders for the data:
* Our_data: is the folder containing all of our data for the paper.
* Data: this folder is for the experiment, we already set some data files for testing purposes here.

# How to collect data
At first, we have to collect the pair of (item, page) via Wikidata query server. In this project, we store several files for Wikidata properties as following:
* data/p108.csv
* data/p166.csv
* data/p26.csv
* data/p39.csv
* data/p54.csv
* data/p69.csv

Open file "collect_data.py" and check these lines before running the command: `python collect_data.py`

```
file_name = 'p54' # predicate property
file_name_output = 'data/output_' + file_name + '.csv'  # the output url
page_list = read_from_csv_file('data/' + file_name + '.csv',  ',', 1) # the input url
page_list = sorted(list(set(page_list))) #filter repetitive pages and sort by alphabet
```


